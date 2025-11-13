from __future__ import annotations

import argparse
import sqlite3
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


def load_tables(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        movies = pd.read_sql(
            """
            SELECT id AS movie_rowid, tmdb_id, title, overview,
                   COALESCE(runtime,0) AS runtime,
                   COALESCE(vote_average,0) AS vote_avg,
                   COALESCE(vote_count,0)  AS vote_count,
                   COALESCE(popularity,0)  AS popularity,
                   release_date, original_language
            FROM movies
            """,
            con,
        )

        genres = pd.read_sql(
            """
            SELECT mg.movie_id AS movie_rowid, g.name AS token
            FROM movie_genres mg JOIN genres g ON mg.genre_id=g.id
            """,
            con,
        )

        keywords = pd.read_sql(
            """
            SELECT mk.movie_id AS movie_rowid, k.name AS token
            FROM movie_keywords mk JOIN keywords k ON mk.keyword_id=k.id
            """,
            con,
        )

        # Optional: add a few cast/director tokens (kept small to avoid exploding vocab)
        cast_top = pd.read_sql(
            """
            SELECT c.movie_id AS movie_rowid, p.name AS token
            FROM cast c JOIN people p ON c.person_id=p.id
            WHERE c.cast_order <= 3
            """,
            con,
        )

        crew_core = pd.read_sql(
            """
            SELECT cr.movie_id AS movie_rowid, p.name AS token
            FROM crew cr JOIN people p ON cr.person_id=p.id
            WHERE cr.job IN ('Director')
            """,
            con,
        )
    finally:
        con.close()

    return movies, genres, keywords, cast_top, crew_core


def bag_tokens(df: pd.DataFrame, movies: pd.DataFrame, col_name: str = "token") -> pd.Series:
    """
    Turn rows (movie_rowid, token) into a space-joined string per movie.
    Ensures index aligns with movies.movie_rowid and fills missing with "".
    """
    if df.empty:
        return pd.Series([""] * len(movies), index=movies["movie_rowid"])
    bag = (
        df.groupby("movie_rowid")[col_name]
          .apply(lambda xs: " ".join(sorted({str(x) for x in xs if pd.notna(x)})))
    )
    # align to full movie index, fill missing with ""
    return bag.reindex(movies["movie_rowid"]).fillna("")


def main():
    ap = argparse.ArgumentParser(description="Build TMDB movie features")
    ap.add_argument("--db", default="tmdb.sqlite", help="SQLite DB path")
    ap.add_argument("--max-features", type=int, default=8000, help="TF-IDF max features")
    ap.add_argument("--min-df", type=int, default=3, help="TF-IDF min document frequency")
    ap.add_argument("--include-names", action="store_true",
                    help="Include top-3 cast and director names as text tokens")
    args = ap.parse_args()

    os.makedirs("feature_store", exist_ok=True)

    movies, genres, keywords, cast_top, crew_core = load_tables(args.db)

    # --- Text corpus construction (robust to NaNs) ---
    g_text = bag_tokens(genres, movies)
    k_text = bag_tokens(keywords, movies)

    if args.include_names:
        cast_text = bag_tokens(cast_top, movies)
        dir_text  = bag_tokens(crew_core, movies)
    else:
        cast_text = pd.Series([""] * len(movies), index=movies["movie_rowid"])
        dir_text  = pd.Series([""] * len(movies), index=movies["movie_rowid"])

    overview = movies["overview"].fillna("").astype(str).values
    corpus = (
        g_text.astype(str).values + " " +
        k_text.astype(str).values + " " +
        cast_text.astype(str).values + " " +
        dir_text.astype(str).values + " " +
        overview
    )

    # Guard: replace any lingering "nan" literals with empty
    corpus = np.array([c if isinstance(c, str) else "" for c in corpus], dtype=object)

    # --- TF-IDF ---
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    X_tfidf = tfidf.fit_transform(corpus)

    # --- Numeric features ---
    rel = pd.to_datetime(movies["release_date"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")

    # years since release
    recency_years = ((now - rel).dt.total_seconds() / (365.25 * 24 * 3600))
    recency_years = pd.Series(recency_years).fillna(100.0)  # default for missing dates
    inv_recency = 1.0 / (1.0 + recency_years)

    release_year = rel.dt.year.astype("Int64").fillna(0).astype("int16")

    num = np.c_[
        movies["runtime"].fillna(0).to_numpy(dtype=np.float32),
        movies["vote_avg"].fillna(0).to_numpy(dtype=np.float32),
        np.log1p(movies["vote_count"].fillna(0)).to_numpy(dtype=np.float32),
        movies["popularity"].fillna(0).to_numpy(dtype=np.float32),
        release_year.to_numpy(dtype=np.int16),
        inv_recency.to_numpy(dtype=np.float32),
    ]

    # Persist
    payload = {
        "movies": movies.reset_index(drop=True),
        "X_tfidf": X_tfidf,
        "num": num,
        "tfidf": tfidf,
        "feature_names": {
            "numeric": ["runtime", "vote_avg", "log1p_vote_count", "popularity", "release_year", "inv_recency"]
        },
        "build_args": vars(args),
    }
    joblib.dump(payload, "feature_store/movie_features.joblib")
    print(f"✓ Saved feature_store/movie_features.joblib "
          f"({X_tfidf.shape[0]} movies x {X_tfidf.shape[1]} vocab, {num.shape[1]} numeric features)")


if __name__ == "__main__":
    main()