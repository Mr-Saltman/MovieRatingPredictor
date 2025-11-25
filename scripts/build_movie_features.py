from __future__ import annotations
import argparse
import os
import sqlite3
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def load_tables(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load movies and genre tokens from tmdb.sqlite."""
    con = sqlite3.connect(db_path)
    try:
        movies = pd.read_sql(
            """
            SELECT id AS movie_rowid,
                   tmdb_id,
                   title,
                   overview,
                   COALESCE(runtime, 0)        AS runtime,
                   COALESCE(vote_average, 0)   AS vote_avg,
                   COALESCE(vote_count, 0)     AS vote_count,
                   COALESCE(popularity, 0)     AS popularity,
                   release_date
            FROM movies
            """,
            con,
        )

        genres = pd.read_sql(
            """
            SELECT mg.movie_id AS movie_rowid,
                   g.name      AS token
            FROM movie_genres mg
            JOIN genres g ON mg.genre_id = g.id
            """,
            con,
        )
    finally:
        con.close()

    return movies, genres


def bag_tokens(df: pd.DataFrame, movies: pd.DataFrame) -> pd.Series:
    """
    Turn (movie_rowid, token) rows into a single 'token1 token2 ...' string per movie.
    Always returns strings (no NaNs), indexed by movie_rowid.
    """
    if df.empty:
        return pd.Series([""] * len(movies), index=movies["movie_rowid"])

    bag = (
        df.groupby("movie_rowid")["token"]
          .apply(lambda xs: " ".join(sorted({str(x) for x in xs if pd.notna(x)})))
    )
    bag = bag.reindex(movies["movie_rowid"]).fillna("")
    return bag.astype(str)


def build_numeric_features(movies: pd.DataFrame) -> np.ndarray:
    """Build numeric feature matrix for each movie."""
    # recency: newer movies → larger value
    rel_dates = pd.to_datetime(movies["release_date"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")

    # age in years (fallback: 100 years for missing dates)
    age_years = ((now - rel_dates).dt.total_seconds() / (365.25 * 24 * 3600))
    age_years = age_years.fillna(100.0)

    inv_recency = 1.0 / (1.0 + age_years)

    num = np.c_[
        movies["runtime"].to_numpy(dtype=np.float32),
        movies["vote_avg"].to_numpy(dtype=np.float32),
        np.log1p(movies["vote_count"]).to_numpy(dtype=np.float32),
        movies["popularity"].to_numpy(dtype=np.float32),
        inv_recency.to_numpy(dtype=np.float32),
    ]
    return num


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build simple movie features from tmdb.sqlite (genres + numeric)."
    )
    ap.add_argument("--db", default="tmdb.sqlite", help="SQLite DB path")
    ap.add_argument(
        "--max-features",
        type=int,
        default=50,
        help="TF-IDF max features (genres are few anyway)",
    )
    ap.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="TF-IDF min document frequency",
    )
    ap.add_argument(
        "--out",
        default="feature_store/movie_features.joblib",
        help="Output joblib path",
    )
    args = ap.parse_args()

    os.makedirs("feature_store", exist_ok=True)

    movies, genres = load_tables(args.db)

    # --- text features: genres only ---
    g_text = bag_tokens(genres, movies)

    # clean to a simple string array
    corpus = (
        g_text.astype("string")
        .fillna("")
        .str.strip()
        .replace("nan", "")  # handle literal "nan" strings if any
        .to_numpy()
    )

    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        lowercase=True,
        strip_accents="unicode",
    )
    X_tfidf = tfidf.fit_transform(corpus)

    # --- numeric features ---
    num = build_numeric_features(movies)

    payload = {
        "movies": movies.reset_index(drop=True),
        "X_tfidf": X_tfidf,
        "num": num,
        "tfidf": tfidf,
        "feature_names": {
            "text": "genres",
            "numeric": ["runtime", "vote_avg", "log1p_vote_count", "popularity", "inv_recency"],
        },
    }

    joblib.dump(payload, args.out)
    print(
        f"✓ saved {args.out} "
        f"({X_tfidf.shape[0]} movies x {X_tfidf.shape[1]} genre-features, "
        f"{num.shape[1]} numeric features)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())