"""
Build simple, interpretable movie features from tmdb.sqlite.

Text features:
  - ONLY genres (e.g. Action, Drama, Comedy)
Numeric features:
  - runtime
  - vote_avg
  - log1p_vote_count
  - popularity
  - inverse recency (newer movies have higher value)

Usage:
  python build_movie_features.py --db tmdb.sqlite
"""

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
    con = sqlite3.connect(db_path)
    try:
        movies = pd.read_sql(
            """
            SELECT id AS movie_rowid, tmdb_id, title, overview,
                   COALESCE(runtime, 0) AS runtime,
                   COALESCE(vote_average, 0) AS vote_avg,
                   COALESCE(vote_count, 0) AS vote_count,
                   COALESCE(popularity, 0) AS popularity,
                   release_date
            FROM movies
            """,
            con,
        )

        genres = pd.read_sql(
            """
            SELECT mg.movie_id AS movie_rowid, g.name AS token
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
    Always returns strings, no NaNs.
    """
    if df.empty:
        return pd.Series([""] * len(movies), index=movies["movie_rowid"])

    bag = (
        df.groupby("movie_rowid")["token"]
          .apply(lambda xs: " ".join(sorted({str(x) for x in xs if pd.notna(x)})))
    )
    bag = bag.reindex(movies["movie_rowid"]).fillna("")
    return bag.astype(str)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build simple movie features from tmdb.sqlite (genres + numeric).")
    ap.add_argument("--db", default="tmdb.sqlite", help="SQLite DB path")
    ap.add_argument("--max-features", type=int, default=50, help="TF-IDF max features (genres are few anyway)")
    ap.add_argument("--min-df", type=int, default=1, help="TF-IDF min document frequency")
    ap.add_argument("--out", default="feature_store/movie_features.joblib",
                    help="Output joblib path")
    args = ap.parse_args()

    os.makedirs("feature_store", exist_ok=True)

    movies, genres = load_tables(args.db)

    # --- text corpus: ONLY genres ---
    g_text = bag_tokens(genres, movies)
    corpus = g_text.values  # one genre string per movie

    # EXTRA safety: ensure all entries are real strings
    clean_corpus = []
    for c in corpus:
        if isinstance(c, str):
            if c.strip().lower() == "nan":
                clean_corpus.append("")
            else:
                clean_corpus.append(c)
        elif pd.isna(c):
            clean_corpus.append("")
        else:
            clean_corpus.append(str(c))

    clean_corpus = np.array(clean_corpus, dtype=object)

    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        stop_words=None,           # genres are short labels, we don't want to drop them
        lowercase=True,
        strip_accents="unicode",
    )
    X_tfidf = tfidf.fit_transform(clean_corpus)

    # --- numeric features ---
    rel = pd.to_datetime(movies["release_date"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")

    recency_years = ((now - rel).dt.total_seconds() / (365.25 * 24 * 3600))
    recency_years = recency_years.fillna(100.0)
    inv_recency = 1.0 / (1.0 + recency_years)

    num = np.c_[
        movies["runtime"].to_numpy(dtype=np.float32),
        movies["vote_avg"].to_numpy(dtype=np.float32),
        np.log1p(movies["vote_count"]).to_numpy(dtype=np.float32),
        movies["popularity"].to_numpy(dtype=np.float32),
        inv_recency.to_numpy(dtype=np.float32),
    ]

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
    print(f"✓ saved {args.out} "
          f"({X_tfidf.shape[0]} movies x {X_tfidf.shape[1]} genre-features, {num.shape[1]} numeric features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())