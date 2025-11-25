from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def load_movielens(ml_dir: Path):
    """Load MovieLens ratings + links (basic cleanup)."""
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    links = pd.read_csv(ml_dir / "links.csv")

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["timestamp"] = ratings["timestamp"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    links["movieId"] = links["movieId"].astype(int)
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)

    return ratings, links


def load_tmdb_movies(sqlite_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(sqlite_path)
    try:
        movies = pd.read_sql(
            "SELECT id AS movie_rowid, tmdb_id, title FROM movies",
            con,
        )
    finally:
        con.close()
    return movies


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert MovieLens ratings into interactions linked to TMDB movies."
    )
    ap.add_argument("--ml-dir", type=Path, required=True,
                    help="Path to MovieLens directory (ratings.csv, links.csv)")
    ap.add_argument("--sqlite", type=Path, required=True,
                    help="Path to tmdb.sqlite")
    ap.add_argument("--min-user-ratings", type=int, default=0,
                    help="Drop users with fewer ratings than this")
    ap.add_argument("--min-movie-ratings", type=int, default=0,
                    help="Drop movies with fewer ratings than this")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output file (.parquet or .csv)")
    args = ap.parse_args()

    ratings, links = load_movielens(args.ml_dir)
    tmdb_movies = load_tmdb_movies(args.sqlite)

    # 1) attach tmdbId to ratings
    df = ratings.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    df = df.dropna(subset=["tmdbId"]).copy()
    df["tmdb_id"] = df["tmdbId"].astype(int)

    # 2) join with local TMDB movies (keep only movies we scraped)
    df = df.merge(tmdb_movies, on="tmdb_id", how="inner")

    # keep only columns we care about
    df = df[["userId", "movie_rowid", "tmdb_id", "rating", "timestamp", "title"]]

    # 3) apply min counts
    if args.min_user_ratings > 0:
        user_counts = df["userId"].value_counts()
        keep_users = user_counts[user_counts >= args.min_user_ratings].index
        df = df[df["userId"].isin(keep_users)]

    if args.min_movie_ratings > 0:
        movie_counts = df["movie_rowid"].value_counts()
        keep_movies = movie_counts[movie_counts >= args.min_movie_ratings].index
        df = df[df["movie_rowid"].isin(keep_movies)]

    # 4) sort for nicer inspection
    df = df.sort_values(["userId", "movie_rowid", "timestamp"]).reset_index(drop=True)

    # 5) write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".csv":
        df.to_csv(args.out, index=False)
    elif args.out.suffix.lower() == ".parquet":
        df.to_parquet(args.out, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")

    print(
        f"✓ wrote {len(df):,} interactions for "
        f"{df['userId'].nunique():,} users and "
        f"{df['movie_rowid'].nunique():,} movies → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())