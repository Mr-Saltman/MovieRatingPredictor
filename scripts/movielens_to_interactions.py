from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd


def load_movielens(ml_dir: Path):
    """Load MovieLens ratings + movies (basic cleanup)."""
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    movies = pd.read_csv(ml_dir / "movies.csv")

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["timestamp"] = ratings["timestamp"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    movies["movieId"] = movies["movieId"].astype(int)

    return ratings, movies


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert MovieLens ratings into interactions (no TMDB)."
    )
    ap.add_argument(
        "--ml-dir",
        type=Path,
        required=True,
        help="Path to MovieLens directory (ratings.csv, movies.csv)",
    )
    ap.add_argument(
        "--min-user-ratings",
        type=int,
        default=0,
        help="Drop users with fewer ratings than this",
    )
    ap.add_argument(
        "--min-movie-ratings",
        type=int,
        default=0,
        help="Drop movies with fewer ratings than this",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output file (.csv)",
    )
    args = ap.parse_args()

    ratings, movies = load_movielens(args.ml_dir)

    # Attach titles
    df = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")

    # Define movie_rowid == movieId (to match movie_features.movies.movie_rowid)
    df["movie_rowid"] = df["movieId"].astype(int)

    # Keep only columns we care about
    df = df[["userId", "movie_rowid", "rating", "timestamp", "title"]]

    # Apply min counts
    if args.min_user_ratings > 0:
        user_counts = df["userId"].value_counts()
        keep_users = user_counts[user_counts >= args.min_user_ratings].index
        df = df[df["userId"].isin(keep_users)]

    if args.min_movie_ratings > 0:
        movie_counts = df["movie_rowid"].value_counts()
        keep_movies = movie_counts[movie_counts >= args.min_movie_ratings].index
        df = df[df["movie_rowid"].isin(keep_movies)]

    # Sort for nicer inspection
    df = df.sort_values(["userId", "movie_rowid", "timestamp"]).reset_index(drop=True)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".csv":
        df.to_csv(args.out, index=False)
    else:
        raise ValueError("Output must be .csv")

    print(
        f"Wrote {len(df):,} interactions for "
        f"{df['userId'].nunique():,} users and "
        f"{df['movie_rowid'].nunique():,} movies => {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())