"""
Build interactions from MovieLens (+ optional TMDB filter)

Reads:
  - MovieLens ratings.csv, links.csv
  - (optional) exports/movies.parquet  OR tmdb.sqlite (to restrict to scraped movies)

Writes:
  - dataset/interactions.parquet (or .csv)
      columns: userId, movie_rowid, tmdb_id, rating, timestamp, title (if available)

Usage examples:
  python movielens_to_interactions.py --ml-dir ml-latest-small \
      --movies-parquet exports/movies.parquet \
      --min-user-ratings 5 --min-movie-ratings 5 --scale-10 \
      --out dataset/interactions.parquet

  python movielens_to_interactions.py --ml-dir ml-latest \
      --sqlite tmdb.sqlite --out dataset/interactions.csv
"""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd


def load_movielens(ml_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Read loosely; normalize dtypes after
    ratings = pd.read_csv(ml_dir / "ratings.csv", low_memory=False)
    links   = pd.read_csv(ml_dir / "links.csv", low_memory=False)

    # --- Ratings normalization ---
    # Coerce ids
    ratings["userId"]  = pd.to_numeric(ratings["userId"], errors="coerce").astype("Int64")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce").astype("Int64")

    # Handle timestamp that may be either unix seconds OR datetime strings
    if "timestamp" in ratings.columns:
        ts = ratings["timestamp"]

        if pd.api.types.is_numeric_dtype(ts):
            # Already numeric (likely unix seconds)
            ts_num = pd.to_numeric(ts, errors="coerce").astype("Int64")
        else:
            # Try parse as datetime strings
            ts_dt = pd.to_datetime(ts, errors="coerce", utc=True, infer_datetime_format=True)
            if ts_dt.notna().mean() > 0.9:
                # Convert to unix seconds
                ts_num = (ts_dt.view("int64") // 1_000_000_000).astype("Int64")
            else:
                # Fallback: try numeric coercion anyway
                ts_num = pd.to_numeric(ts, errors="coerce").astype("Int64")

        ratings["timestamp"] = ts_num

    # Drop any rows with missing critical ids or timestamps
    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
    ratings["userId"]  = ratings["userId"].astype("int32")
    ratings["movieId"] = ratings["movieId"].astype("int32")
    ratings["rating"]  = pd.to_numeric(ratings["rating"], errors="coerce").astype("float32")
    ratings["timestamp"] = ratings["timestamp"].astype("int64")  # unix seconds

    # --- Links normalization ---
    links["movieId"] = pd.to_numeric(links["movieId"], errors="coerce").astype("Int64")
    # tmdbId can be missing; keep as nullable then handle downstream
    if "tmdbId" in links.columns:
        links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    else:
        # Some Kaggle dumps use 'tmdbid' or lack it entirely
        cand = [c for c in links.columns if c.lower() == "tmdbid"]
        if cand:
            links["tmdbId"] = pd.to_numeric(links[cand[0]], errors="coerce").astype("Int64")
        else:
            links["tmdbId"] = pd.Series(pd.array([pd.NA]*len(links), dtype="Int64"))

    # Keep only rows with valid ids
    links = links.dropna(subset=["movieId"]).copy()
    links["movieId"] = links["movieId"].astype("int32")

    return ratings, links


def load_tmdb_movies_from_parquet(path: Path) -> pd.DataFrame:
    mv = pd.read_parquet(path)
    required = {"movie_rowid", "tmdb_id", "title"}
    missing = required - set(mv.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return mv[["movie_rowid", "tmdb_id", "title"]].copy()


def load_tmdb_movies_from_sqlite(sqlite_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(sqlite_path)
    try:
        mv = pd.read_sql(
            """
            SELECT id AS movie_rowid, tmdb_id, title
            FROM movies
            """,
            con,
        )
    finally:
        con.close()
    return mv


def apply_min_counts(df: pd.DataFrame, min_user: int, min_movie: int) -> pd.DataFrame:
    if min_user > 0:
        uc = df["userId"].value_counts()
        keep_users = uc[uc >= min_user].index
        df = df[df["userId"].isin(keep_users)]
    if min_movie > 0:
        mc = df["movie_rowid"].value_counts()
        keep_movies = mc[mc >= min_movie].index
        df = df[df["movie_rowid"].isin(keep_movies)]
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Convert MovieLens into interactions mapped to TMDB.")
    p.add_argument("--ml-dir", required=True, type=Path, help="Path to MovieLens directory (e.g., ml-latest-small)")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--movies-parquet", type=Path, help="exports/movies.parquet to filter & attach titles")
    grp.add_argument("--sqlite", type=Path, help="tmdb.sqlite to filter & attach titles")
    p.add_argument("--min-user-ratings", type=int, default=0, help="Filter users with fewer ratings")
    p.add_argument("--min-movie-ratings", type=int, default=0, help="Filter movies with fewer ratings")
    p.add_argument("--scale-10", action="store_true", help="Rescale MovieLens 0.5–5.0 → 1–10")
    p.add_argument("--out", required=True, type=Path, help="Output file (.parquet or .csv)")
    args = p.parse_args()

    # 1) Load MovieLens
    ratings, links = load_movielens(args.ml_dir)

    # 2) Merge ratings with tmdbId
    df = ratings.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    df = df.dropna(subset=["tmdbId"]).copy()
    df["tmdbId"] = df["tmdbId"].astype("int64")

    # 3) Load TMDB movie inventory (optional, to limit to what you scraped)
    tmdb_movies: Optional[pd.DataFrame] = None
    if args.movies_parquet:
        tmdb_movies = load_tmdb_movies_from_parquet(args.movies_parquet)
    elif args.sqlite:
        tmdb_movies = load_tmdb_movies_from_sqlite(args.sqlite)

    if tmdb_movies is not None:
        # Join on tmdb_id and drop anything we don't have locally
        df = df.merge(
            tmdb_movies.rename(columns={"tmdb_id": "tmdbId"}),
            on="tmdbId",
            how="inner",
        )
        df = df.rename(columns={"tmdbId": "tmdb_id"})
        # Keep tidy columns
        cols = ["userId", "movie_rowid", "tmdb_id", "rating", "timestamp", "title"]
        df = df[cols]
    else:
        # No local movie inventory provided: keep tmdbId only (movie_rowid/title unknown)
        df = df.rename(columns={"tmdbId": "tmdb_id"})
        cols = ["userId", "tmdb_id", "rating", "timestamp"]
        df = df[cols]

    # 4) Optional rating rescale to 1–10
    if args.scale_10:
        df["rating"] = (df["rating"] * 2.0).clip(lower=1.0, upper=10.0)

    # 5) Apply min-count filters if we have movie_rowid; otherwise filter by tmdb_id counts
    if "movie_rowid" in df.columns:
        df = apply_min_counts(df, args.min_user_ratings, args.min_movie_ratings)
    else:
        # Fallback: min-movie-ratings works on tmdb_id
        if args.min_user_ratings > 0:
            uc = df["userId"].value_counts()
            df = df[df["userId"].isin(uc[uc >= args.min_user_ratings].index)]
        if args.min_movie_ratings > 0:
            mc = df["tmdb_id"].value_counts()
            df = df[df["tmdb_id"].isin(mc[mc >= args.min_movie_ratings].index)]

    # 6) Sort (nice for downstream diffs)
    sort_cols = ["userId", "movie_rowid" if "movie_rowid" in df.columns else "tmdb_id", "timestamp"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # 7) Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".parquet":
        df.to_parquet(args.out, index=False)
    elif args.out.suffix.lower() == ".csv":
        df.to_csv(args.out, index=False)
    else:
        raise ValueError("Output file must be .parquet or .csv")

    # 8) Quick report
    n_users = df["userId"].nunique()
    n_items = df["movie_rowid"].nunique() if "movie_rowid" in df.columns else df["tmdb_id"].nunique()
    print(f"✓ wrote {len(df):,} interactions for {n_users:,} users and {n_items:,} movies → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())