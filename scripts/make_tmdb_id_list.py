"""
Create a list of TMDB movie IDs to scrape.

- Reads MovieLens ratings.csv and links.csv
- Counts how many ratings each movie has
- Keeps the TOP N most-rated movies
- Writes their tmdbId (one per line) to tmdb_ids_to_scrape.txt

Usage:
  python make_tmdb_id_list.py \
      --ml-dir ml-latest-small \
      --top-n 5000 \
      --out tmdb_ids_to_scrape.txt
"""

import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description="Create TMDB id list from MovieLens.")
    ap.add_argument("--ml-dir", type=Path, required=True,
                    help="Path to MovieLens directory (contains ratings.csv, links.csv)")
    ap.add_argument("--top-n", type=int, default=5000,
                    help="How many popular movies to keep")
    ap.add_argument("--out", type=Path, default=Path("tmdb_ids_to_scrape.txt"),
                    help="Output text file (one tmdbId per line)")
    args = ap.parse_args()

    ratings = pd.read_csv(args.ml_dir / "ratings.csv")
    links   = pd.read_csv(args.ml_dir / "links.csv")

    # Count ratings per MovieLens movieId
    counts = (
        ratings.groupby("movieId")
               .size()
               .rename("n")
               .reset_index()
    )

    # Attach tmdbId
    df = counts.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    df = df.dropna(subset=["tmdbId"]).copy()
    df["tmdbId"] = df["tmdbId"].astype(int)

    # Take top N by rating count
    top = df.sort_values("n", ascending=False).head(args.top_n)

    # Write plain list of tmdbIds
    args.out.parent.mkdir(parents=True, exist_ok=True)
    top["tmdbId"].to_csv(args.out, index=False, header=False)

    print(f"✓ wrote {len(top)} TMDB IDs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())