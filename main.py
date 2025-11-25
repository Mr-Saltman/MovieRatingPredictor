"""
Orchestrate the full movie recommender data pipeline.

Steps:
  1) make_tmdb_id_list.py
  2) tmdb_scraper_simple.py
  3) build_movie_features_simple.py
  4) movielens_to_interactions_simple.py
  5) build_user_profiles_simple.py

Usage examples:

  # Run all steps
  python main.py --all

  # Skip scraping (e.g., if tmdb.sqlite already exists)
  python main.py --all --skip-scrape

  # Just rebuild features and profiles
  python main.py --build-features --build-interactions --build-profiles
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# Default paths
ML_DIR = ROOT / "data" / "ml-32m"
TMDB_IDS = ROOT / "data" / "tmdb_ids_to_scrape.txt"
TMDB_DB = ROOT / "data" / "tmdb.sqlite"
INTERACTIONS = ROOT / "data" / "dataset" / "interactions.csv"
MOVIE_FEATURES = ROOT / "feature_store" / "movie_features.joblib"
USER_PROFILES = ROOT / "feature_store" / "user_profiles.joblib"


def run(cmd: list[str]) -> None:
    """Run a command and print it nicely."""
    print("\n$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def step_make_ids(top_n: int) -> None:
    TMDB_IDS.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "make_tmdb_id_list.py"),
        "--ml-dir", str(ML_DIR),
        "--top-n", str(top_n),
        "--out", str(TMDB_IDS),
    ]
    run(cmd)


def step_scrape_tmdb() -> None:
    TMDB_DB.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "tmdb_scraper.py"),
        "--ids-file", str(TMDB_IDS),
        "--db", str(TMDB_DB),
    ]
    run(cmd)


def step_build_movie_features() -> None:
    MOVIE_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "build_movie_features.py"),
        "--db", str(TMDB_DB),
        "--out", str(MOVIE_FEATURES),
    ]
    run(cmd)


def step_build_interactions(min_user: int, min_movie: int) -> None:
    INTERACTIONS.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "movielens_to_interactions.py"),
        "--ml-dir", str(ML_DIR),
        "--sqlite", str(TMDB_DB),
        "--min-user-ratings", str(min_user),
        "--min-movie-ratings", str(min_movie),
        "--out", str(INTERACTIONS),
    ]
    run(cmd)


def step_build_user_profiles(min_ratings: int) -> None:
    USER_PROFILES.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPTS / "build_user_profiles.py"),
        "--interactions", str(INTERACTIONS),
        "--features", str(MOVIE_FEATURES),
        "--min-ratings", str(min_ratings),
        "--out", str(USER_PROFILES),
    ]
    run(cmd)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the full data pipeline.")
    ap.add_argument("--all", action="store_true",
                    help="Run all steps in order")

    # individual steps
    ap.add_argument("--make-ids", action="store_true", help="Run: make_tmdb_id_list.py")
    ap.add_argument("--scrape", action="store_true", help="Run: tmdb_scraper_simple.py")
    ap.add_argument("--build-features", action="store_true", help="Run: build_movie_features_simple.py")
    ap.add_argument("--build-interactions", action="store_true", help="Run: movielens_to_interactions_simple.py")
    ap.add_argument("--build-profiles", action="store_true", help="Run: build_user_profiles_simple.py")

    # options
    ap.add_argument("--top-n", type=int, default=5000,
                    help="How many movies to select for scraping")
    ap.add_argument("--min-user-ratings", type=int, default=5,
                    help="Min ratings per user for interactions")
    ap.add_argument("--min-movie-ratings", type=int, default=5,
                    help="Min ratings per movie for interactions")
    ap.add_argument("--min-ratings-profile", type=int, default=5,
                    help="Min ratings per user to build a profile")

    # convenience toggles
    ap.add_argument("--skip-scrape", action="store_true",
                    help="Skip scraping step (use existing tmdb.sqlite)")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # If --all is set, turn on all steps (respecting skip-scrape)
    if args.all:
        args.make_ids = True
        args.scrape = not args.skip_scrape
        args.build_features = True
        args.build_interactions = True
        args.build_profiles = True

    # sanity checks
    if not ML_DIR.exists():
        print(f"ERROR: MovieLens directory not found: {ML_DIR}")
        print("Download ml-latest-small and put it in data/ml-latest-small/")
        return 1

    if args.make_ids:
        step_make_ids(top_n=args.top_n)

    if args.scrape:
        if not TMDB_IDS.exists():
            print(f"ERROR: TMDB id list not found: {TMDB_IDS}")
            print("Run with --make-ids first, or --all without --skip-scrape.")
            return 1
        step_scrape_tmdb()

    if args.build_features:
        if not TMDB_DB.exists():
            print(f"ERROR: tmdb.sqlite not found at {TMDB_DB}")
            return 1
        step_build_movie_features()

    if args.build_interactions:
        if not TMDB_DB.exists():
            print(f"ERROR: tmdb.sqlite not found at {TMDB_DB}")
            return 1
        step_build_interactions(
            min_user=args.min_user_ratings,
            min_movie=args.min_movie_ratings,
        )

    if args.build_profiles:
        if not INTERACTIONS.exists() or not MOVIE_FEATURES.exists():
            print("ERROR: interactions or movie_features not found.")
            print("Make sure previous steps ran successfully.")
            return 1
        step_build_user_profiles(min_ratings=args.min_ratings_profile)

    print("\n✓ Pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
