from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re


def load_movielens(ml_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MovieLens movies, ratings, tags (+ links for tmdb_id)."""
    movies = pd.read_csv(ml_dir / "movies.csv")
    ratings = pd.read_csv(ml_dir / "ratings.csv")

    tags_path = ml_dir / "tags.csv"
    if tags_path.exists():
        tags = pd.read_csv(tags_path)
    else:
        tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    links = pd.read_csv(ml_dir / "links.csv")
    links["movieId"] = links["movieId"].astype(int)
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)

    movies["movieId"] = movies["movieId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    if not tags.empty:
        tags["movieId"] = tags["movieId"].astype(int)
        tags["tag"] = tags["tag"].astype(str)

    # Attach tmdb_id
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    movies = movies.rename(columns={"tmdbId": "tmdb_id"})

    return movies, ratings, tags


def extract_year_from_title(title: str) -> float | None:
    """
    MovieLens titles often look like 'Toy Story (1995)'.
    Extract the year as a float, or NaN if not found.
    """
    if not isinstance(title, str):
        return np.nan
    m = re.search(r"\((\d{4})\)\s*$", title)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def build_numeric_features(movies: pd.DataFrame, ratings: pd.DataFrame) -> np.ndarray:
    """
    Build numeric features for each movie:

      - global_mean_rating (MovieLens mean rating per movie)
      - log1p_rating_count (log(1 + #ratings))
      - year_norm (release year normalized to [0, 1])

    Returns:
      num: ndarray of shape (n_movies, 3)
    """
    # Aggregate stats from ratings
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n_ratings", "mean": "mean_rating"})
        .reset_index()
    )

    movies = movies.merge(stats, on="movieId", how="left")

    # Fill missing stats
    global_mean = ratings["rating"].mean()
    movies["n_ratings"] = movies["n_ratings"].fillna(0.0)
    movies["mean_rating"] = movies["mean_rating"].fillna(global_mean)

    # Extract year from title
    movies["year"] = movies["title"].apply(extract_year_from_title)
    year_min = movies["year"].min(skipna=True)
    year_max = movies["year"].max(skipna=True)

    if pd.isna(year_min) or pd.isna(year_max) or year_min == year_max:
        # If we don't have usable years, just put 0.5 for everyone
        movies["year_norm"] = 0.5
    else:
        movies["year_norm"] = (movies["year"] - year_min) / (year_max - year_min)
        movies["year_norm"] = movies["year_norm"].fillna(0.5)

    # Build numeric feature matrix
    mean_rating = movies["mean_rating"].to_numpy(dtype=np.float32)
    log_n = np.log1p(movies["n_ratings"].to_numpy(dtype=np.float32))
    year_norm = movies["year_norm"].to_numpy(dtype=np.float32)

    num = np.c_[mean_rating, log_n, year_norm]
    return num, movies


def build_text_corpus(movies: pd.DataFrame, tags: pd.DataFrame) -> np.ndarray:
    """
    Build a text corpus per movie by combining:

      - genres from movies.csv
      - tags from tags.csv

    Returns:
      corpus: np.ndarray of shape (n_movies,) with one string per movie.
    """
    # Genres: 'Action|Comedy' -> 'action comedy'
    genres_clean = (
        movies["genres"]
        .fillna("")
        .replace("(no genres listed)", "", regex=False)
        .str.replace("|", " ", regex=False)
        .str.lower()
        .str.strip()
    )

    # Tags: group by movieId -> "tag1 tag2 ..."
    if tags.empty:
        tag_text = pd.Series([""] * len(movies), index=movies.index, dtype="string")
    else:
        tags["tag"] = tags["tag"].astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
        tag_text = (
            tags.groupby("movieId")["tag"]
            .apply(lambda xs: " ".join(xs))
            .reindex(movies["movieId"])
            .fillna("")
            .astype("string")
            .reset_index(drop=True)
        )

    corpus = (genres_clean + " " + tag_text).str.strip()
    corpus = corpus.fillna("").replace("nan", "")
    return corpus.to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build movie features directly from MovieLens (genres + tags + rating stats)."
    )
    ap.add_argument(
        "--ml-dir",
        type=Path,
        required=True,
        help="Path to MovieLens directory (contains movies.csv, ratings.csv, tags.csv)",
    )
    ap.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="TF-IDF max features",
    )
    ap.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="TF-IDF min document frequency",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("feature_store/movie_features.joblib"),
        help="Output joblib path",
    )
    args = ap.parse_args()

    os.makedirs(args.out.parent, exist_ok=True)

    print(f"Loading MovieLens from {args.ml_dir} ...")
    movies, ratings, tags = load_movielens(args.ml_dir)
    print(f"Loaded {len(movies):,} movies, {len(ratings):,} ratings, {len(tags):,} tags.")

    # Set movie_rowid == movieId (so it matches interactions)
    movies = movies.copy()
    movies["movie_rowid"] = movies["movieId"].astype(int)

    # --- numeric features ---
    print("Building numeric features...")
    num, movies_with_stats = build_numeric_features(movies, ratings)

    # --- text features (genres + tags) ---
    print("Building text corpus (genres + tags) and TF-IDF features...")
    corpus = build_text_corpus(movies_with_stats, tags)

    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
    )
    X_tfidf = tfidf.fit_transform(corpus)

    payload = {
        "movies": movies_with_stats.reset_index(drop=True),
        "X_tfidf": X_tfidf,
        "num": num,
        "tfidf": tfidf,
        "feature_names": {
            "text": "genres+tags",
            "numeric": [
                "global_mean_rating",
                "log1p_rating_count",
                "year_norm",
            ],
        },
    }

    joblib.dump(payload, args.out)
    print(
        f"Saved {args.out} "
        f"({X_tfidf.shape[0]} movies x {X_tfidf.shape[1]} text-features, "
        f"{num.shape[1]} numeric features)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())