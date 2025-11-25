"""
Build simple user profiles from interactions + movie features.

- interactions: userId, movie_rowid, rating, timestamp (Parquet or CSV)
- movie_features: feature_store/movie_features.joblib from build_movie_features_simple.py
- For each user:
    - find all movies they rated
    - take the TF-IDF rows of those movies
    - compute a weighted average using the ratings (1–5) as weights
    - L2-normalize the result (so we can use dot-product / cosine later)

Usage:
  python build_user_profiles_simple.py \
      --interactions dataset/interactions.parquet \
      --features feature_store/movie_features.joblib \
      --min-ratings 5 \
      --out feature_store/user_profiles.joblib
"""

from __future__ import annotations
import argparse
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_interactions(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    elif path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("interactions must be .parquet or .csv")

    for col in ["userId", "movie_rowid", "rating"]:
        if col not in df.columns:
            raise ValueError(f"interactions is missing required column '{col}'")

    df["userId"] = df["userId"].astype(int)
    df["movie_rowid"] = df["movie_rowid"].astype(int)
    df["rating"] = df["rating"].astype(float)  # should be 1.0–5.0
    return df


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def build_profiles(
    inter: pd.DataFrame,
    movies_df: pd.DataFrame,
    X_tfidf: csr_matrix,
    min_ratings: int = 5,
) -> Dict[int, np.ndarray]:
    """
    Return {userId: profile_vector}.
    """
    # build map: movie_rowid -> row index in X_tfidf
    movie_to_idx = pd.Series(
        movies_df.index.values,
        index=movies_df["movie_rowid"],
    )

    # attach row_idx to interactions
    inter = inter.merge(
        movie_to_idx.rename("row_idx"),
        left_on="movie_rowid",
        right_index=True,
        how="inner",
    )
    inter["row_idx"] = inter["row_idx"].astype(int)

    profiles: Dict[int, np.ndarray] = {}

    for uid, grp in inter.groupby("userId"):
        if len(grp) < min_ratings:
            continue

        rows = grp["row_idx"].to_numpy(dtype=int)
        weights = grp["rating"].to_numpy(dtype=float).reshape(-1, 1)

        V = X_tfidf[rows]  # (n_ratings, n_features)
        # weighted sum: sum(w_i * v_i)
        num = (V.T @ weights)  # (n_features, 1)
        num = np.asarray(num).ravel()
        denom = weights.sum()
        if denom <= 0:
            continue

        centroid = num / denom
        profiles[int(uid)] = l2_normalize(centroid)

    return profiles


def main() -> int:
    ap = argparse.ArgumentParser(description="Build user profiles from interactions")
    ap.add_argument("--interactions", required=True,
                    help="Path to interactions (.parquet or .csv)")
    ap.add_argument("--features", default="feature_store/movie_features.joblib",
                    help="Path to movie feature store")
    ap.add_argument("--min-ratings", type=int, default=5,
                    help="Minimum ratings per user to build a profile")
    ap.add_argument("--out", default="feature_store/user_profiles.joblib",
                    help="Output joblib path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # load movie features
    fs = joblib.load(args.features)
    movies = fs["movies"]
    X_tfidf = fs["X_tfidf"]

    # load interactions
    inter = load_interactions(args.interactions)

    profiles = build_profiles(
        inter,
        movies_df=movies,
        X_tfidf=X_tfidf,
        min_ratings=args.min_ratings,
    )

    payload = {
        "profiles": profiles,
        "params": {
            "min_ratings": args.min_ratings,
            "features_path": args.features,
            "interactions_path": args.interactions,
        },
        "feature_space": "tfidf",
    }

    joblib.dump(payload, args.out)
    print(f"✓ built {len(profiles)} user profiles → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())