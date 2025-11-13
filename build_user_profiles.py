#!/usr/bin/env python3
"""
Build user "taste profiles" from interactions + movie features.

- Loads movie feature store (TF-IDF + numeric metadata) built earlier
  -> feature_store/movie_features.joblib
- Loads interactions (Parquet or CSV): userId, movie_rowid, rating, timestamp
- Creates a per-user TF‑IDF profile as a **weighted average** of liked/rated movies
  with optional **time decay** and **per-user mean-centering** of ratings.
- Saves to feature_store/user_profiles.joblib

Usage examples:
  python build_user_profiles.py \
    --interactions dataset/interactions.parquet \
    --features feature_store/movie_features.joblib \
    --min-ratings 5 --half-life-days 365 --center-ratings

Notes:
- If your interactions file has tmdb_id but not movie_rowid, this script will map it
  using the movies table in the feature store.
- Profiles are L2-normalized; we also persist raw (unnormalized) centroids if needed.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_interactions(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError("--interactions must be .parquet or .csv")

    # Normalize expected columns
    # Accept either movie_rowid or tmdb_id; rating + timestamp required
    required_any = [("movie_rowid",), ("tmdb_id",)]
    if not any(all(c in df.columns for c in opt) for opt in required_any):
        raise ValueError("interactions must contain either 'movie_rowid' or 'tmdb_id'")
    for c in ["userId", "rating"]:
        if c not in df.columns:
            raise ValueError(f"interactions is missing required column '{c}'")

    # Coerce types
    df["userId"] = pd.to_numeric(df["userId"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype(float)
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if not pd.api.types.is_numeric_dtype(ts):
            ts = pd.to_datetime(ts, errors="coerce", utc=True)
            df["timestamp"] = (ts.view("int64") // 1_000_000_000)  # seconds
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    return df.dropna(subset=["userId", "rating"]).copy()


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec if n < eps else (vec / n)


def build_profiles(
    inter: pd.DataFrame,
    movies_df: pd.DataFrame,
    X_tfidf: csr_matrix,
    *,
    min_ratings: int = 5,
    center_ratings: bool = True,
    half_life_days: float | None = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return dicts: {userId: profile_norm}, {userId: profile_raw} in TF-IDF space."""
    # Map movie id present in interactions to matrix row index
    if "movie_rowid" in inter.columns:
        key = "movie_rowid"
        map_series = pd.Series(range(len(movies_df)), index=movies_df["movie_rowid"])  # rowid -> row idx
        inter = inter.merge(map_series.rename("row_idx"), left_on="movie_rowid", right_index=True, how="left")
    else:
        key = "tmdb_id"
        map_series = pd.Series(range(len(movies_df)), index=movies_df["tmdb_id"])  # tmdb_id -> row idx
        inter = inter.merge(map_series.rename("row_idx"), left_on="tmdb_id", right_index=True, how="left")

    inter = inter.dropna(subset=["row_idx"]).copy()
    inter["row_idx"] = inter["row_idx"].astype(int)

    # Optionally center ratings per user (remove user bias)
    if center_ratings:
        user_mean = inter.groupby("userId")["rating"].transform("mean")
        inter["adj_rating"] = inter["rating"] - user_mean
    else:
        inter["adj_rating"] = inter["rating"]

    # Optional time decay weights
    if half_life_days and "timestamp" in inter.columns and inter["timestamp"].notna().any():
        now = pd.Timestamp.now(tz="UTC").value // 1_000_000_000  # seconds
        age_days = (now - inter["timestamp"].astype("Int64").fillna(now)) / 86400.0
        lam = np.log(2.0) / float(half_life_days)
        decay = np.exp(-lam * age_days.astype(float))
    else:
        decay = 1.0

    # Final weights: positive-only to avoid cancelling too much; add small floor
    w = np.clip(inter["adj_rating"].astype(float).values, a_min=0.0, a_max=None)
    w = w * (decay if np.isscalar(decay) else decay.values)
    w = np.where(w <= 0, 1e-6, w)

    profiles_norm: Dict[int, np.ndarray] = {}
    profiles_raw: Dict[int, np.ndarray] = {}

    for uid, grp in inter.groupby("userId"):
        if len(grp) < min_ratings:
            continue
        rows = grp["row_idx"].astype(int).tolist()
        weights = w[grp.index]
        # Weighted average in sparse space -> compute as sum(w_i * v_i) / sum(w_i)
        V = X_tfidf[rows]
        # Compute weighted centroid in sparse space (robust to return types)
        # Ensure weights is column-shaped for @; then coerce to ndarray
        weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        num = V.T @ weights
        num = np.asarray(num).ravel()  # dense 1D vector
        denom = weights.sum()
        if denom <= 0:
            continue
        centroid = num / denom
        profiles_raw[int(uid)] = centroid
        profiles_norm[int(uid)] = l2_normalize(centroid)

        return profiles_norm, profiles_raw

def main() -> int:
    ap = argparse.ArgumentParser(description="Build user taste profiles from interactions + movie features")
    ap.add_argument("--interactions", required=True, help="Path to interactions .parquet or .csv")
    ap.add_argument("--features", default="feature_store/movie_features.joblib", help="Movie feature store path")
    ap.add_argument("--min-ratings", type=int, default=5, help="Minimum ratings per user to build a profile")
    ap.add_argument("--center-ratings", action="store_true", help="Mean-center ratings per user")
    ap.add_argument("--half-life-days", type=float, default=None, help="Optional time-decay half-life in days")
    ap.add_argument("--out", default="feature_store/user_profiles.joblib", help="Output joblib path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load movie feature store
    fs = joblib.load(args.features)
    movies = fs["movies"]
    X_tfidf = fs["X_tfidf"]

    # Load interactions
    inter = load_interactions(args.interactions)

    profiles_norm, profiles_raw = build_profiles(
        inter,
        movies_df=movies,
        X_tfidf=X_tfidf,
        min_ratings=args.min_ratings,
        center_ratings=args.center_ratings,
        half_life_days=args.half_life_days,
    )

    payload = {
        "profiles_norm": profiles_norm,
        "profiles_raw": profiles_raw,
        "params": {
            "min_ratings": args.min_ratings,
            "center_ratings": args.center_ratings,
            "half_life_days": args.half_life_days,
            "features_path": args.features,
            "interactions_path": args.interactions,
        },
        "feature_space": "tfidf",
    }
    joblib.dump(payload, args.out)
    print(f"✓ built {len(profiles_norm)} user profiles → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
