#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------- data loaders ----------

def load_interactions(path: Path) -> pd.DataFrame:
    """Load interactions from CSV or Parquet and normalize dtypes."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("interactions must be .csv or .parquet")

    required = {"userId", "movie_rowid", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {missing}")

    df["userId"] = df["userId"].astype(int)
    df["movie_rowid"] = df["movie_rowid"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df


def load_movie_features(path: Path) -> dict:
    """Load movie_features.joblib (movies, X_tfidf, num, ...)."""
    return joblib.load(path)


# ---------- feature builder ----------

def build_design_matrix(
    interactions: pd.DataFrame,
    movies: pd.DataFrame,
    X_tfidf,
    num: np.ndarray,
):
    """
    Build the feature matrix X and target vector y.

    X = [movie_tfidf | movie_numeric | user_mean_rating]
    y = rating
    """
    # Map movie_rowid -> row index in X_tfidf / num
    movie_to_idx = pd.Series(
        movies.index.values,
        index=movies["movie_rowid"],
    )

    interactions = interactions.merge(
        movie_to_idx.rename("row_idx"),
        left_on="movie_rowid",
        right_index=True,
        how="inner",
    )
    interactions["row_idx"] = interactions["row_idx"].astype(int)

    # movie features
    rows = interactions["row_idx"].to_numpy(dtype=int)
    X_text = X_tfidf[rows]           # sparse
    X_num_dense = num[rows]          # (n_samples, d_num)
    X_num = csr_matrix(X_num_dense)  # as sparse

    # user feature: mean rating per user (user bias)
    interactions["user_mean_rating"] = (
        interactions.groupby("userId")["rating"].transform("mean")
    )
    user_mean_col = interactions["user_mean_rating"].to_numpy(dtype=float).reshape(-1, 1)
    X_user = csr_matrix(user_mean_col)  # (n_samples, 1)

    # Stack everything horizontally: [tfidf | numeric | user_mean]
    X = hstack([X_text, X_num, X_user]).tocsr()
    y = interactions["rating"].to_numpy(dtype=float)

    return X, y


# ---------- training ----------

def train_model(
    interactions_path: Path,
    features_path: Path,
    model_out: Path,
    test_size: float = 0.2,
    alpha: float = 1.0,
    max_samples: int | None = 200_000,
) -> None:
    print(f"Loading interactions from {interactions_path} ...")
    interactions = load_interactions(interactions_path)
    print(f"Total interactions: {len(interactions):,}")

    # Optional subsampling
    if max_samples is not None and len(interactions) > max_samples:
        print(f"Subsampling to {max_samples:,} interactions (from {len(interactions):,}) ...")
        interactions = interactions.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"Using {len(interactions):,} interactions.")

    print(f"Loading movie features from {features_path} ...")
    fs = load_movie_features(features_path)
    movies = fs["movies"]
    X_tfidf = fs["X_tfidf"]
    num = fs["num"]

    print("Building design matrix...")
    X, y = build_design_matrix(interactions, movies, X_tfidf, num)
    print(f"Feature matrix shape: {X.shape}, target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
    )

    print(f"Training samples: {X_train.shape[0]:,}, test samples: {X_test.shape[0]:,}")

    print(f"Training Ridge regression (alpha={alpha}) ...")
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print("\nEvaluation on held-out test set:")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "alpha": alpha,
        "features_path": str(features_path),
        "interactions_path": str(interactions_path),
        "n_features": X.shape[1],
        "max_samples": max_samples,
        "description": "Ridge regression: [movie_tfidf | movie_numeric | user_mean_rating] -> rating",
    }
    joblib.dump(payload, model_out)
    print(f"\n✓ Saved trained model to {model_out}")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a Ridge regression rating predictor.")
    ap.add_argument("--interactions", type=Path, required=True,
                    help="Path to interactions .csv or .parquet (with userId, movie_rowid, rating)")
    ap.add_argument("--features", type=Path,
                    default=Path("feature_store/movie_features.joblib"),
                    help="Path to movie_features.joblib")
    ap.add_argument("--model-out", type=Path,
                    default=Path("models/ridge_rating_model.joblib"),
                    help="Path to save trained model (joblib file)")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="Fraction of data to use as test set")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Ridge regularization strength")
    ap.add_argument("--max-samples", type=int, default=200_000,
                    help="Maximum number of interactions to use for training "
                         "(set to -1 to use all)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    max_samples = None if args.max_samples == -1 else args.max_samples

    train_model(
        interactions_path=args.interactions,
        features_path=args.features,
        model_out=args.model_out,
        test_size=args.test_size,
        alpha=args.alpha,
        max_samples=max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())