from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix, hstack


def load_model_and_features(model_path: Path, features_path: Path):
    """Load trained Ridge model and movie feature store."""
    model_payload = joblib.load(model_path)
    model = model_payload["model"]

    fs = joblib.load(features_path)
    movies = fs["movies"]          # DataFrame: movie_rowid, tmdb_id, title, ...
    X_tfidf = fs["X_tfidf"]        # sparse text features
    num = fs["num"]                # numeric features (numpy array)

    return model, movies, X_tfidf, num


def ask_user_ratings(movies: pd.DataFrame, num_to_rate: int, min_ratings: int = 5):
    """
    Prompt the user to rate movies until we have `num_to_rate` ratings
    (or the user quits / we run out of candidates).

    Returns:
      ratings_dict: {movie_rowid: rating_float}
    """
    print("\n=== Step 1: Please rate some movies ===")
    print("Enter a rating from 1.0 to 5.0.")
    print("Press ENTER to skip a movie, or type 'q' to finish early.\n")

    # We'll define the candidate pool elsewhere (popular movies),
    # here we assume `movies` is already that pool, shuffled.
    candidates = movies.reset_index(drop=True)

    ratings: dict[int, float] = {}
    i = 0

    while i < len(candidates) and len(ratings) < num_to_rate:
        row = candidates.iloc[i]
        i += 1

        title = row.get("title", "Unknown title")
        movie_rowid = int(row["movie_rowid"])

        prompt = (
            f"[{len(ratings)+1}/{num_to_rate}] {title}\n"
            "Your rating (1-5, ENTER=skip, q=quit): "
        )

        while True:
            val = input(prompt).strip().lower()

            if val == "q":
                print("Stopping rating early.\n")
                break

            if val == "":
                rating = None  # skip this movie
                break

            try:
                rating = float(val)
                if 1.0 <= rating <= 5.0:
                    break
                else:
                    print("Please enter a number between 1.0 and 5.0.")
            except ValueError:
                print("Invalid input. Please enter a number (e.g. 3.5), ENTER to skip, or 'q' to quit.")

        if val == "q":
            break

        if rating is not None:
            ratings[movie_rowid] = rating

    if len(ratings) < min_ratings:
        print(f"\nYou rated only {len(ratings)} movies.")
        print(f"We need at least {min_ratings} ratings to build a user profile.")
        return {}

    print(f"\nThanks! You rated {len(ratings)} movies.\n")
    return ratings


def build_feature_matrix_for_user(
    X_tfidf,
    num: np.ndarray,
    user_mean_rating: float,
):
    """
    Build the full feature matrix X for this user over ALL movies.

    We replicate the same feature layout used during training:
      X = [movie_tfidf | movie_numeric | user_mean_rating_column]
    """
    n_movies = X_tfidf.shape[0]

    # Text and numeric features are already in correct order
    X_text = X_tfidf                         # (n_movies, d_text)
    X_num = csr_matrix(num)                  # (n_movies, d_num)

    # User feature: same scalar for all movies
    user_col = np.full((n_movies, 1), user_mean_rating, dtype=np.float32)
    X_user = csr_matrix(user_col)            # (n_movies, 1)

    X = hstack([X_text, X_num, X_user]).tocsr()
    return X


def recommend_for_user(
    model,
    movies: pd.DataFrame,
    X_tfidf,
    num: np.ndarray,
    user_ratings: dict[int, float],
    num_recs: int = 10,
):
    """Predict ratings for all movies and return top recommendations."""
    # compute simple user bias feature: mean of given ratings
    user_mean_rating = float(np.mean(list(user_ratings.values())))
    print(f"Estimated user_mean_rating: {user_mean_rating:.3f}")

    # build feature matrix for this user
    X_user_all = build_feature_matrix_for_user(X_tfidf, num, user_mean_rating)

    # predict ratings for all movies
    print("Predicting ratings for all movies...")
    y_pred = model.predict(X_user_all)

    # Build DataFrame with predictions
    preds = movies[["movie_rowid", "title"]].copy()
    preds["pred_rating"] = y_pred

    # exclude movies the user already rated
    rated_ids = set(user_ratings.keys())
    mask_unseen = ~preds["movie_rowid"].isin(rated_ids)
    preds = preds[mask_unseen]

    # sort by predicted rating, highest first
    preds = preds.sort_values("pred_rating", ascending=False)

    # take top N
    top = preds.head(num_recs).reset_index(drop=True)
    return top


def select_candidate_movies(movies: pd.DataFrame, pool_size: int = 1000) -> pd.DataFrame:
    """
    Select a pool of 'popular' movies to ask the user about.

    Strategy:
      - sort by vote_count and popularity,
      - take the top `pool_size`,
      - shuffle them.
    """
    cols = movies.columns

    sort_cols = []
    if "vote_count" in cols:
        sort_cols.append("vote_count")
    if "popularity" in cols:
        sort_cols.append("popularity")

    if sort_cols:
        popular = movies.sort_values(sort_cols, ascending=False)
    else:
        # fallback: fully random order if we don't have popularity info
        popular = movies.sample(frac=1.0)

    pool_size = min(pool_size, len(popular))
    pool = popular.head(pool_size)

    # shuffle within the popular pool (no fixed random_state -> different each run)
    pool = pool.sample(frac=1.0).reset_index(drop=True)
    return pool


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ask a user to rate some movies, then recommend movies using the trained Ridge model."
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("models/ridge_rating_model.joblib"),
        help="Path to trained Ridge model (joblib).",
    )
    ap.add_argument(
        "--features",
        type=Path,
        default=Path("feature_store/movie_features.joblib"),
        help="Path to movie features (joblib).",
    )
    ap.add_argument(
        "--num-rate",
        type=int,
        default=20,
        help="How many movies to ask the user to rate.",
    )
    ap.add_argument(
        "--num-recs",
        type=int,
        default=10,
        help="How many movies to recommend.",
    )
    ap.add_argument(
        "--min-ratings",
        type=int,
        default=5,
        help="Minimum number of ratings required to make recommendations.",
    )
    args = ap.parse_args()

    # 1) load model + features
    print("Loading model and movie features...")
    model, movies, X_tfidf, num = load_model_and_features(args.model, args.features)
    print(f"Loaded {len(movies)} movies.\n")

    # 2) ask the user for some ratings
    # 2) choose a popular pool, then ask the user for ratings
    candidate_pool = select_candidate_movies(movies, pool_size=1000)
    user_ratings = ask_user_ratings(
        candidate_pool,
        num_to_rate=args.num_rate,
        min_ratings=args.min_ratings,
    )
    if not user_ratings:
        print("Not enough ratings to make recommendations. Exiting.")
        return 0

    # 3) compute recommendations
    top = recommend_for_user(
        model=model,
        movies=movies,
        X_tfidf=X_tfidf,
        num=num,
        user_ratings=user_ratings,
        num_recs=args.num_recs,
    )

    # 4) print recommendations
    print("\n=== Top Recommendations for You ===")
    for i, row in top.iterrows():
        title = row.get("title", "Unknown title")
        pred = row["pred_rating"]
        print(f"{i+1:2d}. {title}  (predicted rating: {pred:.2f})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())