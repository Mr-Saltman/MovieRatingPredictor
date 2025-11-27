from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix, hstack
import os
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_IMG = "https://image.tmdb.org/t/p/w342"

def fetch_tmdb_details(tmdb_id: int) -> dict | None:
    """Fetch movie details from TMDB API for UI / display purposes."""
    if not TMDB_API_KEY or pd.isna(tmdb_id):
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
    params = {"api_key": TMDB_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.RequestException as e:
        print(f"TMDB request failed for {tmdb_id}: {e}")
        return None

# ---------- loading ----------

def load_model_and_features(model_path: Path, features_path: Path):
    """Load trained Ridge model and movie feature store."""
    model_payload = joblib.load(model_path)
    model = model_payload["model"]

    fs = joblib.load(features_path)
    # movies has: movie_rowid, tmdb_id, title, overview, runtime, vote_avg, vote_count, popularity, release_date
    movies = fs["movies"]
    X_tfidf = fs["X_tfidf"]        # sparse text features
    num = fs["num"]                # numeric features (numpy array)

    return model, movies, X_tfidf, num


# ---------- profile + ratings persistence ----------

def load_user_ratings(path: Path) -> Dict[int, float]:
    """Load saved user ratings from a JSON file (if it exists)."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # JSON keys are strings; convert back to int
    return {int(k): float(v) for k, v in raw.items()}


def save_user_ratings(path: Path, ratings: Dict[int, float]) -> None:
    """Save user ratings to a JSON file."""
    data = {str(k): float(v) for k, v in ratings.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_profiles(profiles_dir: Path) -> List[Path]:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return sorted(profiles_dir.glob("*.json"))


def choose_profile(
    profiles_dir: Path,
    profile_name_arg: str | None,
) -> Tuple[str, Dict[int, float], Path]:
    """
    Decide which profile to use.

    If profile_name_arg is provided:
      - load that profile if it exists, otherwise start a new empty profile.

    Otherwise:
      - list existing profiles (if any),
      - ask user to pick an existing one or create a new one.
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Non-interactive path: profile name provided via CLI
    if profile_name_arg:
        name = profile_name_arg.strip()
        if not name:
            name = "default"
        path = profiles_dir / f"{name}.json"
        ratings = load_user_ratings(path)
        if ratings:
            print(f"Loaded profile '{name}' with {len(ratings)} existing ratings.\n")
        else:
            print(f"Starting new profile '{name}'.\n")
        return name, ratings, path

    # Interactive selection
    existing = list_profiles(profiles_dir)

    if not existing:
        # no profiles yet -> create a new one
        name = input("No existing profiles found. Enter a name for a new profile (blank = 'default'): ").strip()
        if not name:
            name = "default"
        path = profiles_dir / f"{name}.json"
        ratings = load_user_ratings(path)
        print(f"Starting new profile '{name}'.\n")
        return name, ratings, path

    # There are existing profiles -> show menu
    print("Existing profiles:")
    for idx, p in enumerate(existing, start=1):
        print(f"  {idx}. {p.stem}")
    print("  n. Create new profile\n")

    while True:
        choice = input("Select profile number, or type a new name: ").strip()
        if choice.lower() == "n":
            name = input("Enter name for new profile (blank = 'default'): ").strip()
            if not name:
                name = "default"
            path = profiles_dir / f"{name}.json"
            ratings = load_user_ratings(path)
            print(f"Starting new profile '{name}'.\n")
            return name, ratings, path

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(existing):
                path = existing[idx - 1]
                name = path.stem
                ratings = load_user_ratings(path)
                print(f"Using existing profile '{name}' with {len(ratings)} ratings.\n")
                return name, ratings, path
            else:
                print("Invalid number. Please try again.")

        # treat as direct name
        name = choice if choice else "default"
        path = profiles_dir / f"{name}.json"
        ratings = load_user_ratings(path)
        if ratings:
            print(f"Using existing profile '{name}' with {len(ratings)} ratings.\n")
        else:
            print(f"Starting new profile '{name}'.\n")
        return name, ratings, path


# ---------- user rating collection ----------

def print_movie_context(row: pd.Series) -> None:
    """Print title, overview and TMDB link (if available)."""
    title = row.get("title", "Unknown title")
    tmdb_id = row.get("tmdb_id")

    # Start with MovieLens title
    display_title = str(title)

    overview = ""
    year = None
    poster_url = None

    details = None
    if pd.notna(tmdb_id):
        details = fetch_tmdb_details(tmdb_id)
        if details is not None:
            if details.get("title"):
                display_title = details["title"]
            overview = details.get("overview") or ""
            rd = details.get("release_date") or ""
            if len(rd) >= 4:
                year = rd[:4]
            poster_path = details.get("poster_path")
            if poster_path:
                poster_url = f"{TMDB_BASE_IMG}{poster_path}"

    print("\nTitle:", display_title)
    if year:
        print("Year :", year)

    if overview:
        text = str(overview).strip()
        if len(text) > 500:
            text = text[:500] + "..."
        print("Overview:", text)

    if pd.notna(tmdb_id):
        try:
            tmdb_int = int(tmdb_id)
            print(f"TMDB page: https://www.themoviedb.org/movie/{tmdb_int}")
        except (ValueError, TypeError):
            pass

    if poster_url:
        print("Poster:", poster_url)

    print()


def ask_user_ratings(movies: pd.DataFrame, num_to_rate: int) -> Dict[int, float]:
    """
    Prompt the user to rate movies until we have seen up to `num_to_rate` movies
    (or the user quits / we run out of candidates).

    Returns:
      ratings_dict: {movie_rowid: rating_float}
    """
    print("\n=== Step 1: Please rate some movies ===")
    print("For each movie, respond with:")
    print("  - 1–5   : rating if you've seen it")
    print("  - i     : interested (but not seen)  → treated as ~4.0")
    print("  - n     : not interested             → treated as ~2.0")
    print("  - ENTER : skip")
    print("  - q     : quit rating\n")

    candidates = movies.reset_index(drop=True)

    ratings: Dict[int, float] = {}
    i = 0

    while i < len(candidates) and len(ratings) < num_to_rate:
        row = candidates.iloc[i]
        i += 1

        movie_rowid = int(row["movie_rowid"])

        # show context (title, overview, tmdb link)
        print_movie_context(row)

        prompt = (
            f"[{len(ratings)+1}/{num_to_rate}] "
            "Your response (1-5 / i / n / ENTER=skip / q=quit): "
        )

        while True:
            val = input(prompt).strip().lower()
            val = val.replace(",", ".")  # allow 4,5 as 4.5

            if val == "q":
                print("Stopping rating early.\n")
                break

            if val == "":
                rating = None  # skip this movie
                break

            if val == "i":
                rating = 4.0
                break

            if val == "n":
                rating = 2.0
                break

            try:
                rating = float(val)
                if 1.0 <= rating <= 5.0:
                    break
                else:
                    print("Please enter a number between 1.0 and 5.0, 'i', 'n', ENTER, or 'q'.")
            except ValueError:
                print("Invalid input. Please enter 1–5, 'i', 'n', ENTER, or 'q'.")

        if val == "q":
            break

        if rating is not None:
            ratings[movie_rowid] = rating

    print(f"\nThanks! You provided {len(ratings)} new ratings/preferences in this round.\n")
    return ratings


# ---------- user genre profile ----------

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def build_user_profile(
    movies: pd.DataFrame,
    X_tfidf,
    user_ratings: Dict[int, float],
) -> np.ndarray:
    """
    Build a user profile vector in TF-IDF space from this user's ratings.

    Uses *centered* ratings (rating - mean), keeping both likes (positive)
    and dislikes (negative) as signed weights, then L2-normalizes.
    """
    if not user_ratings:
        raise ValueError("No user ratings provided")

    movie_to_idx = pd.Series(movies.index.values, index=movies["movie_rowid"])
    rated_ids = list(user_ratings.keys())

    keep_ids = [mid for mid in rated_ids if mid in movie_to_idx.index]
    if not keep_ids:
        raise ValueError("None of the rated movies are present in the feature matrix")

    ratings_arr = np.array([user_ratings[mid] for mid in keep_ids], dtype=float)
    user_mean = ratings_arr.mean()

    # signed weights: positive for above-mean, negative for below-mean
    deltas = ratings_arr - user_mean
    if np.allclose(deltas, 0.0):
        # fallback: if all ratings identical, just use raw ratings as weights
        weights = ratings_arr.reshape(-1, 1)
    else:
        weights = deltas.reshape(-1, 1)

    rows = movie_to_idx.loc[keep_ids].to_numpy(dtype=int)
    V = X_tfidf[rows]                      # (n_rated, d)
    profile_vec = V.T @ weights            # (d, 1)
    profile = np.asarray(profile_vec).ravel()

    denom = np.sum(np.abs(weights))
    if denom <= 0:
        raise ValueError("Sum of absolute profile weights is non-positive.")

    profile = profile / denom
    profile = l2_normalize(profile)
    return profile


# ---------- feature matrix for this user ----------

def build_feature_matrix_for_user(
    X_tfidf,
    num: np.ndarray,
    user_mean_rating: float,
    user_profile: np.ndarray,
):
    """
    Build X for this user over ALL movies:

      X = [movie_tfidf | movie_numeric | user_mean_rating | user_movie_sim]
    """
    n_movies = X_tfidf.shape[0]

    # TF-IDF and numeric features
    X_text = X_tfidf                      # (n_movies, d_text)
    X_num = csr_matrix(num)               # (n_movies, d_num)

    # user_mean_rating: same scalar for all movies
    user_col = np.full((n_movies, 1), user_mean_rating, dtype=np.float32)
    X_user = csr_matrix(user_col)         # (n_movies, 1)

    # similarity between each movie and user profile
    sims = X_tfidf @ user_profile         # (n_movies,)
    sims = np.asarray(sims).ravel().astype(np.float32)
    sims *= 10.0  # scale same as in training
    X_sim = csr_matrix(sims.reshape(-1, 1))  # (n_movies, 1)

    X = hstack([X_text, X_num, X_user, X_sim]).tocsr()
    return X


# ---------- recommendation ----------

def recommend_for_user(
    model,
    movies: pd.DataFrame,
    X_tfidf,
    num: np.ndarray,
    user_ratings: Dict[int, float],
    num_recs: int = 10,
    candidate_top_n: int = 5000,
    min_year: int | None = 1990,  # tweak this if you want
):
    """Predict ratings for all movies and return top recommendations."""

    # user bias feature
    user_mean_rating = float(np.mean(list(user_ratings.values())))
    print(f"Estimated user_mean_rating: {user_mean_rating:.3f}")

    # user genre profile
    user_profile = build_user_profile(movies, X_tfidf, user_ratings)
    print("Built user genre profile.")

    # build feature matrix for this user
    X_user_all = build_feature_matrix_for_user(
        X_tfidf=X_tfidf,
        num=num,
        user_mean_rating=user_mean_rating,
        user_profile=user_profile,
    )

    print("Predicting ratings for all movies...")
    y_pred = model.predict(X_user_all)

    # --- robust column selection ---
    base_cols = ["movie_rowid", "title"]
    extra_cols = []
    if "overview" in movies.columns:
        extra_cols.append("overview")
    if "tmdb_id" in movies.columns:
        extra_cols.append("tmdb_id")
    if "release_date" in movies.columns:
        extra_cols.append("release_date")
    if "n_ratings" in movies.columns:
        extra_cols.append("n_ratings")
    if "vote_count" in movies.columns:
        extra_cols.append("vote_count")
    if "popularity" in movies.columns:
        extra_cols.append("popularity")

    preds = movies[base_cols + extra_cols].copy()

    if "overview" not in preds.columns:
        preds["overview"] = ""
    if "tmdb_id" not in preds.columns:
        preds["tmdb_id"] = np.nan

    preds["pred_rating"] = y_pred

    # Filter out already-rated movies
    rated_ids = set(user_ratings.keys())
    preds = preds[~preds["movie_rowid"].isin(rated_ids)]

    if "release_date" in preds.columns and min_year is not None:
        rd = pd.to_datetime(preds["release_date"], errors="coerce")
        preds["release_year"] = rd.dt.year.fillna(1900).astype(int)
        preds = preds[preds["release_year"] >= min_year]

    # Sort: primary = predicted rating, tie-breakers = popularity-ish
    sort_cols = ["pred_rating"]
    ascending = [False]

    for col in ["n_ratings", "vote_count", "popularity"]:
        if col in preds.columns:
            sort_cols.append(col)
            ascending.append(False)

    preds = preds.sort_values(sort_cols, ascending=ascending)

    # Limit global candidate pool then take final top
    if candidate_top_n is not None and candidate_top_n > 0:
        preds = preds.head(candidate_top_n)

    top = preds.head(num_recs).reset_index(drop=True)
    return top


def refine_with_recommendations(
    model,
    movies: pd.DataFrame,
    X_tfidf,
    num: np.ndarray,
    user_ratings: Dict[int, float],
    num_recs: int = 10,
) -> pd.DataFrame:
    """
    Show recommendations and let the user give quick feedback to refine the profile.

    Returns:
      A DataFrame of final top recommendations after refinement.
    """
    # 1) Initial recommendations based on current ratings
    initial_top = recommend_for_user(
        model=model,
        movies=movies,
        X_tfidf=X_tfidf,
        num=num,
        user_ratings=user_ratings,
        num_recs=num_recs,
    )

    print("\n=== Based on your ratings, you might like: ===")
    print("Give feedback to refine recommendations.")
    print("For each movie, respond with:")
    print("  - 1–5   : rating if you've seen it")
    print("  - i     : interested (but not seen)  → treated as ~4.0")
    print("  - n     : not interested             → treated as ~2.0")
    print("  - ENTER : skip\n")

    added_feedback = 0

    for _, row in initial_top.iterrows():
        movie_id = int(row["movie_rowid"])

        if movie_id in user_ratings:
            continue

        # Show movie context again
        print_movie_context(row)
        pred = row["pred_rating"]

        prompt = f"(Predicted rating: {pred:.2f}) -> [1-5 / i / n / ENTER]: "
        val = input(prompt).strip().lower().replace(",", ".")

        if val == "":
            continue
        if val == "i":
            user_ratings[movie_id] = 4.0
            added_feedback += 1
        elif val == "n":
            user_ratings[movie_id] = 2.0
            added_feedback += 1
        else:
            try:
                r = float(val)
                if 1.0 <= r <= 5.0:
                    user_ratings[movie_id] = r
                    added_feedback += 1
                else:
                    print("Ignored: rating must be between 1 and 5.")
            except ValueError:
                print("Ignored: unrecognized input.")

    if added_feedback == 0:
        print("\nNo additional feedback given. Keeping initial recommendations.\n")
        return initial_top

    print(f"\nThank you! You now have {len(user_ratings)} ratings/preferences stored.")
    print("Recomputing recommendations with refined profile...\n")

    # 2) Recompute recommendations with updated profile
    refined_top = recommend_for_user(
        model=model,
        movies=movies,
        X_tfidf=X_tfidf,
        num=num,
        user_ratings=user_ratings,
        num_recs=num_recs,
    )
    return refined_top


# ---------- candidate pool ----------

def select_candidate_movies(
    movies: pd.DataFrame,
    X_tfidf,
    per_genre: int = 5,
    top_genres: int = 8,
    extra_random: int = 20,
    min_year: int | None = None,
) -> pd.DataFrame:
    """
    Select a diverse pool of movies to ask the user about.
    """
    cols = movies.columns
    base = movies.copy()

    # Filter by year for the candidate pool as well
    if "release_date" in base.columns and min_year is not None:
        rd = pd.to_datetime(base["release_date"], errors="coerce")
        base["release_year"] = rd.dt.year.fillna(1900).astype(int)
        base = base[base["release_year"] >= min_year]

    sort_cols = []
    if "n_ratings" in cols:
        sort_cols.append("n_ratings")
    if "vote_count" in cols:
        sort_cols.append("vote_count")
    if "popularity" in cols:
        sort_cols.append("popularity")

    if sort_cols:
        base = base.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    else:
        base = base.sample(frac=1.0).reset_index(drop=True)

    # 1) Document frequency per genre column
    df = (X_tfidf > 0).sum(axis=0)
    df = np.asarray(df).ravel()

    top_cols = np.argsort(df)[::-1][:top_genres]

    selected_ids: set[int] = set()
    candidates: list[pd.Series] = []

    for col_idx in top_cols:
        col_vec = X_tfidf[:, col_idx]
        mask = col_vec.toarray().ravel() > 0
        genre_movies = base[mask]

        for _, row in genre_movies.head(per_genre).iterrows():
            mid = int(row["movie_rowid"])
            if mid in selected_ids:
                continue
            selected_ids.add(mid)
            candidates.append(row)

    target_total = per_genre * top_genres + extra_random
    if len(candidates) < target_total:
        for _, row in base.iterrows():
            mid = int(row["movie_rowid"])
            if mid in selected_ids:
                continue
            selected_ids.add(mid)
            candidates.append(row)
            if len(candidates) >= target_total:
                break

    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sample(frac=1.0).reset_index(drop=True)
    return cand_df


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Ask a user to rate some movies, then recommend movies using a "
            "Ridge model with user_mean_rating and user–movie genre similarity."
        )
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
        help="How many movies to ask the user to rate in this session.",
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
        default=10,
        help="Minimum total number of ratings required to make recommendations.",
    )
    ap.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path("user_profiles"),
        help="Directory where user profiles (ratings) are stored as JSON.",
    )
    ap.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile name to use (if omitted, you'll be prompted).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # 1) load model + features
    print("Loading model and movie features...")
    model, movies, X_tfidf, num = load_model_and_features(args.model, args.features)
    print(f"Loaded {len(movies)} movies.\n")

    # 2) choose / load profile
    profile_name, user_ratings, profile_path = choose_profile(
        profiles_dir=args.profiles_dir,
        profile_name_arg=args.profile,
    )

    print(f"Current profile '{profile_name}' has {len(user_ratings)} ratings/preferences.\n")

    # If the user already has enough ratings and num-rate is 0,
    # go straight to recommendations/refinement.
    if len(user_ratings) >= args.min_ratings and args.num_rate <= 0:
        print(
            f"Profile '{profile_name}' already has {len(user_ratings)} ratings "
            f"(>= min_ratings={args.min_ratings}). Skipping new rating phase."
        )

        top = refine_with_recommendations(
            model=model,
            movies=movies,
            X_tfidf=X_tfidf,
            num=num,
            user_ratings=user_ratings,
            num_recs=args.num_recs,
        )

        save_user_ratings(profile_path, user_ratings)
        print(f"\nSaved {len(user_ratings)} total ratings/preferences to profile '{profile_name}'.\n")

        print("\n=== Final Top Recommendations for You ===")
        for i, row in top.iterrows():
            title = row.get("title", "Unknown title")
            pred = row["pred_rating"]
            print(f"{i+1:2d}. {title}  (predicted rating: {pred:.2f})")
        print("\nDone.")
        return 0

    # 3) choose a diverse pool of popular movies, excluding already rated ones
    candidate_pool = select_candidate_movies(
        movies,
        X_tfidf,
        per_genre=5,
        top_genres=8,
        extra_random=20,
        min_year=1990,
    )

    if user_ratings:
        already = set(user_ratings.keys())
        candidate_pool = candidate_pool[~candidate_pool["movie_rowid"].isin(already)]

    # 4) ask for additional ratings in this session
    new_ratings = ask_user_ratings(
        candidate_pool,
        num_to_rate=args.num_rate,
    )

    # merge new ratings into existing ones
    user_ratings.update(new_ratings)
    total_ratings = len(user_ratings)
    print(f"Profile '{profile_name}' now has {total_ratings} total ratings/preferences.\n")

    if total_ratings < args.min_ratings:
        print(
            f"You currently have only {total_ratings} ratings/preferences.\n"
            f"We require at least {args.min_ratings} to make recommendations.\n"
            "Please run again later after adding more ratings."
        )
        # still save what we have so far
        save_user_ratings(profile_path, user_ratings)
        return 0

    # 5) interactive refinement + recommendations
    top = refine_with_recommendations(
        model=model,
        movies=movies,
        X_tfidf=X_tfidf,
        num=num,
        user_ratings=user_ratings,
        num_recs=args.num_recs,
    )

    # 6) save updated profile
    save_user_ratings(profile_path, user_ratings)
    print(f"\nSaved {len(user_ratings)} total ratings/preferences to profile '{profile_name}'.\n")

    # 7) print final recommendations
    print("\n=== Final Top Recommendations for You ===")
    for i, row in top.iterrows():
        title = row.get("title", "Unknown title")
        pred = row["pred_rating"]
        print(f"{i+1:2d}. {title}  (predicted rating: {pred:.2f})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())