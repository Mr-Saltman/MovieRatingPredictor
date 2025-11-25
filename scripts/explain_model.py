"""
Explain Ridge regression model trained with train_model.py.

Shows:
- Coefficients for genre features
- Coefficients for numeric movie features
- Coefficient for user_mean_rating

Usage:
  python explain_model.py --model models/ridge_rating_model.joblib \
                          --features feature_store/movie_features.joblib
"""

from __future__ import annotations
import argparse
import joblib
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Explain Ridge rating model coefficients.")
    ap.add_argument("--model", required=True, help="Path to ridge_rating_model.joblib")
    ap.add_argument("--features", required=True, help="Path to movie_features.joblib")
    args = ap.parse_args()

    payload = joblib.load(args.model)
    model = payload["model"]

    fs = joblib.load(args.features)
    tfidf = fs["tfidf"]
    num_names = fs["feature_names"]["numeric"]

    # Build feature name list
    feature_names = []

    # TF-IDF genre names
    vocab_items = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])
    for tok, idx in vocab_items:
        feature_names.append(f"genre:{tok}")

    # Numeric names
    for n in num_names:
        feature_names.append(f"num:{n}")

    # User feature
    feature_names.append("user_mean_rating")

    coefs = model.coef_

    print("\n=== Top Positive Features (push rating higher) ===")
    top_pos = sorted(zip(coefs, feature_names), reverse=True)[:15]
    for w, name in top_pos:
        print(f"{w:+.4f}   {name}")

    print("\n=== Top Negative Features (push rating lower) ===")
    top_neg = sorted(zip(coefs, feature_names))[:15]
    for w, name in top_neg:
        print(f"{w:+.4f}   {name}")

    print("\nDone.")


if __name__ == "__main__":
    main()