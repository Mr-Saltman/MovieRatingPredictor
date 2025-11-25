from __future__ import annotations
import argparse
import joblib


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain Ridge rating model coefficients.")
    ap.add_argument("--model", required=True, help="Path to ridge_rating_model.joblib")
    ap.add_argument("--features", required=True, help="Path to movie_features.joblib")
    args = ap.parse_args()

    model_payload = joblib.load(args.model)
    model = model_payload["model"]

    fs = joblib.load(args.features)
    tfidf = fs["tfidf"]
    num_names = fs["feature_names"]["numeric"]

    # TF-IDF genre names (sorted by index)
    vocab_items = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])
    feature_names = [f"genre:{tok}" for tok, _ in vocab_items]

    # Numeric names
    feature_names.extend(f"num:{n}" for n in num_names)

    # User feature
    feature_names.append("user_mean_rating")

    coefs = model.coef_
    coef_names = list(zip(coefs, feature_names))

    print("\n=== Top Positive Features (push rating higher) ===")
    top_pos = sorted([x for x in coef_names if x[0] > 0], reverse=True)[:15]
    for w, name in top_pos:
        print(f"{w:+.4f}   {name}")

    print("\n=== Top Negative Features (push rating lower) ===")
    top_neg = sorted([x for x in coef_names if x[0] < 0])[:15]
    for w, name in top_neg:
        print(f"{w:+.4f}   {name}")


    print("\nDone.")


if __name__ == "__main__":
    main()