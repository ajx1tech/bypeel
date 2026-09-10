"""
Retrains the IsolationForest anomaly-detection model from a labeled or
unlabeled transaction CSV and saves it via joblib in the exact bundle shape
`app/ml_engine.py` expects: {"model", "scaler", "feature_names", "contamination",
"random_state"}.

Usage:
    python scripts/train_model.py sample_data/labeled_transactions.csv models/isolation_forest.pkl

You only need to run this if you want to retrain on your own historical
corpus — a working pre-trained model is already bundled at
models/isolation_forest.pkl.
"""
import sys
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.ingest import load_many          # noqa: E402
from app.geoip_lookup import enrich_dataframe  # noqa: E402
from app.features import build_features, FEATURE_NAMES  # noqa: E402


def main(csv_path, out_path, contamination=0.012, n_estimators=200, random_state=42):
    with open(csv_path, "rb") as f:
        raw = f.read()
    df = load_many([(os.path.basename(csv_path), raw)])
    df = enrich_dataframe(df)
    df = build_features(df)

    X = df[FEATURE_NAMES].astype(float).values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model = IsolationForest(contamination=contamination, n_estimators=n_estimators,
                              random_state=random_state, n_jobs=-1)
    model.fit(Xs)

    bundle = {"model": model, "scaler": scaler, "feature_names": FEATURE_NAMES,
              "contamination": contamination, "random_state": random_state}
    joblib.dump(bundle, out_path)
    print(f"Saved retrained model to {out_path} (trained on {len(df):,} rows, "
          f"{len(FEATURE_NAMES)} features).")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/train_model.py <transactions.csv> <output_model.pkl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
