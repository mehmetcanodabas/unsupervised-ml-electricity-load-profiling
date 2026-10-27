
import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

SCALER_FILE = MODELS_DIR / "feature_scaler.joblib"
MODEL_FILE = MODELS_DIR / "kmeans_k3.joblib"


CLUSTER_LABELS = {
    0: "Weekend-Intensive / Evening-Oriented",
    1: "Balanced / Regular",
    2: "Winter-Dominant / High-Variability",
}


def predict_customers(input_file, output_file):

    # ---------------------------------------------------
    # Load saved preprocessing and K-Means model
    # ---------------------------------------------------

    preprocessing = joblib.load(SCALER_FILE)
    model_bundle = joblib.load(MODEL_FILE)

    scaler = preprocessing["scaler"]

    feature_names = preprocessing["feature_names"]
    log_features = preprocessing["log_features"]
    time_feature_count = preprocessing["time_feature_count"]

    model = model_bundle["model"]

    if feature_names != model_bundle["feature_names"]:
        raise ValueError(
            "Model and scaler feature names differ."
        )

    # ---------------------------------------------------
    # Load customer features
    # ---------------------------------------------------

    df = pd.read_csv(input_file)

    required_columns = ["LCLid"] + feature_names

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Input file contains no customers."
        )

    if df["LCLid"].isna().any():
        raise ValueError(
            "Missing customer IDs."
        )

    if df["LCLid"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs."
        )

    X = df[feature_names].copy()

    if X.isna().any().any():
        raise ValueError(
            "Missing feature values."
        )

    X = X.astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Non-finite feature values."
        )

    # ---------------------------------------------------
    # Apply log transformation
    # ---------------------------------------------------

    for feature in log_features:

        if (X[feature] < 0).any():
            raise ValueError(
                f"Negative values in {feature}"
            )

        X[feature] = np.log1p(
            X[feature]
        )

    # ---------------------------------------------------
    # Apply saved StandardScaler
    # ---------------------------------------------------

    # Important: transform(), NOT fit_transform().
    # No new training takes place here.

    X_scaled = scaler.transform(X)

    # ---------------------------------------------------
    # Apply time-group weighting
    # ---------------------------------------------------

    X_scaled[:, :time_feature_count] /= math.sqrt(
        time_feature_count
    )

    # ---------------------------------------------------
    # Predict customer clusters
    # ---------------------------------------------------

    labels = model.predict(X_scaled)

    distances = model.transform(X_scaled)

    assigned_distances = distances[
        np.arange(len(labels)),
        labels
    ]

    # ---------------------------------------------------
    # Build customer profiling output
    # ---------------------------------------------------

    results = df[required_columns].copy()

    results.insert(
        1,
        "cluster",
        labels
    )

    results.insert(
        2,
        "behavior_profile",
        [
            CLUSTER_LABELS[int(label)]
            for label in labels
        ]
    )

    results["cluster_distance"] = assigned_distances

    # ---------------------------------------------------
    # Save results
    # ---------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        output_file,
        index=False,
        float_format="%.6f"
    )

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------

    print("=" * 60)
    print("CUSTOMER PROFILING INFERENCE")
    print("=" * 60)

    print(f"Customers processed: {len(results):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Clusters: {model.n_clusters}")

    print("\nCLUSTER DISTRIBUTION")
    print("-" * 60)

    for cluster, label in CLUSTER_LABELS.items():

        count = int(
            np.sum(labels == cluster)
        )

        print(
            f"Cluster {cluster}: "
            f"{count} customers | {label}"
        )

    print("\nSAMPLE CUSTOMER PROFILES")
    print("-" * 60)

    sample_columns = [
        "LCLid",
        "cluster",
        "behavior_profile",
        "cluster_distance",
    ]

    print(
        results[sample_columns]
        .head(5)
        .to_string(index=False)
    )

    print(f"\nSaved predictions: {output_file}")

    print("\nInference completed.")


# -------------------------------------------------------
# Command-line interface
# -------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Assign customers to previously learned "
            "electricity consumption behavior clusters."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input CSV containing LCLid "
            "and nine behavior features."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "predicted_customers.csv"
        ),
        help="Output CSV path."
    )

    args = parser.parse_args()

    predict_customers(
        args.input,
        args.output
    )
