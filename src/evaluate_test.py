
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_FILE = DATA_DIR / "test_features.csv"

SCALER_FILE = MODELS_DIR / "feature_scaler.joblib"
MODEL_FILE = MODELS_DIR / "kmeans_k3.joblib"

OUTPUT_FILE = DATA_DIR / "test_cluster_profiles.csv"


# -------------------------------------------------------
# Load saved preprocessing and model
# -------------------------------------------------------

preprocessing = joblib.load(SCALER_FILE)
model_bundle = joblib.load(MODEL_FILE)

scaler = preprocessing["scaler"]
feature_names = preprocessing["feature_names"]
log_features = preprocessing["log_features"]
time_feature_count = preprocessing["time_feature_count"]

model = model_bundle["model"]

if feature_names != model_bundle["feature_names"]:
    raise ValueError("Feature names do not match.")


# -------------------------------------------------------
# Load test features
# -------------------------------------------------------

df = pd.read_csv(TEST_FILE)

if df["LCLid"].duplicated().any():
    raise ValueError("Duplicate test customer IDs.")

X = df[feature_names].copy()

if X.isna().any().any():
    raise ValueError("Missing test feature values.")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite test feature values.")


# -------------------------------------------------------
# Apply saved preprocessing
# -------------------------------------------------------

for feature in log_features:
    if (X[feature] < 0).any():
        raise ValueError(f"Negative values in {feature}")

    X[feature] = np.log1p(X[feature])

# Important: transform(), not fit_transform()
X_scaled = scaler.transform(X)

X_scaled[:, :time_feature_count] /= math.sqrt(
    time_feature_count
)


# -------------------------------------------------------
# Predict clusters
# -------------------------------------------------------

labels = model.predict(X_scaled)

distances = model.transform(X_scaled)

assigned_distances = distances[
    np.arange(len(labels)),
    labels
]


# -------------------------------------------------------
# Evaluate test results
# -------------------------------------------------------

print("=" * 65)
print("HELD-OUT TEST EVALUATION")
print("=" * 65)

print(f"Test customers: {len(df):,}")
print(f"Features: {len(feature_names)}")
print(f"K: {model.n_clusters}")

print("\nTEST CLUSTER SIZES")

for cluster in range(model.n_clusters):

    count = int(np.sum(labels == cluster))

    print(
        f"Cluster {cluster}: "
        f"{count:,} customers "
        f"({100 * count / len(labels):.2f}%)"
    )

if len(set(labels)) > 1:
    test_silhouette = silhouette_score(
        X_scaled,
        labels
    )

    print(f"\nTest Silhouette: {test_silhouette:.4f}")
else:
    print("\nTest Silhouette: Not defined")

print("\nTEST CLUSTER DISTANCES")

for cluster in range(model.n_clusters):

    cluster_distances = assigned_distances[
        labels == cluster
    ]

    if len(cluster_distances) == 0:
        print(f"Cluster {cluster}: No customers")
        continue

    print(
        f"Cluster {cluster}: "
        f"median={np.median(cluster_distances):.4f}, "
        f"p90={np.percentile(cluster_distances, 90):.4f}"
    )


# -------------------------------------------------------
# Save customer profiles
# -------------------------------------------------------

results = df.copy()

results["cluster"] = labels
results["cluster_distance"] = assigned_distances

results.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.8f"
)

print(f"\nSaved test profiles: {OUTPUT_FILE}")
print("\nTest evaluation completed.")
