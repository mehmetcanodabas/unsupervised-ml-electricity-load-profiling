
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

INPUT_FILE = DATA_DIR / "development_features_scaled.csv"
MODEL_FILE = MODELS_DIR / "kmeans_k3.joblib"
OUTPUT_FILE = DATA_DIR / "development_cluster_assignments.csv"

FEATURE_NAMES = [
    "share_00_04",
    "share_04_08",
    "share_08_12",
    "share_12_16",
    "share_16_20",
    "share_20_24",
    "weekend_weekday_ratio",
    "winter_summer_ratio",
    "daily_variability_cv",
]

K = 3
RANDOM_STATE = 42


# -------------------------------------------------------
# Load development features
# -------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

if df["LCLid"].duplicated().any():
    raise ValueError("Duplicate customer IDs.")

X = df[FEATURE_NAMES].to_numpy(dtype=float)

if not np.isfinite(X).all():
    raise ValueError("Non-finite features detected.")

print("=" * 60)
print("FINAL K-MEANS TRAINING")
print("=" * 60)

print(f"Customers: {len(df):,}")
print(f"Features: {len(FEATURE_NAMES)}")
print(f"K: {K}")


# -------------------------------------------------------
# Train model
# -------------------------------------------------------

model = KMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    n_init=10,
)

labels = model.fit_predict(X)

silhouette = silhouette_score(X, labels)


# -------------------------------------------------------
# Cluster distances
# -------------------------------------------------------

# Distance to each learned cluster center.
distances = model.transform(X)

# Distance to the assigned cluster center.
assigned_distances = distances[
    np.arange(len(X)),
    labels
]


# -------------------------------------------------------
# Save model
# -------------------------------------------------------

MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "k": K,
        "random_state": RANDOM_STATE,
    },
    MODEL_FILE
)


# -------------------------------------------------------
# Save assignments
# -------------------------------------------------------

results = pd.DataFrame({
    "LCLid": df["LCLid"],
    "cluster": labels,
    "cluster_distance": assigned_distances,
})

results.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.8f",
)


# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print(f"\nSilhouette: {silhouette:.4f}")

print("\nCLUSTER SIZES")

for cluster in range(K):
    count = int(np.sum(labels == cluster))

    print(
        f"Cluster {cluster}: "
        f"{count:,} customers "
        f"({100 * count / len(labels):.2f}%)"
    )

print("\nCLUSTER DISTANCE SUMMARY")

for cluster in range(K):
    cluster_distances = assigned_distances[
        labels == cluster
    ]

    print(
        f"Cluster {cluster}: "
        f"median={np.median(cluster_distances):.4f}, "
        f"p90={np.percentile(cluster_distances, 90):.4f}"
    )

print(f"\nSaved model: {MODEL_FILE}")
print(f"Saved assignments: {OUTPUT_FILE}")

print("\nTraining completed.")
