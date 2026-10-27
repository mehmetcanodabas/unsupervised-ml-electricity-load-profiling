
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "development_features_scaled.csv"
)

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

RANDOM_STATE = 42

df = pd.read_csv(INPUT_FILE)

if df["LCLid"].duplicated().any():
    raise ValueError("Duplicate customer IDs.")

X = df[FEATURE_NAMES].to_numpy(dtype=float)

if not np.isfinite(X).all():
    raise ValueError("Invalid feature values.")

print("=" * 85)
print("K-MEANS CLUSTER COUNT EVALUATION")
print("=" * 85)

print(f"Customers: {len(df):,}")
print(f"Features: {X.shape[1]}")
print(f"Random state: {RANDOM_STATE}")

print("\n" + "-" * 85)
print(
    f"{'K':<6}"
    f"{'Silhouette':>13}"
    f"{'Inertia':>15}"
    f"{'Smallest':>13}"
    f"{'Largest':>13}"
)
print("-" * 85)

results = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    labels = model.fit_predict(X)

    silhouette = silhouette_score(X, labels)

    sizes = np.bincount(labels, minlength=k)

    results.append({
        "k": k,
        "silhouette": silhouette,
        "inertia": model.inertia_,
        "smallest_cluster": int(sizes.min()),
        "largest_cluster": int(sizes.max()),
    })

    print(
        f"{k:<6}"
        f"{silhouette:>13.4f}"
        f"{model.inertia_:>15.2f}"
        f"{sizes.min():>13}"
        f"{sizes.max():>13}",
        flush=True
    )

print("\n" + "=" * 85)
print("BEST SILHOUETTE")
print("=" * 85)

best = max(
    results,
    key=lambda item: item["silhouette"]
)

print(f"K: {best['k']}")
print(f"Silhouette: {best['silhouette']:.4f}")
print(
    f"Cluster sizes: "
    f"{best['smallest_cluster']} to "
    f"{best['largest_cluster']}"
)

print("\nEvaluation completed.")
