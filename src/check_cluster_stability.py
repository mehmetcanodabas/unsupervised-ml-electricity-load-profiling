
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT / "data" / "development_features_scaled.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" / "kmeans_k3.joblib"
)

RANDOM_STATE = 42
N_RUNS = 10
SAMPLE_FRACTION = 0.80


# -------------------------------------------------------
# Load data and saved model
# -------------------------------------------------------

df = pd.read_csv(DATA_FILE)

bundle = joblib.load(MODEL_FILE)

reference_model = bundle["model"]
feature_names = bundle["feature_names"]

X = df[feature_names].to_numpy(dtype=float)

if not np.isfinite(X).all():
    raise ValueError("Non-finite feature values detected.")

reference_labels = reference_model.predict(X)

rng = np.random.default_rng(RANDOM_STATE)

scores = []

print("=" * 65)
print("K-MEANS STABILITY CHECK")
print("=" * 65)

print(f"Customers: {len(X):,}")
print(f"K: {reference_model.n_clusters}")
print(f"Runs: {N_RUNS}")
print(f"Sample fraction: {SAMPLE_FRACTION:.0%}")

print("\nRUN RESULTS")
print("-" * 45)

for run in range(N_RUNS):

    sample_indices = rng.choice(
        len(X),
        size=int(len(X) * SAMPLE_FRACTION),
        replace=False,
    )

    sample_X = X[sample_indices]

    sample_model = KMeans(
        n_clusters=reference_model.n_clusters,
        random_state=RANDOM_STATE + run,
        n_init=10,
    )

    sample_model.fit(sample_X)

    # Compare on the sampled customers only.
    sample_labels = sample_model.predict(sample_X)
    reference_sample_labels = reference_labels[sample_indices]

    ari = adjusted_rand_score(
        reference_sample_labels,
        sample_labels,
    )

    scores.append(ari)

    print(
        f"Run {run + 1:>2}: "
        f"ARI = {ari:.4f}",
        flush=True
    )

print("\nSTABILITY SUMMARY")
print("-" * 45)

print(f"Mean ARI: {np.mean(scores):.4f}")
print(f"Min ARI: {np.min(scores):.4f}")
print(f"Max ARI: {np.max(scores):.4f}")

print("\nStability check completed.")
