from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REFERENCE_FILE = DATA_DIR / "test_cluster_profiles.csv"
PREDICTION_FILE = DATA_DIR / "predicted_customers.csv"

EXPECTED_CUSTOMERS = 357
DISTANCE_TOLERANCE = 1e-5


# -------------------------------------------------------
# Load results
# -------------------------------------------------------

reference = pd.read_csv(REFERENCE_FILE)
predictions = pd.read_csv(PREDICTION_FILE)

required_columns = [
    "LCLid",
    "cluster",
    "cluster_distance",
]

for name, df in [
    ("Reference", reference),
    ("Predictions", predictions),
]:
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: Missing columns: {missing}"
        )

    if df["LCLid"].isna().any():
        raise ValueError(
            f"{name}: Missing customer IDs."
        )

    if df["LCLid"].duplicated().any():
        raise ValueError(
            f"{name}: Duplicate customer IDs."
        )


# -------------------------------------------------------
# Match customers by ID
# -------------------------------------------------------

merged = reference[required_columns].merge(
    predictions[required_columns],
    on="LCLid",
    how="outer",
    suffixes=("_reference", "_prediction"),
    indicator=True,
    validate="one_to_one",
)

missing_customers = int(
    (merged["_merge"] != "both").sum()
)

matched = merged[
    merged["_merge"] == "both"
].copy()


# -------------------------------------------------------
# Compare cluster assignments
# -------------------------------------------------------

cluster_mismatches = int(
    (
        matched["cluster_reference"]
        != matched["cluster_prediction"]
    ).sum()
)


# -------------------------------------------------------
# Compare distances
# -------------------------------------------------------

reference_distances = matched[
    "cluster_distance_reference"
].to_numpy(dtype=float)

prediction_distances = matched[
    "cluster_distance_prediction"
].to_numpy(dtype=float)

if (
    not np.isfinite(reference_distances).all()
    or not np.isfinite(prediction_distances).all()
):
    raise ValueError("Non-finite cluster distances.")

distance_differences = np.abs(
    reference_distances - prediction_distances
)

distance_mismatches = int(
    np.sum(distance_differences > DISTANCE_TOLERANCE)
)

max_distance_difference = (
    float(np.max(distance_differences))
    if len(distance_differences) > 0
    else float("nan")
)


# -------------------------------------------------------
# Final validation summary
# -------------------------------------------------------

print("=" * 65)
print("FINAL MODEL INFERENCE VALIDATION")
print("=" * 65)

print(f"Expected customers: {EXPECTED_CUSTOMERS}")
print(f"Reference customers: {len(reference)}")
print(f"Predicted customers: {len(predictions)}")
print(f"Matched customers: {len(matched)}")
print(f"Missing/extra customers: {missing_customers}")

print(f"\nCluster mismatches: {cluster_mismatches}")
print(f"Distance mismatches: {distance_mismatches}")

print(
    f"Maximum distance difference: "
    f"{max_distance_difference:.10f}"
)


# -------------------------------------------------------
# Pass / Fail
# -------------------------------------------------------

passed = (
    len(reference) == EXPECTED_CUSTOMERS
    and len(predictions) == EXPECTED_CUSTOMERS
    and len(matched) == EXPECTED_CUSTOMERS
    and missing_customers == 0
    and cluster_mismatches == 0
    and distance_mismatches == 0
)

print("\n" + "=" * 65)

if passed:
    print("FINAL VALIDATION: PASSED")
    print("Saved model reproduces test predictions.")
else:
    print("FINAL VALIDATION: FAILED")
    print("Differences detected between prediction files.")

print("=" * 65)

if not passed:
    raise SystemExit(1)
