
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

INPUT_FILE = DATA_DIR / "development_features.csv"
OUTPUT_FILE = DATA_DIR / "development_features_scaled.csv"
SCALER_FILE = MODELS_DIR / "feature_scaler.joblib"

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

LOG_FEATURES = [
    "winter_summer_ratio",
    "daily_variability_cv",
]

EXCLUDED_CUSTOMERS = {
    "MAC004735",
}


# -------------------------------------------------------
# Load features
# -------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Original customers: {len(df):,}")

df = df[
    ~df["LCLid"].isin(EXCLUDED_CUSTOMERS)
].copy()

print(f"Customers after exclusion: {len(df):,}")

if df["LCLid"].duplicated().any():
    raise ValueError("Duplicate customer IDs detected.")

if df[FEATURE_NAMES].isna().any().any():
    raise ValueError("Missing feature values detected.")

if not np.isfinite(df[FEATURE_NAMES].to_numpy()).all():
    raise ValueError("Non-finite feature values detected.")


# -------------------------------------------------------
# Log transformation
# -------------------------------------------------------

X = df[FEATURE_NAMES].copy()

for feature in LOG_FEATURES:
    if (X[feature] < 0).any():
        raise ValueError(f"Negative values in {feature}")

    X[feature] = np.log1p(X[feature])


# -------------------------------------------------------
# Standard scaling
# -------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------------------------------
# Balance time-of-day feature group
# -------------------------------------------------------

TIME_FEATURE_COUNT = 6

X_scaled[:, :TIME_FEATURE_COUNT] /= math.sqrt(
    TIME_FEATURE_COUNT
)


# -------------------------------------------------------
# Save transformed data
# -------------------------------------------------------

MODELS_DIR.mkdir(parents=True, exist_ok=True)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=FEATURE_NAMES,
    index=df.index
)

scaled_df.insert(
    0,
    "LCLid",
    df["LCLid"].values
)

scaled_df.to_csv(
    OUTPUT_FILE,
    index=False,
    float_format="%.8f"
)

joblib.dump(
    {
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "log_features": LOG_FEATURES,
        "time_feature_count": TIME_FEATURE_COUNT,
        "excluded_customers": sorted(EXCLUDED_CUSTOMERS),
    },
    SCALER_FILE
)


# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE PREPROCESSING SUMMARY")
print("=" * 60)

print(f"Customers: {len(scaled_df):,}")
print(f"Feature count: {len(FEATURE_NAMES)}")
print(f"Log features: {LOG_FEATURES}")
print(f"Time feature divisor: sqrt(6)")
print(f"Missing values: {scaled_df[FEATURE_NAMES].isna().sum().sum()}")
print(f"Saved features: {OUTPUT_FILE}")
print(f"Saved scaler: {SCALER_FILE}")

print("\nTRANSFORMED FEATURE STANDARD DEVIATIONS")

for name, std in zip(
    FEATURE_NAMES,
    X_scaled.std(axis=0)
):
    print(f"{name:<28} {std:.4f}")
