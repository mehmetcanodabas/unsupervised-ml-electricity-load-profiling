
from pathlib import Path
import math

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "docs" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cluster_profiles.png"


# -------------------------------------------------------
# Cluster labels and colors
# -------------------------------------------------------

CLUSTERS = {
    0: {
        "name": "Weekend-Intensive",
        "color": "#E69F00",
    },
    1: {
        "name": "Balanced / Regular",
        "color": "#0072B2",
    },
    2: {
        "name": "Winter-Dominant",
        "color": "#009E73",
    },
}

TIME_FEATURES = [
    "share_00_04",
    "share_04_08",
    "share_08_12",
    "share_12_16",
    "share_16_20",
    "share_20_24",
]

TIME_LABELS = [
    "00-04",
    "04-08",
    "08-12",
    "12-16",
    "16-20",
    "20-24",
]


# -------------------------------------------------------
# Load data and trained artifacts
# -------------------------------------------------------

df = pd.read_csv(
    DATA_DIR / "development_features.csv"
)

preprocessing = joblib.load(
    MODELS_DIR / "feature_scaler.joblib"
)

model_bundle = joblib.load(
    MODELS_DIR / "kmeans_k3.joblib"
)

scaler = preprocessing["scaler"]
feature_names = list(preprocessing["feature_names"])
log_features = preprocessing["log_features"]
time_feature_count = preprocessing["time_feature_count"]

model = model_bundle["model"]

if feature_names != list(model_bundle["feature_names"]):
    raise ValueError("Model/preprocessing feature mismatch.")

# Remove the same excluded customers as during training.
excluded = preprocessing.get("excluded_customers", [])

df = df.loc[
    ~df["LCLid"].isin(excluded)
].copy()

if df["LCLid"].duplicated().any():
    raise ValueError("Duplicate customer IDs.")

if df[feature_names].isna().any().any():
    raise ValueError("Missing feature values.")


# -------------------------------------------------------
# Reproduce training preprocessing
# -------------------------------------------------------

X = df[feature_names].astype(float).copy()

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite feature values.")

for feature in log_features:
    if (X[feature] < 0).any():
        raise ValueError(f"Negative values: {feature}")

    X[feature] = np.log1p(X[feature])

X_scaled = scaler.transform(X)

X_scaled[:, :time_feature_count] /= math.sqrt(
    time_feature_count
)

df["cluster"] = model.predict(X_scaled)

if len(df) != 3208:
    raise ValueError(
        f"Expected 3208 customers, found {len(df)}."
    )


# -------------------------------------------------------
# Calculate cluster medians
# -------------------------------------------------------

cluster_medians = df.groupby("cluster")[
    TIME_FEATURES
    + [
        "weekend_weekday_ratio",
        "winter_summer_ratio",
    ]
].median()

cluster_counts = df["cluster"].value_counts()


# -------------------------------------------------------
# Chart design
# -------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6),
    gridspec_kw={"width_ratios": [1.35, 1]},
)

ax1, ax2 = axes

fig.suptitle(
    "Electricity Consumption Behavioral Segments",
    fontsize=19,
    fontweight="bold",
    y=0.98,
)

fig.text(
    0.5,
    0.91,
    "K-Means (K=3) | 3,208 Development Customers | Cluster Medians",
    ha="center",
    fontsize=11,
    color="#555555",
)


# -------------------------------------------------------
# Chart 1: Daily consumption profiles
# -------------------------------------------------------

x = np.arange(len(TIME_LABELS))

for cluster, info in CLUSTERS.items():

    values = (
        cluster_medians.loc[
            cluster,
            TIME_FEATURES
        ].to_numpy(dtype=float) * 100
    )

    count = int(cluster_counts.get(cluster, 0))

    ax1.plot(
        x,
        values,
        marker="o",
        linewidth=2.6,
        markersize=7,
        color=info["color"],
        label=f'{info["name"]} (n={count:,})',
    )

ax1.set_title(
    "A  |  Time-of-Day Consumption",
    loc="left",
    fontsize=13,
    pad=15,
)

ax1.set_xticks(x, TIME_LABELS)
ax1.set_xlabel("Time of Day (4-hour intervals)")
ax1.set_ylabel("Median Share of Annual Energy (%)")

ax1.grid(
    axis="y",
    alpha=0.2,
    linestyle="--",
)

ax1.legend(
    loc="upper left",
    fontsize=9,
    frameon=False,
)


# -------------------------------------------------------
# Chart 2: Behavioral ratios
# -------------------------------------------------------

categories = [
    "Weekend / Weekday",
    "Winter / Summer",
]

bar_x = np.arange(len(categories))
bar_width = 0.23

for i, (cluster, info) in enumerate(CLUSTERS.items()):

    values = [
        cluster_medians.loc[
            cluster,
            "weekend_weekday_ratio"
        ],
        cluster_medians.loc[
            cluster,
            "winter_summer_ratio"
        ],
    ]

    positions = bar_x + (i - 1) * bar_width

    bars = ax2.bar(
        positions,
        values,
        width=bar_width,
        color=info["color"],
        alpha=0.9,
        label=info["name"],
    )

    for bar, value in zip(bars, values):
        ax2.annotate(
            f"{value:.2f}",
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

ax2.set_title(
    "B  |  Behavioral Characteristics",
    loc="left",
    fontsize=13,
    pad=15,
)

ax2.set_xticks(bar_x, categories)
ax2.set_ylabel("Median Consumption Ratio")

ax2.set_ylim(
    0,
    max(
        cluster_medians[
            [
                "weekend_weekday_ratio",
                "winter_summer_ratio",
            ]
        ].to_numpy().max() * 1.2,
        1.5,
    ),
)

ax2.axhline(
    1,
    color="#777777",
    linewidth=1,
    linestyle="--",
    alpha=0.6,
)

ax2.grid(
    axis="y",
    alpha=0.2,
    linestyle="--",
)

ax2.set_axisbelow(True)


# -------------------------------------------------------
# Footnote and save
# -------------------------------------------------------

fig.text(
    0.5,
    0.045,
    "Profiles describe learned clusters; labels are analyst interpretations, not ground-truth classes.",
    ha="center",
    fontsize=9,
    color="#666666",
)

fig.subplots_adjust(
    left=0.07,
    right=0.97,
    top=0.82,
    bottom=0.20,
    wspace=0.26,
)

plt.savefig(
    OUTPUT_FILE,
    dpi=220,
    bbox_inches="tight",
    facecolor="white",
)

print("=" * 60)
print("CLUSTER VISUALIZATION")
print("=" * 60)

print(f"Customers: {len(df):,}")

for cluster, info in CLUSTERS.items():
    print(
        f'Cluster {cluster}: '
        f'{cluster_counts.get(cluster, 0):,} | '
        f'{info["name"]}'
    )

print(f"\nSaved: {OUTPUT_FILE}")
print("Visualization completed.")

plt.show()
