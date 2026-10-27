
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "Small LCL Data"
DATA_DIR = PROJECT_ROOT / "data"

CUSTOMERS_FILE = DATA_DIR / "development_customers.csv"
OUTPUT_FILE = DATA_DIR / "development_features.csv"

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


# -------------------------------------------------------
# Load development customers
# -------------------------------------------------------

with open(
    CUSTOMERS_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    development_ids = {
        row["LCLid"].strip()
        for row in reader
    }

print(f"Development customers: {len(development_ids):,}")


# -------------------------------------------------------
# Feature extraction
# -------------------------------------------------------

def extract_features(customer, measurements):

    daily_totals = defaultdict(float)

    bin_energy = [0.0] * 6

    winter_daily = defaultdict(float)
    summer_daily = defaultdict(float)

    for timestamp, consumption in measurements.items():

        day = timestamp.date()

        daily_totals[day] += consumption

        bin_energy[timestamp.hour // 4] += consumption

        if timestamp.month in (12, 1, 2):
            winter_daily[day] += consumption

        elif timestamp.month in (6, 7, 8):
            summer_daily[day] += consumption

    total_energy = sum(bin_energy)

    # Check 1: Zero or missing consumption
    if total_energy <= 0 or not daily_totals:

        print(f"\nSKIPPED CUSTOMER: {customer}")
        print("Reason: Zero or empty consumption")
        print(f"Total energy: {total_energy:.6f}")
        print(f"Days represented: {len(daily_totals)}")
        print(f"Unique measurements: {len(measurements)}")

        return None

    shares = [
        energy / total_energy
        for energy in bin_energy
    ]

    weekday_values = [
        energy
        for day, energy in daily_totals.items()
        if day.weekday() < 5
    ]

    weekend_values = [
        energy
        for day, energy in daily_totals.items()
        if day.weekday() >= 5
    ]

    # Check 2: Missing day categories
    if (
        not weekday_values
        or not weekend_values
        or not winter_daily
        or not summer_daily
    ):

        print(f"\nSKIPPED CUSTOMER: {customer}")
        print("Reason: Missing day or seasonal category")
        print(f"Weekday days: {len(weekday_values)}")
        print(f"Weekend days: {len(weekend_values)}")
        print(f"Winter days: {len(winter_daily)}")
        print(f"Summer days: {len(summer_daily)}")

        return None

    weekday_mean = mean(weekday_values)
    weekend_mean = mean(weekend_values)

    winter_mean = mean(winter_daily.values())
    summer_mean = mean(summer_daily.values())

    daily_values = list(daily_totals.values())
    daily_mean = mean(daily_values)

    # Check 3: Invalid ratio denominators
    if min(
        weekday_mean,
        summer_mean,
        daily_mean
    ) <= 0:

        print(f"\nSKIPPED CUSTOMER: {customer}")
        print("Reason: Zero denominator in feature ratios")
        print(f"Weekday mean: {weekday_mean:.6f}")
        print(f"Weekend mean: {weekend_mean:.6f}")
        print(f"Winter mean: {winter_mean:.6f}")
        print(f"Summer mean: {summer_mean:.6f}")
        print(f"Daily mean: {daily_mean:.6f}")
        print(f"Total energy: {total_energy:.6f}")

        return None

    weekly_ratio = weekend_mean / weekday_mean

    seasonal_ratio = winter_mean / summer_mean

    variability_cv = (
        pstdev(daily_values) / daily_mean
    )

    features = shares + [
        weekly_ratio,
        seasonal_ratio,
        variability_cv,
    ]

    # Check 4: Non-finite features
    if not all(
        math.isfinite(value)
        for value in features
    ):

        print(f"\nSKIPPED CUSTOMER: {customer}")
        print("Reason: Non-finite feature detected")
        print(f"Features: {features}")

        return None

    return features


# -------------------------------------------------------
# Process raw CSV files
# -------------------------------------------------------

csv_files = sorted(
    RAW_DATA_DIR.glob("LCL-June2015v2_*.csv"),
    key=lambda p: int(p.stem.split("_")[-1])
)

processed_ids = set()
skipped_ids = set()

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as output:

    writer = csv.writer(output)

    writer.writerow(["LCLid"] + FEATURE_NAMES)

    for csv_file in csv_files:

        customer_measurements = defaultdict(dict)

        with open(
            csv_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                customer = row["LCLid"].strip()

                if customer not in development_ids:
                    continue

                if row["stdorToU"].strip() != "Std":
                    continue

                timestamp = datetime.fromisoformat(
                    row["DateTime"]
                )

                if timestamp.year != 2013:
                    continue

                if (
                    timestamp.minute not in (0, 30)
                    or timestamp.second != 0
                    or timestamp.microsecond != 0
                ):
                    continue

                try:
                    consumption = float(
                        row["KWH/hh (per half hour) "].strip()
                    )

                except ValueError:
                    continue

                if (
                    not math.isfinite(consumption)
                    or consumption < 0
                ):
                    continue

                # Duplicate timestamps have identical values.
                customer_measurements[customer][timestamp] = (
                    consumption
                )

        for customer, measurements in (
            customer_measurements.items()
        ):

            if customer in processed_ids or customer in skipped_ids:
                raise ValueError(
                    f"Customer found in multiple files: {customer}"
                )

            features = extract_features(
                customer,
                measurements
            )

            if features is None:

                skipped_ids.add(customer)
                continue

            writer.writerow(
                [customer] +
                [f"{value:.8f}" for value in features]
            )

            processed_ids.add(customer)

        print(
            f"Processed {csv_file.name} | "
            f"Extracted customers: {len(processed_ids):,}",
            flush=True
        )


# -------------------------------------------------------
# Final summary
# -------------------------------------------------------

missing_ids = (
    development_ids
    - processed_ids
    - skipped_ids
)

print("\n" + "=" * 55)
print("DEVELOPMENT FEATURE EXTRACTION SUMMARY")
print("=" * 55)

print(f"Expected customers: {len(development_ids):,}")
print(f"Extracted customers: {len(processed_ids):,}")
print(f"Skipped customers: {len(skipped_ids):,}")
print(f"Not found: {len(missing_ids):,}")
print(f"Feature count: {len(FEATURE_NAMES)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nSKIPPED CUSTOMER IDS")

for customer in sorted(skipped_ids):
    print(customer)

print("\nMISSING CUSTOMER IDS")

for customer in sorted(missing_ids):
    print(customer)
