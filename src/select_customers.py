
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Small LCL Data"
OUTPUT_FILE = PROJECT_ROOT / "data" / "eligible_customers.csv"

START = datetime(2013, 1, 1)
END = datetime(2014, 1, 1)

EXPECTED_SLOTS = 365 * 48
MIN_COMPLETENESS = 0.999

csv_files = sorted(
    DATA_DIR.glob("LCL-June2015v2_*.csv"),
    key=lambda p: int(p.stem.split("_")[-1])
)

customers = defaultdict(
    lambda: {
        "files": set(),
        "first": None,
        "last": None,
        "valid_slots": set(),
    }
)

for csv_file in csv_files:
    print(f"Processing {csv_file.name}", flush=True)

    with open(csv_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["stdorToU"].strip() != "Std":
                continue

            customer = row["LCLid"].strip()
            timestamp = datetime.fromisoformat(row["DateTime"])

            info = customers[customer]
            info["files"].add(csv_file.name)

            if info["first"] is None or timestamp < info["first"]:
                info["first"] = timestamp

            if info["last"] is None or timestamp > info["last"]:
                info["last"] = timestamp

            if not (START <= timestamp < END):
                continue

            value = row["KWH/hh (per half hour) "].strip()

            try:
                consumption = float(value)
            except ValueError:
                continue

            if not (0 <= consumption < float("inf")):
                continue

            if (
                timestamp.minute not in (0, 30)
                or timestamp.second != 0
                or timestamp.microsecond != 0
            ):
                continue

            info["valid_slots"].add(timestamp)

eligible_customers = sorted(
    customer
    for customer, info in customers.items()
    if len(info["files"]) == 1
    and info["first"] <= START
    and info["last"] >= END
    and len(info["valid_slots"]) / EXPECTED_SLOTS
    >= MIN_COMPLETENESS
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["LCLid"])

    for customer in eligible_customers:
        writer.writerow([customer])

print("\n" + "=" * 50)
print("CUSTOMER SELECTION SUMMARY")
print("=" * 50)
print(f"Selected customers: {len(eligible_customers):,}")
print(f"Completeness threshold: {MIN_COMPLETENESS:.1%}")
print(f"Saved to: {OUTPUT_FILE}")
