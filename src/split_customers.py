
import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "eligible_customers.csv"

RANDOM_SEED = 42
TEST_RATIO = 0.10

# Read eligible customer IDs
with open(INPUT_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)
    customer_ids = sorted(row["LCLid"] for row in reader)

# Reproducible customer-level split
rng = random.Random(RANDOM_SEED)
rng.shuffle(customer_ids)

test_size = round(len(customer_ids) * TEST_RATIO)

test_customers = sorted(customer_ids[:test_size])
development_customers = sorted(customer_ids[test_size:])

def save_customers(filename, customers):
    with open(DATA_DIR / filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["LCLid"])

        for customer in customers:
            writer.writerow([customer])

save_customers("development_customers.csv", development_customers)
save_customers("test_customers.csv", test_customers)

print("=" * 45)
print("CUSTOMER TRAIN/TEST SPLIT")
print("=" * 45)

print(f"Total customers: {len(customer_ids):,}")
print(f"Development customers: {len(development_customers):,}")
print(f"Test customers: {len(test_customers):,}")
print(f"Random seed: {RANDOM_SEED}")

print(
    "Customer overlap: "
    f"{len(set(development_customers) & set(test_customers))}"
)
