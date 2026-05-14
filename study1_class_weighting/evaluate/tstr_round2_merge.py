import csv
import sys

eval_dir = sys.argv[1]
timestamp = sys.argv[2]

backup_path = f"{eval_dir}/tstr_per_category_backup_{timestamp}.csv"
current_path = f"{eval_dir}/tstr_per_category.csv"

backup = []
with open(backup_path, newline="") as f:
    backup = list(csv.DictReader(f))

new = []
with open(current_path, newline="") as f:
    new = list(csv.DictReader(f))

new_only = [r for r in new if r["run_id"] != "ORACLE_real"]
merged = backup + new_only

with open(current_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
    w.writeheader()
    w.writerows(merged)

print(f"Backup rows: {len(backup)}")
print(f"New rows (excl ORACLE): {len(new_only)}")
print(f"Merged total: {len(merged)}")

runs = set(r["run_id"] for r in merged)
print(f"Unique runs: {len(runs)}")
for r in sorted(runs):
    n = sum(1 for row in merged if row["run_id"] == r)
    print(f"  {r}: {n} categories")

# Also merge fairness CSV if backup exists
fairness_backup = f"{eval_dir}/tstr_fairness_backup_{timestamp}.csv"
fairness_current = f"{eval_dir}/tstr_fairness.csv"

try:
    fb = []
    with open(fairness_backup, newline="") as f:
        fb = list(csv.DictReader(f))
    fn = []
    with open(fairness_current, newline="") as f:
        fn = list(csv.DictReader(f))
    fn_only = [r for r in fn if r["run_id"] != "ORACLE_real"]
    fm = fb + fn_only
    with open(fairness_current, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fm[0].keys()))
        w.writeheader()
        w.writerows(fm)
    print(f"Fairness merged: {len(fb)} backup + {len(fn_only)} new = {len(fm)} total")
except FileNotFoundError:
    print("No fairness CSV to merge (file not found)")
