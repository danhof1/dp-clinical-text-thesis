#!/bin/bash
#SBATCH --job-name=strat_mauve
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/strat_mauve_%j.log

set -e
cd /fs1/projects/unlearning_pretraining/Proj_code

EVAL=/fs1/projects/unlearning_pretraining/Proj_code/eval
PY=/fs1/home/h702839428/python_daniel/bin/python

# Backup the master per_category_mauve.csv before running
cp -v "$EVAL/per_category_mauve.csv" "$EVAL/per_category_mauve.csv.pre_strat_$(date +%Y%m%d_%H%M%S)"

# Run per-cat eval on the three new stratified outputs (skip adherence; only need MAUVE here)
$PY Scripts_2.0/per_category_eval.py \
  --runs eps05_unweighted_stratified,eps1_unweighted_stratified,eps4_unweighted_stratified \
  --skip_adherence

# The script will have OVERWRITTEN per_category_mauve.csv with only the 3 new runs.
# Merge: rebuild as <backup minus those 3 ids if present> + <new>
$PY - <<'PYEOF'
import csv, os, glob
EVAL = "/fs1/projects/unlearning_pretraining/Proj_code/eval"
MAIN = f"{EVAL}/per_category_mauve.csv"

new_rows = []
with open(MAIN) as f:
    new_rows = list(csv.DictReader(f))
print(f"New stratified rows: {len(new_rows)}")

# Find most recent pre-strat backup
backups = sorted(glob.glob(f"{EVAL}/per_category_mauve.csv.pre_strat_*"))
assert backups, "No backup found!"
backup = backups[-1]
print(f"Restoring base from: {backup}")

with open(backup) as f:
    base_rows = list(csv.DictReader(f))
print(f"Backup rows: {len(base_rows)}")

new_ids = {(r['run_id'], r['category']) for r in new_rows}
merged = [r for r in base_rows if (r['run_id'], r['category']) not in new_ids] + new_rows
print(f"Merged total: {len(merged)}")

with open(MAIN, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run_id","category","mauve_score","n_synthetic","n_real"])
    w.writeheader()
    w.writerows(merged)
print(f"Wrote merged CSV to {MAIN}")

# Summary: cats per stratified run
from collections import Counter
by_run = Counter(r['run_id'] for r in merged if 'stratified' in r['run_id'])
print(f"Stratified runs in final CSV: {dict(by_run)}")
PYEOF

echo "DONE."
