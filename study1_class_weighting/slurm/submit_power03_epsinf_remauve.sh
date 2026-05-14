#!/bin/bash
#SBATCH --job-name=power03_inf_remauve
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/power03_inf_remauve_%j.log

set -e
cd /fs1/projects/unlearning_pretraining/Proj_code

EVAL=/fs1/projects/unlearning_pretraining/Proj_code/eval
PY=/fs1/home/h702839428/python_daniel/bin/python

# Backup the master per_category_mauve.csv before running
cp -v "$EVAL/per_category_mauve.csv" "$EVAL/per_category_mauve.csv.pre_power03inf_$(date +%Y%m%d_%H%M%S)"

# Re-run per-cat eval on epsinf_power03cap10_mimic with relaxed min_mauve_n
# (existing run has 19 cats; Symptoms/Signs n=117 and Unknown n=32 are mauve=NA in eval_results.json)
$PY Scripts_2.0/per_category_eval.py \
  --runs epsinf_power03cap10_mimic \
  --skip_adherence \
  --min_mauve_n 25

# The script OVERWRITES per_category_mauve.csv with only the 1 run we passed.
# Merge: <pre-backup minus epsinf_power03cap10_mimic rows> + <new>
$PY - <<'PYEOF'
import csv, os, glob
EVAL = "/fs1/projects/unlearning_pretraining/Proj_code/eval"
MAIN = f"{EVAL}/per_category_mauve.csv"

with open(MAIN) as f:
    new_rows = list(csv.DictReader(f))
print(f"New rows from re-eval: {len(new_rows)}")

# Find most recent pre-power03inf backup
backups = sorted(glob.glob(f"{EVAL}/per_category_mauve.csv.pre_power03inf_*"))
assert backups, "No backup found!"
backup = backups[-1]
print(f"Restoring base from: {backup}")

with open(backup) as f:
    base_rows = list(csv.DictReader(f))
print(f"Backup rows: {len(base_rows)}")

new_runs = {r['run_id'] for r in new_rows}
merged = [r for r in base_rows if r['run_id'] not in new_runs] + new_rows
print(f"Merged total: {len(merged)}")

with open(MAIN, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run_id","category","mauve_score","n_synthetic","n_real"])
    w.writeheader()
    w.writerows(merged)
print(f"Wrote merged CSV to {MAIN}")

# Sanity: how many cats does epsinf_power03cap10_mimic now have?
from collections import Counter
by_run = Counter(r['run_id'] for r in merged)
print(f"epsinf_power03cap10_mimic cats: {by_run['epsinf_power03cap10_mimic']}")
PYEOF

echo "DONE."
