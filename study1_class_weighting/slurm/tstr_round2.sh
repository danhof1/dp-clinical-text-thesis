#!/bin/bash
#SBATCH --job-name=tstr_r2
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/tstr_round2_%j.log

set -euo pipefail

PROJ="/fs1/projects/unlearning_pretraining/Proj_code"
PYTHON="/fs1/home/h702839428/python_daniel/bin/python"
SCRIPT="$PROJ/Scripts_2.0/07b_tstr_per_category.py"
EVAL="$PROJ/eval"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Phase 1: Backup ==="
cp "$EVAL/tstr_per_category.csv" "$EVAL/tstr_per_category_backup_${TIMESTAMP}.csv"
cp "$EVAL/tstr_fairness.csv" "$EVAL/tstr_fairness_backup_${TIMESTAMP}.csv" 2>/dev/null || echo "No fairness CSV to backup"

ORIG_LINES=$(wc -l < "$EVAL/tstr_per_category_backup_${TIMESTAMP}.csv")
echo "Backup verified: $ORIG_LINES lines"

echo ""
echo "=== Phase 2: TSTR Evaluation ==="
RUNS="eps05_log10_mimic,eps1_log10_mimic,eps4_log10_mimic,epsinf_log10_mimic,eps05_eff0999_mimic,eps1_eff0999_mimic,eps4_eff0999_mimic,epsinf_eff0999_mimic"
echo "Processing runs: $RUNS"

$PYTHON "$SCRIPT" --runs "$RUNS" --epochs 3

echo ""
echo "=== Phase 3: Merge ==="

$PYTHON "$PROJ/tstr_round2_merge.py" "$EVAL" "$TIMESTAMP"

echo ""
echo "=== All phases complete ==="
