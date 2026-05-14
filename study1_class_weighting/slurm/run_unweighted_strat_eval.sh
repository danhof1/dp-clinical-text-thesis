#!/bin/bash
# run_unweighted_strat_eval.sh
# Per-category MAUVE evaluation for unweighted-stratified generation runs
# Run AFTER all 3 generation jobs from submit_unweighted_stratified.sh complete
#
# Estimated compute: ~3 GPU-hours total (1 per epsilon)

set -euo pipefail

PROJ="/fs1/projects/unlearning_pretraining/Proj_code"
PYTHON="/fs1/home/h702839428/python_daniel/bin/python"
SCRIPT="$PROJ/Scripts_2.0/per_category_eval.py"
LOGDIR="$PROJ/logs"

# Verify all 3 generation runs completed
for tag in eps05_unweighted_stratified eps1_unweighted_stratified eps4_unweighted_stratified; do
    STATS="$PROJ/generated/mimic/$tag/generation_stats.json"
    if [ ! -f "$STATS" ]; then
        echo "ERROR: $STATS not found — generation for $tag may not be complete"
        exit 1
    fi
    N=$(python3 -c "import json; print(json.load(open('$STATS'))['total_generated'])")
    echo "[$tag] $N notes generated"
done

echo ""
echo "All generation runs complete. Starting per-category MAUVE evaluation..."
echo ""

cd "$PROJ"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup $PYTHON $SCRIPT \
    --runs eps05_unweighted_stratified,eps1_unweighted_stratified,eps4_unweighted_stratified \
    --skip_adherence \
    --min_mauve_n 30 \
    > "$LOGDIR/eval_unweighted_strat_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

PID=$!
echo "Eval launched PID=$PID"
echo "Log: $LOGDIR/eval_unweighted_strat_*.log"
echo ""
echo "Results will append to: $PROJ/eval/per_category_mauve.csv"
echo "Expected: 63 new rows (21 categories × 3 epsilon values)"
