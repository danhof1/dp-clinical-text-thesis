#!/bin/bash
# submit_unweighted_stratified.sh
# Run stratified generation (n_per_category=500) from existing unweighted checkpoints
# Purpose: 2x2 factorial control isolating weighting from stratified sampling
#
# PPL reference: Llama-3.2-1B-Instruct (matches sqrt-stratified runs)
# Note: existing unweighted-proportional runs used Llama-3.1-8B-Instruct as ppl_ref
#
# Estimated compute: ~2 GPU-hrs per epsilon, ~6 GPU-hrs total

set -euo pipefail

PROJ="/fs1/projects/unlearning_pretraining/Proj_code"
PYTHON="/fs1/home/h702839428/python_daniel/bin/python"
SCRIPT="$PROJ/Scripts_2.0/generation_scripts/03.2_generate.py"
BASE_MODEL="/fs1/shared/model/llm/Llama-3.2-1B-Instruct"
DATA="$PROJ/data/train.jsonl"
LOGDIR="$PROJ/logs"
GENDIR="$PROJ/generated/mimic"

declare -A CHECKPOINTS=(
    ["eps05"]="$PROJ/models/llama_dp_eps0.5_20260319_1349/final"
    ["eps1"]="$PROJ/models/llama_dp_eps1.0_20260318_1831/final"
    ["eps4"]="$PROJ/models/llama_dp_eps4.0_20260318_1813/final"
)

declare -A EPSILONS=(
    ["eps05"]="0.5"
    ["eps1"]="1.0"
    ["eps4"]="4.0"
)

for tag in eps05 eps1 eps4; do
    OUTDIR="$GENDIR/${tag}_unweighted_stratified"
    CKPT="${CHECKPOINTS[$tag]}"
    EPS="${EPSILONS[$tag]}"

    if [ -d "$OUTDIR" ] && [ -f "$OUTDIR/synthetic_bhc.jsonl" ]; then
        echo "[$tag] Output dir exists — script will auto-resume: $OUTDIR"
    else
        echo "[$tag] Starting fresh generation → $OUTDIR"
    fi

    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup $PYTHON $SCRIPT \
        --checkpoint "$CKPT" \
        --base_model "$BASE_MODEL" \
        --data "$DATA" \
        --output "$OUTDIR" \
        --n_per_category 500 \
        --k 4 \
        --max_new_tokens 1024 \
        --temperature 0.1 \
        --top_p 1.0 \
        --repetition_penalty 1.2 \
        --epsilon "$EPS" \
        --seed 42 \
        > "$LOGDIR/gen_unweighted_strat_${tag}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    PID=$!
    echo "[$tag] Launched PID=$PID (eps=$EPS, checkpoint=$CKPT)"
    echo "[$tag] Log: $LOGDIR/gen_unweighted_strat_${tag}_*.log"
    echo ""
done

echo "All 3 generation jobs launched. Monitor with: tail -f $LOGDIR/gen_unweighted_strat_*.log"
echo ""
echo "After all complete, run eval:"
echo "  bash $PROJ/Scripts_2.0/run_unweighted_strat_eval.sh"
