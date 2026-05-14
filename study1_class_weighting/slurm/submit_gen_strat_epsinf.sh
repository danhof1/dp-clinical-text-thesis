#!/bin/bash
#SBATCH --job-name=gen_strat_inf
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/gen_strat_inf_%j.log

PROJ="/fs1/projects/unlearning_pretraining/Proj_code"
PYTHON="/fs1/home/h702839428/python_daniel/bin/python"
SCRIPT="$PROJ/Scripts_2.0/generation_scripts/03.2_generate.py"
BASE_MODEL="/fs1/shared/model/llm/Llama-3.2-1B-Instruct"
DATA="$PROJ/data/train.jsonl"

CKPT="$PROJ/models/llama_dp_eps999.0_20260321_2101/final"
EPS="999.0"
OUTDIR="$PROJ/generated/mimic/epsinf_unweighted_stratified"

echo "Stratified gen for eps=inf unweighted"
echo "Checkpoint: $CKPT"
echo "Output: $OUTDIR"

$PYTHON $SCRIPT \
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
    --seed 42
