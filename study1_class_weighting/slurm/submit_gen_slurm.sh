#!/bin/bash
#SBATCH --job-name=gen_strat
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/gen_strat_%A_%a.log

PROJ="/fs1/projects/unlearning_pretraining/Proj_code"
PYTHON="/fs1/home/h702839428/python_daniel/bin/python"
SCRIPT="$PROJ/Scripts_2.0/generation_scripts/03.2_generate.py"
BASE_MODEL="/fs1/shared/model/llm/Llama-3.2-1B-Instruct"
DATA="$PROJ/data/train.jsonl"

TAGS=("eps05" "eps1" "eps4")
CHECKPOINTS=(
    "$PROJ/models/llama_dp_eps0.5_20260319_1349/final"
    "$PROJ/models/llama_dp_eps1.0_20260318_1831/final"
    "$PROJ/models/llama_dp_eps4.0_20260318_1813/final"
)
EPSILONS=("0.5" "1.0" "4.0")

TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
CKPT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}
EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}
OUTDIR="$PROJ/generated/mimic/${TAG}_unweighted_stratified"

echo "Job $SLURM_ARRAY_TASK_ID: tag=$TAG eps=$EPS"
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
