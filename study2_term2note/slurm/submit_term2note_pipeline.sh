#!/bin/bash
# Term2Note DP-LoRA training pipeline
# Usage: ssh star "bash /fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/thesis_scripts/submit_term2note_pipeline.sh"

PROJ=/fs1/projects/unlearning_pretraining/Proj_code
SCRIPTS=$PROJ/Scripts_3.0/thesis_scripts
PYTHON=/fs1/home/h702839428/python_daniel/bin/python
DP_LORA="$PROJ/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/finetune/dp_lora.py"
LOGDIR=$PROJ/logs

# Step 1: Convert JSONL → HuggingFace DatasetDict (CPU, fast)
echo "=== Step 1: Converting Term2Note data ==="
$PYTHON $SCRIPTS/convert_term2note_to_hf.py \
    --input $PROJ/data/term2note/train_term2note.jsonl \
    --output $PROJ/data/term2note/splits_term2note
echo ""

# Step 2: Submit training jobs via SLURM
cat > /tmp/term2note_eps4.sbatch << 'SBEOF'
#!/bin/bash
#SBATCH --job-name=t2n_eps4
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/dp_lora_term2note_eps4_%j.log

PROJ=/fs1/projects/unlearning_pretraining/Proj_code
PYTHON=/fs1/home/h702839428/python_daniel/bin/python

CUDA_VISIBLE_DEVICES=0 $PYTHON "$PROJ/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/finetune/dp_lora.py" \
    --config $PROJ/Scripts_3.0/thesis_scripts/config_term2note_eps4.yaml
SBEOF

cat > /tmp/term2note_eps8.sbatch << 'SBEOF'
#!/bin/bash
#SBATCH --job-name=t2n_eps8
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/dp_lora_term2note_eps8_%j.log

PROJ=/fs1/projects/unlearning_pretraining/Proj_code
PYTHON=/fs1/home/h702839428/python_daniel/bin/python

CUDA_VISIBLE_DEVICES=0 $PYTHON "$PROJ/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/finetune/dp_lora.py" \
    --config $PROJ/Scripts_3.0/thesis_scripts/config_term2note_eps8.yaml
SBEOF

echo "=== Step 2: Submitting training jobs ==="
JOB4=$(sbatch /tmp/term2note_eps4.sbatch | awk '{print $4}')
echo "ε=4 job: $JOB4"

JOB8=$(sbatch /tmp/term2note_eps8.sbatch | awk '{print $4}')
echo "ε=8 job: $JOB8"

echo ""
echo "Monitor: tail -f $LOGDIR/dp_lora_term2note_eps*"
echo "Queue:   squeue -u h702839428"
