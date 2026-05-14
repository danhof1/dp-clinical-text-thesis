#!/bin/bash
# Submit Term2Note section-wise generation jobs
# Run after training jobs 21903 (eps4) and 21904 (eps8) complete

PROJ=/fs1/projects/unlearning_pretraining/Proj_code
SCRIPTS=$PROJ/Scripts_3.0/thesis_scripts
PYTHON=/fs1/home/h702839428/python_daniel/bin/python
LOGDIR=$PROJ/logs

BASE_MODEL=/fs1/shared/model/llm/Llama-3.2-1B-Instruct
TERM2NOTE_DATA=$PROJ/data/term2note/train_term2note.jsonl

# ε=4 generation
cat > /tmp/gen_term2note_eps4.sbatch << 'SBEOF'
#!/bin/bash
#SBATCH --job-name=gen_t2n_e4
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/gen_term2note_eps4_%j.log

export HF_DATASETS_CACHE=/fs1/projects/unlearning_pretraining/Proj_code/.cache/hf_datasets
export HF_HOME=/fs1/projects/unlearning_pretraining/Proj_code/.cache/hf_home

PROJ=/fs1/projects/unlearning_pretraining/Proj_code
PYTHON=/fs1/home/h702839428/python_daniel/bin/python

CUDA_VISIBLE_DEVICES=0 $PYTHON $PROJ/Scripts_3.0/thesis_scripts/generate_term2note.py \
    --checkpoint $PROJ/Scripts_3.0/thesis_scripts/outputs/dp_lora_term2note_eps4/final \
    --base_model /fs1/shared/model/llm/Llama-3.2-1B-Instruct \
    --term2note_data $PROJ/data/term2note/train_term2note.jsonl \
    --output $PROJ/Scripts_3.0/thesis_scripts/outputs/generated/term2note_eps4 \
    --n_generate 2000 \
    --k 4
SBEOF

# ε=8 generation
cat > /tmp/gen_term2note_eps8.sbatch << 'SBEOF'
#!/bin/bash
#SBATCH --job-name=gen_t2n_e8
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/gen_term2note_eps8_%j.log

export HF_DATASETS_CACHE=/fs1/projects/unlearning_pretraining/Proj_code/.cache/hf_datasets
export HF_HOME=/fs1/projects/unlearning_pretraining/Proj_code/.cache/hf_home

PROJ=/fs1/projects/unlearning_pretraining/Proj_code
PYTHON=/fs1/home/h702839428/python_daniel/bin/python

CUDA_VISIBLE_DEVICES=0 $PYTHON $PROJ/Scripts_3.0/thesis_scripts/generate_term2note.py \
    --checkpoint $PROJ/Scripts_3.0/thesis_scripts/outputs/dp_lora_term2note_eps8/final \
    --base_model /fs1/shared/model/llm/Llama-3.2-1B-Instruct \
    --term2note_data $PROJ/data/term2note/train_term2note.jsonl \
    --output $PROJ/Scripts_3.0/thesis_scripts/outputs/generated/term2note_eps8 \
    --n_generate 2000 \
    --k 4
SBEOF

# Submit with dependency on training jobs
echo "Submitting generation jobs with dependency on training..."
JOB_GEN4=$(sbatch --dependency=afterok:21903 /tmp/gen_term2note_eps4.sbatch | awk '{print $4}')
echo "ε=4 generation job: $JOB_GEN4 (depends on training 21903)"

JOB_GEN8=$(sbatch --dependency=afterok:21904 /tmp/gen_term2note_eps8.sbatch | awk '{print $4}')
echo "ε=8 generation job: $JOB_GEN8 (depends on training 21904)"

echo ""
echo "Queue: squeue -u h702839428"
echo "Logs:  tail -f $LOGDIR/gen_term2note_eps*.log"
