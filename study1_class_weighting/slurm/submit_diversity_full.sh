#!/bin/bash
#SBATCH --job-name=diversity_full
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/diversity_full_%j.log

cd /fs1/projects/unlearning_pretraining/Proj_code
/fs1/home/h702839428/python_daniel/bin/python /fs1/projects/unlearning_pretraining/Proj_code/diversity_analysis_full.py
