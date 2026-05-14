#!/bin/bash
#SBATCH --job-name=diversity
#SBATCH --account=unlearning_pretraining
#SBATCH --partition=defq
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/fs1/projects/unlearning_pretraining/Proj_code/logs/diversity_analysis_%j.log

cd /fs1/projects/unlearning_pretraining/Proj_code
/fs1/home/h702839428/python_daniel/bin/python diversity_analysis.py
