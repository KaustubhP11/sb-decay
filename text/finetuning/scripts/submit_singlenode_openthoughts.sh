#!/bin/bash
#SBATCH --job-name=smollm3-sft-openthoughts
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --time=11:30:00
#SBATCH --output=./logs/train-openthoughts-%j.out
#SBATCH --error=./logs/train-openthoughts-%j.err

set -euo pipefail
mkdir -p logs

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-text/finetuning/configs/zero3.yaml}" \
TRAIN_CONFIG="${TRAIN_CONFIG:-text/finetuning/configs/sft_openthoughts_answer_only.yaml}" \
bash text/finetuning/scripts/single_node.sh
