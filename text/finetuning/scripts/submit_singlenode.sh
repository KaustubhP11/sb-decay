#!/bin/bash
#SBATCH --job-name=smollm3-sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --time=11:30:00
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err

set -euo pipefail
mkdir -p logs

bash text/finetuning/scripts/single_node.sh