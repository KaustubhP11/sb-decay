#!/bin/bash
#SBATCH --job-name=smollm3-SFT
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err
#SBATCH --time=1:30:00

bash text/finetuning/scripts/sft_openthoughts.sh