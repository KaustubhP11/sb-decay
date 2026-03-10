#!/bin/bash
#SBATCH --job-name=smollm3-SFT
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err
#SBATCH --time=11:30:00

bash text/finetuning/scripts/sft_openthoughts.sh