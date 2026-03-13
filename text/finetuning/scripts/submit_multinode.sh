#!/bin/bash
#SBATCH --job-name=smollm3-sft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err

set -x

srun  --environment=/iopsstor/scratch/cscs/kponkshe/Megatron-LM/ncg_new_v2.toml --export=ALL /iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/scripts/multi_node.sh
