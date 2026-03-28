#!/bin/bash
#SBATCH --job-name=smollm3-sft-openthoughts
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=normal
#SBATCH --time=11:30:00
#SBATCH --output=./logs/train-openthoughts-%j.out
#SBATCH --error=./logs/train-openthoughts-%j.err

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-/iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/configs/zero3.yaml}" \
TRAIN_CONFIG="${TRAIN_CONFIG:-/iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/configs/sft_openthoughts_answer_only.yaml}" \
srun --mpi=pmix --network=disable_rdzv_get --environment=ngc_25-11-nemo-alps3 --export=NONE \
  /iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/scripts/multi_node.sh
