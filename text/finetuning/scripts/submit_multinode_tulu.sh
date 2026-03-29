#!/bin/bash
#SBATCH --job-name=smollm3-sft-tulu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=normal
#SBATCH --time=11:30:00
#SBATCH --output=./logs/train-tulu-%j.out
#SBATCH --error=./logs/train-tulu-%j.err

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-/iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/configs/zero3.yaml}" \
TRAIN_CONFIG="${TRAIN_CONFIG:-/iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/configs/sft_tulu_chat.yaml}" \
srun --mpi=pmix --network=disable_rdzv_get --environment=ngc_25-11-nemo-alps3 --export=ACCELERATE_CONFIG,TRAIN_CONFIG \
  /iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/scripts/multi_node.sh
