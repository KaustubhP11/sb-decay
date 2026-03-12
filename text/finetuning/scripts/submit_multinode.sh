#!/bin/bash -l
#SBATCH --job-name=smollm3-sft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --time=11:30:00
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err

set -euo pipefail
mkdir -p logs

export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
export NNODES="${SLURM_NNODES:-1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
export NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
bash text/finetuning/scripts/multi_node.sh
