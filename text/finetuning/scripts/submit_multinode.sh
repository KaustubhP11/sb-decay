#!/bin/bash
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

if [[ "${NNODES}" -gt 1 ]]; then
  export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
else
  export MASTER_ADDR="127.0.0.1"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

srun --ntasks="${NNODES}" --ntasks-per-node=1 bash -lc '
  export NODE_RANK="${SLURM_NODEID}"
  bash text/finetuning/scripts/multi_node.sh
'
