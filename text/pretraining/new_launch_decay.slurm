#!/bin/bash
#SBATCH --job-name=smollm3-decay
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err
#SBATCH --time=11:30:00

set -euo pipefail

echo "START TIME: $(date)"
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')

export MASTER_ADDR
export COUNT_NODE
export HOSTNAMES

echo "START TIME: $(date)"
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}"

srun --ntasks="${COUNT_NODE}" --ntasks-per-node=1 --mpi=pmix --network=disable_rdzv_get --environment=ngc_25-11-nemo-alps3 --export=MASTER_ADDR,COUNT_NODE,HOSTNAMES /iopsstor/scratch/cscs/kponkshe/sb-decay/text/pretraining/scripts/multi_node_pretraining.sh

echo "END TIME: $(date)"
