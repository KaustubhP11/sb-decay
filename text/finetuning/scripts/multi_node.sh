#!/usr/bin/env bash
set -euo pipefail
set -x

cd /iopsstor/scratch/cscs/kponkshe/sb-decay

# Conda activate/deactivate hooks may read unset CONDA_BACKUP_* vars.
# Run activation with nounset disabled, then restore strict mode.
set +u
source /users/kponkshe/miniconda3/etc/profile.d/conda.sh
conda activate .oss
set -u
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export ACCELERATE_LOG_LEVEL=info

mkdir -p $SCRATCH/triton_cache

export TRITON_CACHE_DIR=$SCRATCH/triton_cache

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-text/finetuning/configs/zero3.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-text/finetuning/configs/sft_full.yaml}"

ACCEL_PROCS=$(( $SLURM_NNODES * $SLURM_GPUS_PER_NODE ))

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MASTER_PORT=12802



accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "$ACCEL_PROCS" \
  --num_machines "${SLURM_NNODES}" \
  --machine_rank $SLURM_NODEID \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  text/finetuning/sft_train.py \
  --config "${TRAIN_CONFIG}"
