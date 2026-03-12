#!/usr/bin/env bash
set -euo pipefail
set -x

mkdir -p $SCRATCH/triton_cache

export TRITON_CACHE_DIR=$SCRATCH/triton_cache

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-text/finetuning/configs/zero3.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-text/finetuning/configs/sft_full.yaml}"

ACCEL_PROCS=$(( $SLURM_NNODES * $SLURM_GPUS_PER_NODE ))

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MASTER_PORT=12802



accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "$SLURM_GPUS_PER_NODE" \
  --num_machines "${SLURM_NNODES}" \
  --machine_rank $SLURM_PROCID \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  text/finetuning/sft_train.py \
  --config "${TRAIN_CONFIG}"
