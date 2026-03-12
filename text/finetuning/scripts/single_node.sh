#!/usr/bin/env bash
set -euo pipefail

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-text/finetuning/configs/zero3.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-text/finetuning/configs/sft_full.yaml}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${GPUS_PER_NODE}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${MASTER_PORT:-29500}" \
  text/finetuning/sft_train.py \
  --config "${TRAIN_CONFIG}"
