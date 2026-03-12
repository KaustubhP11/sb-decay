#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}/text/finetuning"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/text/finetuning/configs/zero3.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-${REPO_ROOT}/text/finetuning/configs/sft_full.yaml}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "$((GPUS_PER_NODE * NNODES))" \
  --num_machines "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  sft_train.py \
  --config "${TRAIN_CONFIG}"
