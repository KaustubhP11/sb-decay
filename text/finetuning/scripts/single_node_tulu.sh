#!/usr/bin/env bash
set -euo pipefail

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-text/finetuning/configs/zero3.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-text/finetuning/configs/sft_tulu_chat.yaml}"

export ACCELERATE_CONFIG
export TRAIN_CONFIG

bash text/finetuning/scripts/single_node.sh
