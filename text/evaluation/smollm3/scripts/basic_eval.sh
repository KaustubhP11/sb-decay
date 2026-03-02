#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/run_stable_dolma/hf}"
OUTPUT_PATH="${OUTPUT_PATH:-${MODEL_PATH}/aggregated_results.json}"
TASKS="${TASKS:-mmlu,piqa,hellaswag,arc_challenge,arc_easy,openbookqa,commonsense_qa}"
LM_EVAL_MODEL="${LM_EVAL_MODEL:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DTYPE="${DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

MODEL_ARGS="pretrained=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},data_parallel_size=${DATA_PARALLEL_SIZE}"

if [[ -n "${THINK_END_TOKEN:-}" ]]; then
    MODEL_ARGS+=",think_end_token=${THINK_END_TOKEN}"
fi

lm_eval --model "${LM_EVAL_MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --output_path "${OUTPUT_PATH}" \
    --batch_size "${BATCH_SIZE}"
