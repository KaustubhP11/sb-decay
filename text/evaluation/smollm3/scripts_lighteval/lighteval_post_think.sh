#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

# Post-trained SmolLM3 LightEval profile with thinking enabled.
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/run_decay_dolma/hf}"
TASKS_FILE="${TASKS_FILE:-smollm3_instruct.txt}"
TASKS_SPEC="${TASKS_SPEC:-${EVAL_ROOT}/${TASKS_FILE}}"
CUSTOM_TASKS="${CUSTOM_TASKS:-${EVAL_ROOT}/tasks.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/evals/post-think}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-/think}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
GENERATION_PARAMETERS="${GENERATION_PARAMETERS:-{max_new_tokens:32768,temperature:0.6,top_p:0.95}}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

mkdir -p "${OUTPUT_DIR}"

MODEL_ARGS="model_name=${MODEL_PATH},dtype=${DTYPE},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},generation_parameters=${GENERATION_PARAMETERS}"

if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
    MODEL_ARGS+=",${EXTRA_MODEL_ARGS}"
fi

lighteval vllm \
    "${MODEL_ARGS}" \
    "${TASKS_SPEC}" \
    --use-chat-template \
    --system-prompt "${SYSTEM_PROMPT}" \
    --custom-tasks "${CUSTOM_TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-details
