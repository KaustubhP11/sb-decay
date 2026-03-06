#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

# Base SmolLM3 LightEval profile.
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/run_stable_dolma/hf}"
TASKS_FILE="${TASKS_FILE:-smollm3_base.txt}"
TASKS_SPEC="${TASKS_SPEC:-${EVAL_ROOT}/${TASKS_FILE}}"
CUSTOM_TASKS="${CUSTOM_TASKS:-${EVAL_ROOT}/tasks.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/evals/base}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
GENERATION_PARAMETERS="${GENERATION_PARAMETERS:-{temperature:0}}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

mkdir -p "${OUTPUT_DIR}"

MODEL_ARGS="model_name=${MODEL_PATH},dtype=${DTYPE},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},max_model_length=${MAX_MODEL_LENGTH},max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},generation_parameters=${GENERATION_PARAMETERS}"

if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
    MODEL_ARGS+=",${EXTRA_MODEL_ARGS}"
fi

export HF_HOME="/capstor/scratch/cscs/kponkshe/hf-home-22"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "Using temporary HF cache at: ${HF_HOME}"

lighteval vllm \
    "${MODEL_ARGS}" \
    "${TASKS_SPEC}" \
    --custom-tasks "${CUSTOM_TASKS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-details
