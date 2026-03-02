#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/run_decay_dolma/hf}"
OUTPUT_PATH="${OUTPUT_PATH:-${MODEL_PATH}/math_results.json}"
TASKS="${TASKS:-aime,gsm8k,minerva_math}"
LM_EVAL_MODEL="${LM_EVAL_MODEL:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DTYPE="${DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-2048}"

MODEL_ARGS="pretrained=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},data_parallel_size=${DATA_PARALLEL_SIZE}"

if [[ -n "${THINK_END_TOKEN:-}" ]]; then
    MODEL_ARGS+=",think_end_token=${THINK_END_TOKEN}"
fi

lm_eval --model "${LM_EVAL_MODEL}" \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS}" "max_gen_toks=${MAX_GEN_TOKS}" \
    --output_path "${OUTPUT_PATH}" \
    --batch_size "${BATCH_SIZE}"
