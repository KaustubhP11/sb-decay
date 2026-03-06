#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/run_decay_dolma/hf}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/evals/lm_eval/mid-reasoning}"
LM_EVAL_MODEL="${LM_EVAL_MODEL:-vllm}"
TASKS="${TASKS:-aime25,gpqa_diamond,ifeval,lcb_codegeneration,gsm_plus,mixeval_hard}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
GEN_KWARGS="${GEN_KWARGS:-temperature=0.6,top_p=0.95,max_gen_toks=32768}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
INCLUDE_PATH="${INCLUDE_PATH:-}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

mkdir -p "${OUTPUT_DIR}"

MODEL_ARGS="pretrained=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN}"
if [[ -n "${THINK_END_TOKEN:-}" ]]; then
  MODEL_ARGS+=",think_end_token=${THINK_END_TOKEN}"
fi
if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
  MODEL_ARGS+=",${EXTRA_MODEL_ARGS}"
fi

CMD=(
  lm_eval
  --model "${LM_EVAL_MODEL}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --num_fewshot "${NUM_FEWSHOT}"
  --gen_kwargs "${GEN_KWARGS}"
  --output_path "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
)
if [[ "${APPLY_CHAT_TEMPLATE}" == "1" ]]; then
  CMD+=(--apply_chat_template)
fi
if [[ -n "${INCLUDE_PATH}" ]]; then
  CMD+=(--include_path "${INCLUDE_PATH}")
fi

"${CMD[@]}"
