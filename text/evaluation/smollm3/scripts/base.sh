#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/run_decay_dolma/hf}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/evals/lm_eval/base}"
LM_EVAL_MODEL="${LM_EVAL_MODEL:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
INCLUDE_PATH="${INCLUDE_PATH:-}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"

# SmolLM3 base-equivalent splits: 0-shot MC/NLU, 5-shot GSM8K, 4-shot MATH.
TASKS_0SHOT="${TASKS_0SHOT:-hellaswag,arc_easy,arc_challenge,mmlu,mmlu_pro,boolq,commonsense_qa,winogrande,openbookqa,piqa}"
TASKS_5SHOT="${TASKS_5SHOT:-gsm8k}"
TASKS_4SHOT="${TASKS_4SHOT:-hendrycks_math}"

GEN_KWARGS_0SHOT="${GEN_KWARGS_0SHOT:-temperature=0}"
GEN_KWARGS_5SHOT="${GEN_KWARGS_5SHOT:-temperature=0,max_gen_toks=256}"
GEN_KWARGS_4SHOT="${GEN_KWARGS_4SHOT:-temperature=0,max_gen_toks=2048}"

mkdir -p "${OUTPUT_DIR}"

MODEL_ARGS="pretrained=${MODEL_PATH},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN}"

if [[ -n "${THINK_END_TOKEN:-}" ]]; then
  MODEL_ARGS+=",think_end_token=${THINK_END_TOKEN}"
fi
if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
  MODEL_ARGS+=",${EXTRA_MODEL_ARGS}"
fi

COMMON_ARGS=(
  --model "${LM_EVAL_MODEL}"
  --model_args "${MODEL_ARGS}"
  --batch_size "${BATCH_SIZE}"
)
if [[ -n "${INCLUDE_PATH}" ]]; then
  COMMON_ARGS+=(--include_path "${INCLUDE_PATH}")
fi

if [[ -n "${TASKS_0SHOT}" ]]; then
  lm_eval "${COMMON_ARGS[@]}" \
    --tasks "${TASKS_0SHOT}" \
    --num_fewshot 0 \
    --gen_kwargs "${GEN_KWARGS_0SHOT}" \
    --output_path "${OUTPUT_DIR}/0shot"
fi

if [[ -n "${TASKS_5SHOT}" ]]; then
  lm_eval "${COMMON_ARGS[@]}" \
    --tasks "${TASKS_5SHOT}" \
    --num_fewshot 5 \
    --gen_kwargs "${GEN_KWARGS_5SHOT}" \
    --output_path "${OUTPUT_DIR}/5shot"
fi

# if [[ -n "${TASKS_4SHOT}" ]]; then
#   lm_eval "${COMMON_ARGS[@]}" \
#     --tasks "${TASKS_4SHOT}" \
#     --num_fewshot 0 \
#     --gen_kwargs "${GEN_KWARGS_4SHOT}" \
#     --output_path "${OUTPUT_DIR}/4shot"
# fi
