#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROFILE="${1:-${SMOLLM3_PROFILE:-base}}"

case "${PROFILE}" in
    base|foundation)
        PROFILE_SLUG="base"
        DEFAULT_MODEL_NAME="HuggingFaceTB/SmolLM3-3B-Base"
        DEFAULT_TASKS_FILE="smollm3_base.txt"
        DEFAULT_REVISION=""
        DEFAULT_USE_CHAT_TEMPLATE=0
        DEFAULT_SYSTEM_PROMPT=""
        DEFAULT_DTYPE="bfloat16"
        DEFAULT_TENSOR_PARALLEL_SIZE=2
        DEFAULT_MAX_MODEL_LENGTH=32768
        DEFAULT_MAX_NUM_BATCHED_TOKENS=32768
        DEFAULT_GPU_MEMORY_UTILIZATION=0.7
        DEFAULT_GENERATION_PARAMETERS="{temperature:0}"
        ;;
    mid|mid-reasoning|reasoning)
        PROFILE_SLUG="mid-reasoning"
        DEFAULT_MODEL_NAME="HuggingFaceTB/SmolLM3-3B-checkpoints"
        DEFAULT_TASKS_FILE="smollm3_instruct.txt"
        DEFAULT_REVISION="it-mid-training"
        DEFAULT_USE_CHAT_TEMPLATE=1
        DEFAULT_SYSTEM_PROMPT=""
        DEFAULT_DTYPE="bfloat16"
        DEFAULT_TENSOR_PARALLEL_SIZE=2
        DEFAULT_MAX_MODEL_LENGTH=32768
        DEFAULT_MAX_NUM_BATCHED_TOKENS=""
        DEFAULT_GPU_MEMORY_UTILIZATION=0.8
        DEFAULT_GENERATION_PARAMETERS="{max_new_tokens:32768,temperature:0.6,top_p:0.95}"
        ;;
    post|post-no-think|instruct)
        PROFILE_SLUG="post-no-think"
        DEFAULT_MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
        DEFAULT_TASKS_FILE="smollm3_instruct.txt"
        DEFAULT_REVISION=""
        DEFAULT_USE_CHAT_TEMPLATE=1
        DEFAULT_SYSTEM_PROMPT="/no_think"
        DEFAULT_DTYPE="bfloat16"
        DEFAULT_TENSOR_PARALLEL_SIZE=2
        DEFAULT_MAX_MODEL_LENGTH=32768
        DEFAULT_MAX_NUM_BATCHED_TOKENS=""
        DEFAULT_GPU_MEMORY_UTILIZATION=0.8
        DEFAULT_GENERATION_PARAMETERS="{max_new_tokens:32768,temperature:0.6,top_p:0.95}"
        ;;
    post-think|think)
        PROFILE_SLUG="post-think"
        DEFAULT_MODEL_NAME="HuggingFaceTB/SmolLM3-3B"
        DEFAULT_TASKS_FILE="smollm3_instruct.txt"
        DEFAULT_REVISION=""
        DEFAULT_USE_CHAT_TEMPLATE=1
        DEFAULT_SYSTEM_PROMPT="/think"
        DEFAULT_DTYPE="bfloat16"
        DEFAULT_TENSOR_PARALLEL_SIZE=2
        DEFAULT_MAX_MODEL_LENGTH=32768
        DEFAULT_MAX_NUM_BATCHED_TOKENS=""
        DEFAULT_GPU_MEMORY_UTILIZATION=0.8
        DEFAULT_GENERATION_PARAMETERS="{max_new_tokens:32768,temperature:0.6,top_p:0.95}"
        ;;
    *)
        echo "Unsupported SmolLM3 LightEval profile: ${PROFILE}" >&2
        echo "Expected one of: base, mid-reasoning, post-no-think, post-think" >&2
        exit 1
        ;;
esac

MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
MODEL_REVISION="${MODEL_REVISION:-${DEFAULT_REVISION}}"
TASKS_FILE="${TASKS_FILE:-${DEFAULT_TASKS_FILE}}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-${DEFAULT_USE_CHAT_TEMPLATE}}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-${DEFAULT_SYSTEM_PROMPT}}"
DTYPE="${DTYPE:-${DEFAULT_DTYPE}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${DEFAULT_TENSOR_PARALLEL_SIZE}}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-${DEFAULT_MAX_MODEL_LENGTH}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${DEFAULT_MAX_NUM_BATCHED_TOKENS}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEMORY_UTILIZATION}}"
GENERATION_PARAMETERS="${GENERATION_PARAMETERS:-${DEFAULT_GENERATION_PARAMETERS}}"
TASKS_SPEC="${TASKS_SPEC:-${EVAL_ROOT}/${TASKS_FILE}}"
CUSTOM_TASKS="${CUSTOM_TASKS:-${EVAL_ROOT}/tasks.py}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_ROOT}/evals/${PROFILE_SLUG}}"

mkdir -p "${OUTPUT_DIR}"

MODEL_ARGS_PARTS=(
    "model_name=${MODEL_NAME}"
    "dtype=${DTYPE}"
    "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    "max_model_length=${MAX_MODEL_LENGTH}"
    "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${MODEL_REVISION}" ]]; then
    MODEL_ARGS_PARTS+=("revision=${MODEL_REVISION}")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
    MODEL_ARGS_PARTS+=("max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}")
fi

if [[ -n "${GENERATION_PARAMETERS}" ]]; then
    MODEL_ARGS_PARTS+=("generation_parameters=${GENERATION_PARAMETERS}")
fi

if [[ -n "${EXTRA_MODEL_ARGS:-}" ]]; then
    MODEL_ARGS_PARTS+=("${EXTRA_MODEL_ARGS}")
fi

MODEL_ARGS="$(IFS=,; echo "${MODEL_ARGS_PARTS[*]}")"

CMD=(
    lighteval
    vllm
    "${MODEL_ARGS}"
    "${TASKS_SPEC}"
    --custom-tasks
    "${CUSTOM_TASKS}"
    --output-dir
    "${OUTPUT_DIR}"
    --save-details
)

if [[ "${USE_CHAT_TEMPLATE}" == "1" ]]; then
    CMD+=(--use-chat-template)
fi

if [[ -n "${SYSTEM_PROMPT}" ]]; then
    CMD+=(--system-prompt "${SYSTEM_PROMPT}")
fi

"${CMD[@]}"
