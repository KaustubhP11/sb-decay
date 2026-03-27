#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/ckpts/SmolLM3-3B/stage1-step-2000000/hf}"
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
TMP_OUTPUT_DIR="$(mktemp -d "${OUTPUT_DIR}/.lm_eval_tmp.XXXXXX")"
trap 'rm -rf "${TMP_OUTPUT_DIR}"' EXIT

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

RESULT_FILES=()

run_eval() {
  local shot_label="$1"
  local num_fewshot="$2"
  local tasks="$3"
  local gen_kwargs="$4"
  local shot_output_dir="${TMP_OUTPUT_DIR}/${shot_label}"
  local result_file

  if [[ -z "${tasks}" ]]; then
    return 0
  fi

  lm_eval "${COMMON_ARGS[@]}" \
    --tasks "${tasks}" \
    --num_fewshot "${num_fewshot}" \
    --gen_kwargs "${gen_kwargs}" \
    --output_path "${shot_output_dir}"

  result_file="$(python - "${shot_output_dir}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
result_files = sorted(output_dir.rglob("*.json"))
for path in reversed(result_files):
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            print(path)
            break
    except Exception:
        continue
else:
    raise SystemExit(f"No lm-eval result JSON found under {output_dir}")
PY
)"
  RESULT_FILES+=("${result_file}")
}

run_eval "0shot" 0 "${TASKS_0SHOT}" "${GEN_KWARGS_0SHOT}"
run_eval "5shot" 5 "${TASKS_5SHOT}" "${GEN_KWARGS_5SHOT}"

# if [[ -n "${TASKS_4SHOT}" ]]; then
#   run_eval "4shot" 4 "${TASKS_4SHOT}" "${GEN_KWARGS_4SHOT}"
# fi

if [[ "${#RESULT_FILES[@]}" -eq 0 ]]; then
  echo "No evaluation tasks were configured; nothing to merge." >&2
  exit 1
fi

COMBINED_OUTPUT_PATH="$(python - "${OUTPUT_DIR}" <<'PY'
from datetime import datetime
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
print(output_dir / f"results_{timestamp}.json")
PY
)"

python - "${COMBINED_OUTPUT_PATH}" "${RESULT_FILES[@]}" <<'PY'
import json
import sys
from copy import deepcopy
from pathlib import Path


def merge_task_map(merged, incoming, key):
    for task, value in incoming.items():
        if task in merged[key]:
            raise SystemExit(f"Duplicate task '{task}' while merging '{key}'")
        merged[key][task] = value


output_path = Path(sys.argv[1])
source_paths = [Path(arg) for arg in sys.argv[2:]]

merged = None
merged_time = 0.0
total_eval_time = 0.0
source_runs = []

for source_path in source_paths:
    with source_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if merged is None:
        merged = deepcopy(data)
        merged.setdefault("results", {})
        merged.setdefault("group_subtasks", {})
        merged.setdefault("configs", {})
        merged.setdefault("versions", {})
        merged.setdefault("n-shot", {})
        merged.setdefault("higher_is_better", {})
        merged.setdefault("n-samples", {})
        merged.setdefault("task_hashes", {})
        merged["results"].clear()
        merged["group_subtasks"].clear()
        merged["configs"].clear()
        merged["versions"].clear()
        merged["n-shot"].clear()
        merged["higher_is_better"].clear()
        merged["n-samples"].clear()
        merged["task_hashes"].clear()
    elif data.get("model_name") != merged.get("model_name"):
        raise SystemExit(
            f"Cannot merge lm-eval outputs from different models: {merged.get('model_name')} vs {data.get('model_name')}"
        )

    merge_task_map(merged, data.get("results", {}), "results")
    merge_task_map(merged, data.get("group_subtasks", {}), "group_subtasks")
    merge_task_map(merged, data.get("configs", {}), "configs")
    merge_task_map(merged, data.get("versions", {}), "versions")
    merge_task_map(merged, data.get("n-shot", {}), "n-shot")
    merge_task_map(merged, data.get("higher_is_better", {}), "higher_is_better")
    merge_task_map(merged, data.get("n-samples", {}), "n-samples")
    merge_task_map(merged, data.get("task_hashes", {}), "task_hashes")

    source_runs.append(
        {
            "source_file": str(source_path),
            "tasks": sorted((data.get("results") or {}).keys()),
            "n_shot": data.get("n-shot", {}),
            "gen_kwargs": (data.get("config") or {}).get("gen_kwargs"),
        }
    )

    merged_time = max(float(data.get("date", 0.0) or 0.0), merged_time)
    total_eval_time += float(data.get("total_evaluation_time_seconds", 0.0) or 0.0)

merged["date"] = merged_time
merged["total_evaluation_time_seconds"] = str(total_eval_time)
merged["source_runs"] = source_runs

with output_path.open("w", encoding="utf-8") as handle:
    json.dump(merged, handle, indent=2)
    handle.write("\n")

print(output_path)
PY

echo "Merged lm-eval results written to: ${COMBINED_OUTPUT_PATH}"
