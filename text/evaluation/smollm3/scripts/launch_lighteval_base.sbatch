#!/usr/bin/env bash
#SBATCH --job-name=smollm3-base-evals
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

# Base SmolLM3 LightEval profile.
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/checkpoints/run_stable_dolma/hf}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/evals/base}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_PREFIX="${SLURM_JOB_NAME:-smollm3-base-evals}-${SLURM_JOB_ID:-manual}"

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/${LOG_PREFIX}.out") 2> >(tee -a "${LOG_DIR}/${LOG_PREFIX}.err" >&2)

MODEL_PATH="${MODEL_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash "${SCRIPT_DIR}/lighteval_base.sh"
