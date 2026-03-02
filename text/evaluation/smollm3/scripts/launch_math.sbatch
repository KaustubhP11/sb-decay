#!/usr/bin/env bash
#SBATCH --job-name=basic-evals
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SCRIPT_DIR="${SCRIPT_DIR:-${REPO_ROOT}/text/evaluation/smollm3/scripts}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_PREFIX="${SLURM_JOB_NAME:-basic-evals}-${SLURM_JOB_ID:-manual}"

mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/${LOG_PREFIX}.out") 2> >(tee -a "${LOG_DIR}/${LOG_PREFIX}.err" >&2)

bash "${SCRIPT_DIR}/math_eval.sh"
