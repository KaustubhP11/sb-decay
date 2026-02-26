#!/usr/bin/env bash
#SBATCH --job-name=dolma-tokenize
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=10:00:00
#SBATCH --output=/iopsstor/scratch/cscs/kponkshe/sb-decay/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/kponkshe/sb-decay/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/iopsstor/scratch/cscs/kponkshe/sb-decay"
SCRIPT_PATH="${REPO_ROOT}/text/data/dolma/scripts/tokenize_dolma.sh"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

echo "Starting Dolma data prep+tokenization"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Start: $(date)"

bash "${SCRIPT_PATH}"

echo "Done: $(date)"
