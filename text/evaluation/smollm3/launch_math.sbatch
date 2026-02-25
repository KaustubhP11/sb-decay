#!/usr/bin/env bash
#SBATCH --job-name=basic-evals
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/iopsstor/scratch/cscs/kponkshe/sb-decay/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/kponkshe/sb-decay/logs/%x-%j.err

set -euo pipefail

mkdir -p /iopsstor/scratch/cscs/kponkshe/sb-decay/logs

SCRIPT_DIR="/iopsstor/scratch/cscs/kponkshe/sb-decay/text/evaluation/smollm3/"

bash "${SCRIPT_DIR}/math_eval.sh"