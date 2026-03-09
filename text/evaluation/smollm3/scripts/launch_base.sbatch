#!/usr/bin/env bash
#SBATCH --job-name=smollm3-lm-base
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

bash text/evaluation/smollm3/scripts/base.sh
