#!/usr/bin/env bash
#SBATCH --job-name=smollm3-base-evals
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --error=./logs/%x-%j.err
#SBATCH --output=./logs/%x-%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

bash text/evaluation/smollm3/scripts/lighteval_base.sh
