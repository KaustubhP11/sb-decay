#!/usr/bin/env bash
#SBATCH --job-name=smollm3-lm-post-think
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

bash text/evaluation/smollm3/scripts/post_think.sh
