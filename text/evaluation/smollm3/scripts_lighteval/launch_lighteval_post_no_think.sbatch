#!/usr/bin/env bash
#SBATCH --job-name=smollm3-post-no-think-evals
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

bash text/evaluation/smollm3/scripts_lighteval/lighteval_post_no_think.sh
