#!/usr/bin/env bash
set -euo pipefail

# Run basic evals on the model

MODEL_PATH="/iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_stable_openthoughts/hf"
# Run basic evals on the model
lm_eval --model vllm \
    --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks mmlu,piqa,hellaswag,arc_challenge,arc_easy,openbookqa,commonsense_qa \
    --output_path /iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_stable_openthoughts/hf/aggregated_results.json \
    --batch_size auto

