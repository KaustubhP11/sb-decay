#!/usr/bin/env bash
set -euo pipefail

# Run basic evals on the model

MODEL_PATH="/iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_decay_openthoughts/hf"
# Run basic evals on the model
lm_eval --model vllm \
    --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks aime,gsm8k,minerva_math \
    --gen_kwargs max_new_tokens=2048 max_gen_toks=2048\
    --output_path /iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_decay_openthoughts/hf/math_results.json \
    --batch_size auto

