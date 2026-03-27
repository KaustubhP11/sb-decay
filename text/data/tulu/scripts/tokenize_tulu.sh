#!/bin/bash
set -euo pipefail

python3 /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/tulu/prepare_tulu.py \
  --dataset_id allenai/tulu-3-sft-mixture \
  --split train \
  --output_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/tulu/processed \
  --num_proc 16 \
  --preview 3 \
  --parquet_shards 256

python3 /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/tulu/tokenize_tulu.py \
  --input_parquet_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/tulu/processed/raw_parquet \
  --output_tokenized_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/tulu/processed/tokenized \
  --tokenizer meta-llama/Llama-3.2-1B \
  --tasks 256 \
  --workers 64 \
  --clean_output
