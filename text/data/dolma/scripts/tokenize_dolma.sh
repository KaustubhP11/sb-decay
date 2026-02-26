#!/bin/bash
set -euo pipefail

python3 /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/dolma/prepare_dolma.py \
  --dataset_id allenai/dolma3_dolmino_mix-10B-1025 \
  --split train \
  --output_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/dolma/processed \
  --num_proc 128 \
  --preview 1 \
  --parquet_shards 256

python3 /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/dolma/tokenize_dolma.py \
  --input_parquet_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/dolma/processed/raw_parquet \
  --output_tokenized_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/dolma/processed/tokenized \
  --tokenizer meta-llama/Llama-3.2-1B \
  --tasks 256 \
  --workers 64 \
  --clean_output
