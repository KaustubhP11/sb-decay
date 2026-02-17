#!/bin/bash

python3 /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/openthoughts/prepare_openthoughts.py \
  --dataset_id open-thoughts/OpenThoughts2-1M \
  --split train \
  --output_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/openthoughts/processed \
  --num_proc 16 \
  --preview 3 \
  --parquet_shards 128

python3 /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/openthoughts/tokenize_openthoughts.py \
  --input_parquet_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/openthoughts/processed/raw_parquet \
  --output_tokenized_dir /iopsstor/scratch/cscs/kponkshe/sb-decay/text/data/openthoughts/processed/tokenized \
  --tokenizer meta-llama/Llama-3.2-1B \
  --tasks 128 \
  --workers 64 \
  --clean_output