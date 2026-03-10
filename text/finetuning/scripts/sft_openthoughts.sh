#!/usr/bin/env bash
cd text/finetuning
MODEL_PATH=/iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_stable_dolma/hf
accelerate launch train.py \
  --model_id  $MODEL_PATH \
  --tokenizer_id $MODEL_PATH \
  --dataset_name open-thoughts/OpenThoughts2-1M \
  --subset data \
  --split train \
  --answer_only_loss true \
  --question_field question \
  --answer_field answer \
  --max_seq_length 4096 \
  --micro_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_steps 3900 \
  --learning_rate 1e-4 \
  --bf16 true \
  --push_to_hub false \
  --output_dir ${MODEL_PATH}/sft_openthoughts \

