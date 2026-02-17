#!/bin/bash

python utils/convert_hf_to_nanotron.py \
--hf-checkpoint-path /iopsstor/scratch/cscs/kponkshe/sb-decay/ckpts/SmolLM3-3B/stage2-step-4200000/hf \
--output-path /iopsstor/scratch/cscs/kponkshe/sb-decay/ckpts/SmolLM3-3B/stage2-step-4200000/nanotron \
--nanotron-repo /iopsstor/scratch/cscs/kponkshe/nanotron