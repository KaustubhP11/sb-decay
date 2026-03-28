#!/bin/bash

python utils/convert_hf_to_nanotron.py \
--hf-checkpoint-path /iopsstor/scratch/cscs/kponkshe/sb-decay/ckpts/SmolLM3-3B/stage1-step-2000000/hf \
--output-path /iopsstor/scratch/cscs/kponkshe/sb-decay/ckpts/SmolLM3-3B/stage1-step-2000000/nanotron \
--nanotron-repo /iopsstor/scratch/cscs/kponkshe/nanotron