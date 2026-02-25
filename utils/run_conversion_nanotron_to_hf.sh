#!/bin/bash

# Nanotron -> Hugging Face conversion (SmolLM2/Llama-style checkpoints)
# Update paths below before running.
NANOTRON_REPO="/iopsstor/scratch/cscs/kponkshe/nanotron"
NANOTRON_CHECKPOINT="/iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_stable_openthoughts/1772"
HF_OUTPUT_PATH="/iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_stable_openthoughts/hf"
TOKENIZER_NAME_OR_PATH="meta-llama/Llama-3.2-1B"
# Options: LlamaConfig, Qwen2Config, or AUTO (recommended)
CONFIG_CLS="AUTO"

cd "$NANOTRON_REPO" || exit 1

if [ "$CONFIG_CLS" = "AUTO" ]; then
  CONFIG_CLS="$(NANOTRON_CHECKPOINT="$NANOTRON_CHECKPOINT" python - <<'PY'
import os
import json
from pathlib import Path

ckpt = Path(os.environ["NANOTRON_CHECKPOINT"])
cfg = ckpt / "model_config.json"
with cfg.open("r", encoding="utf-8") as f:
    data = json.load(f)
print("Qwen2Config" if data.get("is_qwen2_config", False) else "LlamaConfig")
PY
)"
fi

echo "Using config class: $CONFIG_CLS"

torchrun --nproc_per_node=1 --module examples.llama.convert_nanotron_to_hf \
  --checkpoint_path "$NANOTRON_CHECKPOINT" \
  --save_path "$HF_OUTPUT_PATH" \
  --tokenizer_name "$TOKENIZER_NAME_OR_PATH" \
  --config_cls "$CONFIG_CLS"
