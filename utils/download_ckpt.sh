#!/usr/bin/env bash
set -euo pipefail

REVISION="${1:-stage1-step-2000000}"
REPO_ID="HuggingFaceTB/SmolLM3-3B-checkpoints"
TARGET_DIR="/iopsstor/scratch/cscs/kponkshe/sb-decay/ckpts/SmolLM3-3B/${REVISION}/hf"

mkdir -p "${TARGET_DIR}"

python3 - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${REPO_ID}",
    revision="${REVISION}",
    local_dir="${TARGET_DIR}",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("Downloaded to: ${TARGET_DIR}")
PY