#!/usr/bin/env bash
set -euo pipefail

cd /iopsstor/scratch/cscs/kponkshe/sb-decay

set -a 
source .env
set +a

# set -x

# Conda activate/deactivate hooks may read unset CONDA_BACKUP_* vars.
# Run activation with nounset disabled, then restore strict mode.
# set +u
# source /users/kponkshe/miniconda3/etc/profile.d/conda.sh
# conda activate .oss
# set -u

# Keep Python imports bound to the active Conda env.
# unset PYTHONHOME
# export PYTHONNOUSERSITE=1
# PYTHONPATH_CLEANED="$(printf '%s' "${PYTHONPATH:-}" | awk -v RS=: -v ORS=: '
#   $0 !~ "^/usr/local/lib/python[0-9.]+/dist-packages(/|$)" &&
#   $0 !~ "^/usr/lib/python[0-9.]+/dist-packages(/|$)" &&
#   $0 != ""
# ')"
# PYTHONPATH_CLEANED="${PYTHONPATH_CLEANED%:}"
# if [ -n "${PYTHONPATH_CLEANED}" ]; then
#   export PYTHONPATH="${PYTHONPATH_CLEANED}"
# else
#   unset PYTHONPATH
# fi

# # Avoid mixing system CUDA libs with Conda/PyTorch CUDA libs.
# # The mismatch commonly shows up as missing nvJitLink symbols in libcusparse.
# LD_LIBRARY_PATH_CLEANED=""
# unset LD_PRELOAD

# TORCH_NVIDIA_LIBS="$(python -c 'import os,site; sp=next((p for p in site.getsitepackages() if os.path.isdir(p)),""); c=[os.path.join(sp,"nvidia","nvjitlink","lib"), os.path.join(sp,"nvidia","cusparse","lib"), os.path.join(sp,"nvidia","cublas","lib"), os.path.join(sp,"nvidia","cuda_runtime","lib"), os.path.join(sp,"nvidia","cudnn","lib")]; print(":".join(p for p in c if os.path.isdir(p)))')"
pip install --no-build-isolation --no-cache-dir -r text/finetuning/requirements.txt "torch==$(python -c 'import torch; print(torch.__version__)')"

# export LD_LIBRARY_PATH="${TORCH_NVIDIA_LIBS:+${TORCH_NVIDIA_LIBS}:}${CONDA_PREFIX}/lib${LD_LIBRARY_PATH_CLEANED:+:${LD_LIBRARY_PATH_CLEANED}}"
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# export ACCELERATE_LOG_LEVEL=info

SCRATCH_BASE="${SCRATCH:-/tmp/${USER}}"
mkdir -p "${SCRATCH_BASE}/triton_cache"

export TRITON_CACHE_DIR="${SCRATCH_BASE}/triton_cache"

# Shared/mounted filesystems may not support flock(), which HF datasets uses.
# Keep HF caches on node-local storage to avoid "No locks available".
HF_CACHE_BASE="${SCRATCH_BASE}/hf-cache"
mkdir -p "${HF_CACHE_BASE}/datasets" "${HF_CACHE_BASE}/hub"
export HF_HOME="${HF_CACHE_BASE}"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export HF_HUB_CACHE="${HF_CACHE_BASE}/hub"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-text/finetuning/configs/zero3.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-text/finetuning/configs/sft_full.yaml}"
# PYTHON_BIN="${CONDA_PREFIX}/bin/python"

ACCEL_PROCS=$(( $SLURM_NNODES * $SLURM_GPUS_PER_NODE ))

MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
MASTER_PORT=12802



accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "$ACCEL_PROCS" \
  --num_machines "${SLURM_NNODES}" \
  --machine_rank $SLURM_NODEID \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  text/finetuning/sft_train.py \
  --config "${TRAIN_CONFIG}"
