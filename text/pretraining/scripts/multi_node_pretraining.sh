#!/usr/bin/env bash
set -euo pipefail

cd /iopsstor/scratch/cscs/kponkshe/nanotron



# if [ "${SLURM_PROCID}" = "0" ]; then
#     rm -rf .nano
# fi

# sleep 10

# uv venv --python 3.10 .nano 
source .nano/bin/activate
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

# uv pip install numpy
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# uv pip install setuptools
# uv pip install --no-build-isolation flash-attn==2.5.8
# uv pip install datasets transformers wandb dacite pyyaml psutil
# uv pip install -e .



export NCCL_DEBUG=WARN
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 5000))
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_API_KEY="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"
echo "Number of nodes: $COUNT_NODE"
echo "Hostnames: $HOSTNAMES"
which python
python --version

CONFIG_ORIG="/iopsstor/scratch/cscs/kponkshe/sb-decay/text/pretraining/smollm3/run_stable_no_reasoning.yaml"
CKPT_DIR="/iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_stable_dolma_no_reasoning"
LATEST_FILE="$CKPT_DIR/latest.txt"

if [ -f "$LATEST_FILE" ]; then
    LATEST_STEP=$(cat "$LATEST_FILE")
    echo "Resuming from checkpoint at step $LATEST_STEP"
    CONFIG_PATH_YAML="/iopsstor/scratch/cscs/kponkshe/sb-decay/text/pretraining/smollm3/run_stable_no_reasoning/.resume_stable_no_reasoning_${SLURM_JOB_ID}.yaml"
    python -c "
import yaml
with open('${CONFIG_ORIG}') as f:
    cfg = yaml.safe_load(f)
cfg['checkpoints']['resume_checkpoint_path'] = '${CKPT_DIR}'
cfg['checkpoints']['load_optimizer'] = True
cfg['checkpoints']['load_lr_scheduler'] = True
with open('${CONFIG_PATH_YAML}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    echo "Resume config: ${CONFIG_PATH_YAML}"
else
    echo "No existing checkpoint — starting fresh"
    CONFIG_PATH_YAML="$CONFIG_ORIG"
fi
export LAUNCHER="torchrun \
    --nproc_per_node 4 \
    --nnodes $COUNT_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    --tee 3 \
    "
$LAUNCHER --node_rank "${SLURM_NODEID:-$SLURM_PROCID}" run_train.py --config-file "$CONFIG_PATH_YAML"

if [[ "$CONFIG_PATH_YAML" == *".resume_"* ]]; then
    rm -f "$CONFIG_PATH_YAML"
fi