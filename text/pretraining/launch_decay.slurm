#!/bin/bash
#SBATCH --job-name=smollm3-decay
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err
#SBATCH --time=11:30:00

set -e

CONFIG_ORIG="/iopsstor/scratch/cscs/kponkshe/sb-decay/text/pretraining/smollm3/run_decay.yaml"
CKPT_DIR="/iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/run_decay_dolma"

cd /iopsstor/scratch/cscs/kponkshe/nanotron

TRAINER_PYTHON_FILE="run_train.py"

# ──────────────────────────────────────────────────────────────
# Resume detection
# ──────────────────────────────────────────────────────────────
LATEST_FILE="$CKPT_DIR/latest.txt"

if [ -f "$LATEST_FILE" ]; then
    LATEST_STEP=$(cat "$LATEST_FILE")
    echo "Resuming from checkpoint at step $LATEST_STEP"
    CONFIG_PATH_YAML="/iopsstor/scratch/cscs/kponkshe/sb-decay/text/pretraining/smollm3/.resume_decay_${SLURM_JOB_ID}.yaml"
    python3 -c "
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

# ──────────────────────────────────────────────────────────────
# WandB
# ──────────────────────────────────────────────────────────────
export WANDB_INIT_TIMEOUT=300

# ──────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────
nvidia-smi

echo "python3 version = $(python3 --version)"
echo "Python path: $(which python3)"
echo "NCCL version: $(python -c 'import torch;print(torch.cuda.nccl.version())')"
echo "CUDA version: $(python -c 'import torch;print(torch.version.cuda)')"

echo "START TIME: $(date)"
start=$(date +%s)
echo "$(date -d @${start} "+%Y-%m-%d %H:%M:%S"): ${SLURM_JOB_NAME} start id=${SLURM_JOB_ID}"

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((15000 + SLURM_JOB_ID % 5000))
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export CUDA_DEVICE_MAX_CONNECTIONS="1"

echo "Number of nodes: $COUNT_NODE"
echo "Hostnames: $HOSTNAMES"

CMD=" $TRAINER_PYTHON_FILE --config-file $CONFIG_PATH_YAML"
export LAUNCHER="torchrun \
    --nproc_per_node 4 \
    --nnodes $COUNT_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    --tee 3 \
    "

random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

srun $SRUN_ARGS --ntasks=$COUNT_NODE --ntasks-per-node=1 -u bash -c "$LAUNCHER --node_rank \$SLURM_PROCID --role \$SLURMD_NODENAME: $CMD"

echo "END TIME: $(date)"

# Clean up temp resume config
if [[ "$CONFIG_PATH_YAML" == *".resume_"* ]]; then
    rm -f "$CONFIG_PATH_YAML"
fi
