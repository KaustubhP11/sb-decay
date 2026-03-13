#!/bin/bash
#SBATCH --job-name=smollm3-sft
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --output=./logs/train-%j.out
#SBATCH --error=./logs/train-%j.err

# srun  --environment=/iopsstor/scratch/cscs/kponkshe/Megatron-LM/ncg_new_v2.toml --export=NONE /iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/scripts/multi_node.sh
srun --mpi=pmix --network=disable_rdzv_get --environment=ngc_25-11-nemo-alps3 --export=NONE /iopsstor/scratch/cscs/kponkshe/sb-decay/text/finetuning/scripts/multi_node.sh