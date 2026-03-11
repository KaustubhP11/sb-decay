# SmolLM3-3B evaluation scripts

## Goal

This directory is organized for one specific workflow:
- keep the original SmolLM-style **LightEval** runs as reference
- run the **same profile types** with **lm-eval-harness** using simple scripts

So there are two folders:
- `scripts_lighteval/`: reference scripts (LightEval)
- `scripts/`: replication scripts (lm-eval)

## Profile Mapping

- Base profile:
  - LightEval: `scripts_lighteval/lighteval_base.sh`
  - lm-eval: `scripts/base.sh`
- Mid-reasoning profile:
  - LightEval: `scripts_lighteval/lighteval_mid_reasoning.sh`
  - lm-eval: `scripts/mid_reasoning.sh`
- Post profile (`/no_think`):
  - LightEval: `scripts_lighteval/lighteval_post_no_think.sh`
  - lm-eval: `scripts/post_no_think.sh`
- Post profile (`/think`):
  - LightEval: `scripts_lighteval/lighteval_post_think.sh`
  - lm-eval: `scripts/post_think.sh`

## Important Scope

The lm-eval scripts are designed to replicate the same **evaluation intent** (profile, shots, chat/system behavior), but task IDs and prompt internals can differ by framework/version.

That is why lm-eval scripts expose overrides like:
- `TASKS=...`
- `INCLUDE_PATH=/path/to/custom_lm_eval_yaml_tasks`

Use those when a task name or prompt needs exact matching.

## Setup

Use `python>=3.11`.

lm-eval env:

```sh
pip install uv
uv venv smol3_venv --python 3.11
source smol3_venv/bin/activate
uv pip install -r text/evaluation/smollm3/requirements_lm_eval.txt
```

LightEval env:

```sh
pip install uv
uv venv smol3_venv --python 3.11
source smol3_venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -r text/evaluation/smollm3/requirements.txt
```

## Run (lm-eval)

```sh
bash text/evaluation/smollm3/scripts/base.sh
bash text/evaluation/smollm3/scripts/mid_reasoning.sh
bash text/evaluation/smollm3/scripts/post_no_think.sh
bash text/evaluation/smollm3/scripts/post_think.sh
```

Base profile details in lm-eval:
- 0-shot: core MC/NLU tasks
- 5-shot: `gsm8k`
- 4-shot: `hendrycks_math`

## Run (LightEval reference)

```sh
bash text/evaluation/smollm3/scripts_lighteval/lighteval_base.sh
bash text/evaluation/smollm3/scripts_lighteval/lighteval_mid_reasoning.sh
bash text/evaluation/smollm3/scripts_lighteval/lighteval_post_no_think.sh
bash text/evaluation/smollm3/scripts_lighteval/lighteval_post_think.sh
```

## Common Overrides

```sh
MODEL_PATH=/path/to/checkpoint \
MAX_MODEL_LEN=4096 \
TASKS="mmlu,hellaswag" \
bash text/evaluation/smollm3/scripts/base.sh
```

```sh
MODEL_PATH=/path/to/checkpoint \
INCLUDE_PATH=/path/to/custom_lm_eval_tasks \
TASKS="my_custom_task" \
bash text/evaluation/smollm3/scripts/post_no_think.sh
```

Notes:
- Defaults target SmolLM-style 32k context.
- For 4k checkpoints, set:
  - lm-eval: `MAX_MODEL_LEN=4096`
  - LightEval: `MAX_MODEL_LENGTH=4096`

## SLURM

lm-eval:

```sh
sbatch text/evaluation/smollm3/scripts/launch_base.sbatch
sbatch text/evaluation/smollm3/scripts/launch_mid_reasoning.sbatch
sbatch text/evaluation/smollm3/scripts/launch_post_no_think.sbatch
sbatch text/evaluation/smollm3/scripts/launch_post_think.sbatch
```

LightEval:

```sh
sbatch text/evaluation/smollm3/scripts_lighteval/launch_lighteval_base.sbatch
sbatch text/evaluation/smollm3/scripts_lighteval/launch_lighteval_mid_reasoning.sbatch
sbatch text/evaluation/smollm3/scripts_lighteval/launch_lighteval_post_no_think.sbatch
sbatch text/evaluation/smollm3/scripts_lighteval/launch_lighteval_post_think.sbatch
```

## Plotting (decay vs no_decay)

Install plotting deps:

```sh
uv pip install -r text/evaluation/smollm3/plotting/requirements.txt
```

Generate comparison plots:

```sh
python text/evaluation/smollm3/plotting/plot_eval_comparison.py \
  --baseline /path/to/no_decay_checkpoint/evals/lm_eval \
  --candidate /path/to/decay_checkpoint/evals/lm_eval \
  --baseline_label no_decay \
  --candidate_label decay \
  --out_dir text/evaluation/smollm3/plotting/out
```

Notebook version:

`text/evaluation/smollm3/plotting/eval_comparison.ipynb`
