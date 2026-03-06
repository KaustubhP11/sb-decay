# SmolLM3-3B evaluation scripts

There are now two script folders:
- `scripts_lighteval/`: LightEval scripts (the SmolLM-style reference flow)
- `scripts/`: lm-eval-harness scripts mapped to the same profile structure

## Setup

Use `python>=3.11`.

lm-eval:

```sh
pip install uv
uv venv smol3_venv --python 3.11
source smol3_venv/bin/activate
uv pip install -r text/evaluation/smollm3/requirements_lm_eval.txt
```

LightEval:

```sh
pip install uv
uv venv smol3_venv --python 3.11
source smol3_venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -r text/evaluation/smollm3/requirements.txt
```

## LightEval (reference)

```sh
bash text/evaluation/smollm3/scripts_lighteval/lighteval_base.sh
bash text/evaluation/smollm3/scripts_lighteval/lighteval_mid_reasoning.sh
bash text/evaluation/smollm3/scripts_lighteval/lighteval_post_no_think.sh
bash text/evaluation/smollm3/scripts_lighteval/lighteval_post_think.sh
```

## lm-eval (matching profiles)

```sh
bash text/evaluation/smollm3/scripts/base.sh
bash text/evaluation/smollm3/scripts/mid_reasoning.sh
bash text/evaluation/smollm3/scripts/post_no_think.sh
bash text/evaluation/smollm3/scripts/post_think.sh
```

Notes:
- `base.sh` mirrors the SmolLM base split: 0-shot core tasks, 5-shot `gsm8k`, 4-shot math.
- Instruct/post profiles use 0-shot and chat template by default.
- If your local lm-eval task names differ, override `TASKS=...` in the script call.
- Defaults target SmolLM-style 32k context; set `MAX_MODEL_LEN=4096` (lm-eval) or `MAX_MODEL_LENGTH=4096` (LightEval) if your checkpoint is 4k.
- For custom lm-eval YAML prompts/tasks, set `INCLUDE_PATH=/path/to/task_yamls`.

Examples:

```sh
MODEL_PATH=/path/to/checkpoint \
bash text/evaluation/smollm3/scripts/base.sh

MODEL_PATH=/path/to/checkpoint \
TASKS="gpqa_diamond,ifeval,mixeval_hard" \
bash text/evaluation/smollm3/scripts/post_no_think.sh
```

## SLURM

lm-eval launchers:

```sh
sbatch text/evaluation/smollm3/scripts/launch_base.sbatch
sbatch text/evaluation/smollm3/scripts/launch_mid_reasoning.sbatch
sbatch text/evaluation/smollm3/scripts/launch_post_no_think.sbatch
sbatch text/evaluation/smollm3/scripts/launch_post_think.sbatch
```

LightEval launchers:

```sh
sbatch text/evaluation/smollm3/scripts_lighteval/launch_lighteval_base.sbatch
sbatch text/evaluation/smollm3/scripts_lighteval/launch_lighteval_mid_reasoning.sbatch
sbatch text/evaluation/smollm3/scripts_lighteval/launch_lighteval_post_no_think.sbatch
sbatch text/evaluation/smollm3/scripts_lighteval/launch_lighteval_post_think.sbatch
```
