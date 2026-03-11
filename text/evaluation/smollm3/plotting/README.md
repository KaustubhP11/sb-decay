# SmolLM3 Eval Plotting

Compare `no_decay` vs `decay` eval runs and visualize directly.

## What this expects

Point to two lm-eval result roots. The script recursively scans for JSON files that look like lm-eval outputs (top-level `results`).

Examples of roots:
- `/path/to/no_decay_checkpoint/evals/lm_eval`
- `/path/to/decay_checkpoint/evals/lm_eval`

## Preferred: Notebook

Use:

`text/evaluation/smollm3/plotting/eval_comparison.ipynb`

## Optional CLI

This displays plots without writing files:

```bash
python text/evaluation/smollm3/plotting/plot_eval_comparison.py \
  --baseline /path/to/no_decay_checkpoint/evals/lm_eval \
  --candidate /path/to/decay_checkpoint/evals/lm_eval \
  --baseline_label no_decay \
  --candidate_label decay
```

If you explicitly want artifacts, pass `--save_dir /path/to/output`.
