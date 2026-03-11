# SmolLM3 Eval Plotting

Compare `no_decay` vs `decay` eval runs and generate simple bar plots.

## What this expects

Point to two lm-eval result roots. The script recursively scans for JSON files that look like lm-eval outputs (top-level `results`).

Examples of roots:
- `/path/to/no_decay_checkpoint/evals/lm_eval`
- `/path/to/decay_checkpoint/evals/lm_eval`

## Run

```bash
python text/evaluation/smollm3/plotting/plot_eval_comparison.py \
  --baseline /path/to/no_decay_checkpoint/evals/lm_eval \
  --candidate /path/to/decay_checkpoint/evals/lm_eval \
  --baseline_label no_decay \
  --candidate_label decay \
  --out_dir text/evaluation/smollm3/plotting/out
```

## Outputs

- `task_comparison.csv`: per-task scores for both runs + delta
- `summary.csv`: aggregate summary
- `task_scores_grouped_bar.png`: grouped bars (baseline vs candidate)
- `task_deltas_bar.png`: bar plot of per-task delta (`candidate - baseline`)
