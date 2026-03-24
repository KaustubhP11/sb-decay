#!/usr/bin/env python3
"""Compare lm-eval results between two runs (e.g., no_decay vs decay).

Usage:
  python text/evaluation/smollm3/plotting/plot_eval_comparison.py \
    --baseline /path/to/no_decay/evals/lm_eval \
    --candidate /path/to/decay/evals/lm_eval
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError as exc:
    missing = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
    raise SystemExit(
        f"Missing dependency: {missing}. Install plotting deps with:\n"
        "  uv pip install -r text/evaluation/smollm3/plotting/requirements.txt"
    ) from exc

PRIMARY_METRIC_CANDIDATES = [
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "pass@1,none",
    "f1,none",
    "score,none",
]

TASK_METRIC_OVERRIDES = {
    "gsm8k": [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
    ],
}


@dataclass
class RunInfo:
    label: str
    root: Path


def find_lm_eval_jsons(root: Path) -> list[Path]:
    files = []
    if root.is_file() and root.suffix == ".json":
        return [root]

    for p in root.rglob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("results"), dict):
                files.append(p)
        except Exception:
            continue
    return sorted(files)


def choose_metric(task: str, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    for metric_name in TASK_METRIC_OVERRIDES.get(task, []):
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)):
            return metric_name, float(value)

    for m in PRIMARY_METRIC_CANDIDATES:
        v = metrics.get(m)
        if isinstance(v, (int, float)):
            return m, float(v)

    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not k.endswith("_stderr,none"):
            return k, float(v)

    return None, None


def parse_result_file(path: Path, run_label: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    config = data.get("config", {}) if isinstance(data, dict) else {}
    num_fewshot = config.get("num_fewshot")

    rows: list[dict[str, Any]] = []
    for task, metrics in (data.get("results", {}) or {}).items():
        if not isinstance(metrics, dict):
            continue
        metric_name, metric_value = choose_metric(task, metrics)
        if metric_name is None or metric_value is None:
            continue
        rows.append(
            {
                "run": run_label,
                "task": task,
                "metric": metric_name,
                "value": metric_value,
                "num_fewshot": num_fewshot,
                "source_file": str(path),
            }
        )
    return rows


def load_run(run: RunInfo) -> pd.DataFrame:
    files = find_lm_eval_jsons(run.root)
    if not files:
        raise FileNotFoundError(f"No lm-eval JSON files with top-level 'results' found under: {run.root}")

    rows: list[dict[str, Any]] = []
    for p in files:
        rows.extend(parse_result_file(p, run.label))

    if not rows:
        raise RuntimeError(f"Found JSON files under {run.root}, but no usable metric rows were parsed.")

    df = pd.DataFrame(rows)

    # Keep latest row per (task, metric, fewshot) if duplicates exist.
    # Sort by source filename so deterministic.
    df = df.sort_values(["task", "metric", "num_fewshot", "source_file"]).drop_duplicates(
        subset=["run", "task", "metric", "num_fewshot"], keep="last"
    )
    return df.reset_index(drop=True)


def plot_grouped_bars(
    merged: pd.DataFrame,
    out_dir: Path | None,
    baseline_label: str,
    candidate_label: str,
    save: bool = False,
    show: bool = True,
) -> None:
    merged = merged.sort_values("task").reset_index(drop=True)
    x = range(len(merged))

    plt.figure(figsize=(max(10, len(merged) * 0.4), 6))
    plt.bar([i - 0.2 for i in x], merged[baseline_label], width=0.4, label=baseline_label)
    plt.bar([i + 0.2 for i in x], merged[candidate_label], width=0.4, label=candidate_label)
    plt.xticks(list(x), merged["task"], rotation=75, ha="right")
    plt.ylabel("Score")
    plt.title("Task Scores: baseline vs candidate")
    plt.legend()
    plt.tight_layout()
    if save and out_dir is not None:
        plt.savefig(out_dir / "task_scores_grouped_bar.png", dpi=180)
    if show:
        plt.show()
    plt.close()


def plot_deltas(
    merged: pd.DataFrame,
    out_dir: Path | None,
    save: bool = False,
    show: bool = True,
) -> None:
    delta_df = merged.copy()
    delta_df = delta_df.sort_values("delta", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(max(10, len(delta_df) * 0.4), 6))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta_df["delta"]]
    plt.bar(delta_df["task"], delta_df["delta"], color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Delta (candidate - baseline)")
    plt.title("Per-task Delta")
    plt.tight_layout()
    if save and out_dir is not None:
        plt.savefig(out_dir / "task_deltas_bar.png", dpi=180)
    if show:
        plt.show()
    plt.close()


def make_summary_table(merged: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "num_tasks": [len(merged)],
            "baseline_mean": [merged.iloc[:, 1].mean()],
            "candidate_mean": [merged.iloc[:, 2].mean()],
            "delta_mean": [merged["delta"].mean()],
            "delta_median": [merged["delta"].median()],
            "num_improved": [(merged["delta"] > 0).sum()],
            "num_regressed": [(merged["delta"] < 0).sum()],
            "num_tied": [(merged["delta"] == 0).sum()],
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two lm-eval runs and plot decay vs no-decay.")
    parser.add_argument("--baseline", required=True, type=Path, help="Path to baseline run root (e.g., no_decay eval dir).")
    parser.add_argument("--candidate", required=True, type=Path, help="Path to candidate run root (e.g., decay eval dir).")
    parser.add_argument("--baseline_label", default="no_decay", help="Label for baseline bars.")
    parser.add_argument("--candidate_label", default="decay", help="Label for candidate bars.")
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Optional directory to save CSV/PNG artifacts. If unset, plots are only displayed.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = load_run(RunInfo(label=args.baseline_label, root=args.baseline))
    candidate_df = load_run(RunInfo(label=args.candidate_label, root=args.candidate))

    base_sel = baseline_df[["task", "value"]].rename(columns={"value": args.baseline_label})
    cand_sel = candidate_df[["task", "value"]].rename(columns={"value": args.candidate_label})

    merged = base_sel.merge(cand_sel, on="task", how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping tasks between baseline and candidate runs.")

    merged["delta"] = merged[args.candidate_label] - merged[args.baseline_label]

    summary = make_summary_table(merged[["task", args.baseline_label, args.candidate_label, "delta"]])

    print(summary.to_string(index=False))

    save = save_dir is not None
    plot_grouped_bars(
        merged[["task", args.baseline_label, args.candidate_label]],
        save_dir,
        args.baseline_label,
        args.candidate_label,
        save=save,
        show=True,
    )
    plot_deltas(merged[["task", "delta"]], save_dir, save=save, show=True)

    if save and save_dir is not None:
        merged.to_csv(save_dir / "task_comparison.csv", index=False)
        summary.to_csv(save_dir / "summary.csv", index=False)
        print(f"Saved artifacts under: {save_dir}")


if __name__ == "__main__":
    main()
