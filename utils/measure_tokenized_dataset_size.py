#!/usr/bin/env python3
"""
Summarize tokenized dataset size for Datatrove outputs.

Examples:
  python utils/measure_tokenized_dataset_size.py
  python utils/measure_tokenized_dataset_size.py text/data/tulu/processed/tokenized
  python utils/measure_tokenized_dataset_size.py text/data/tulu text/data/dolma
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class DatasetStats:
    path: Path
    tokenizer_names: set[str]
    shard_count: int
    total_tokens: int
    ds_bytes: int
    index_bytes: int
    metadata_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.ds_bytes + self.index_bytes + self.metadata_bytes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure tokenized dataset size from Datatrove .ds metadata."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("text/data")],
        help=(
            "Tokenized directory or parent directory to scan. If omitted, scan "
            "text/data for processed/tokenized outputs."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of plain text.",
    )
    return parser


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _format_tokens(num_tokens: int) -> str:
    units = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]
    for scale, suffix in units:
        if num_tokens >= scale:
            return f"{num_tokens / scale:.2f} {suffix}"
    return str(num_tokens)


def _parse_metadata_file(path: Path) -> tuple[str, int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Malformed metadata file: {path}")

    tokenizer_name = lines[0].split("|", maxsplit=1)[0]
    total_tokens = int(lines[1])
    return tokenizer_name, total_tokens


def _looks_like_tokenized_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.ds.metadata"))


def _discover_tokenized_dirs(inputs: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for raw_path in inputs:
        path = raw_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {raw_path}")

        candidates: list[Path]
        if _looks_like_tokenized_dir(path):
            candidates = [path]
        else:
            candidates = sorted(
                candidate for candidate in path.rglob("tokenized") if _looks_like_tokenized_dir(candidate)
            )

        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                discovered.append(candidate)

    if not discovered:
        input_list = ", ".join(str(path) for path in inputs)
        raise FileNotFoundError(f"No tokenized dataset directories found under: {input_list}")

    return sorted(discovered)


def _collect_stats(tokenized_dir: Path) -> DatasetStats:
    metadata_files = sorted(tokenized_dir.glob("*.ds.metadata"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata shards found in: {tokenized_dir}")

    tokenizer_names: set[str] = set()
    total_tokens = 0
    metadata_bytes = 0
    ds_bytes = 0
    index_bytes = 0

    for metadata_file in metadata_files:
        tokenizer_name, shard_tokens = _parse_metadata_file(metadata_file)
        tokenizer_names.add(tokenizer_name)
        total_tokens += shard_tokens
        metadata_bytes += metadata_file.stat().st_size

        shard_prefix = metadata_file.name.removesuffix(".metadata")
        ds_file = tokenized_dir / shard_prefix
        index_file = tokenized_dir / f"{shard_prefix}.index"

        if ds_file.exists():
            ds_bytes += ds_file.stat().st_size
        if index_file.exists():
            index_bytes += index_file.stat().st_size

    return DatasetStats(
        path=tokenized_dir,
        tokenizer_names=tokenizer_names,
        shard_count=len(metadata_files),
        total_tokens=total_tokens,
        ds_bytes=ds_bytes,
        index_bytes=index_bytes,
        metadata_bytes=metadata_bytes,
    )


def _print_stats(stats_by_dataset: list[DatasetStats]) -> None:
    total_tokens = 0
    total_bytes = 0
    total_ds_bytes = 0
    total_index_bytes = 0
    total_metadata_bytes = 0

    for stats in stats_by_dataset:
        tokenizer_label = ", ".join(sorted(stats.tokenizer_names))
        print(f"Dataset: {stats.path}")
        print(f"  tokenizer: {tokenizer_label}")
        print(f"  shards: {stats.shard_count}")
        print(f"  total_tokens: {stats.total_tokens} ({_format_tokens(stats.total_tokens)})")
        print(f"  total_disk: {_format_bytes(stats.total_bytes)}")
        print(f"  ds_bytes: {_format_bytes(stats.ds_bytes)}")
        print(f"  index_bytes: {_format_bytes(stats.index_bytes)}")
        print(f"  metadata_bytes: {_format_bytes(stats.metadata_bytes)}")
        print()

        total_tokens += stats.total_tokens
        total_bytes += stats.total_bytes
        total_ds_bytes += stats.ds_bytes
        total_index_bytes += stats.index_bytes
        total_metadata_bytes += stats.metadata_bytes

    if len(stats_by_dataset) > 1:
        print("Total:")
        print(f"  datasets: {len(stats_by_dataset)}")
        print(f"  total_tokens: {total_tokens} ({_format_tokens(total_tokens)})")
        print(f"  total_disk: {_format_bytes(total_bytes)}")
        print(f"  ds_bytes: {_format_bytes(total_ds_bytes)}")
        print(f"  index_bytes: {_format_bytes(total_index_bytes)}")
        print(f"  metadata_bytes: {_format_bytes(total_metadata_bytes)}")


def _emit_json(stats_by_dataset: list[DatasetStats]) -> None:
    payload = {
        "datasets": [
            {
                "path": str(stats.path),
                "tokenizer_names": sorted(stats.tokenizer_names),
                "shard_count": stats.shard_count,
                "total_tokens": stats.total_tokens,
                "total_bytes": stats.total_bytes,
                "ds_bytes": stats.ds_bytes,
                "index_bytes": stats.index_bytes,
                "metadata_bytes": stats.metadata_bytes,
            }
            for stats in stats_by_dataset
        ],
        "total": {
            "dataset_count": len(stats_by_dataset),
            "total_tokens": sum(stats.total_tokens for stats in stats_by_dataset),
            "total_bytes": sum(stats.total_bytes for stats in stats_by_dataset),
            "ds_bytes": sum(stats.ds_bytes for stats in stats_by_dataset),
            "index_bytes": sum(stats.index_bytes for stats in stats_by_dataset),
            "metadata_bytes": sum(stats.metadata_bytes for stats in stats_by_dataset),
        },
    }
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = _build_parser().parse_args()
    tokenized_dirs = _discover_tokenized_dirs(args.paths)
    stats_by_dataset = [_collect_stats(tokenized_dir) for tokenized_dir in tokenized_dirs]
    if args.json:
        _emit_json(stats_by_dataset)
    else:
        _print_stats(stats_by_dataset)


if __name__ == "__main__":
    main()
