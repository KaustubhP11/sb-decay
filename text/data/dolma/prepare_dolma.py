import argparse
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


TEXT_SCHEMA = pa.schema([("text", pa.string())])


def _flush_buffer(
    shard_idx: int,
    shard_buffer: list[str],
    writers: dict[int, pq.ParquetWriter],
    raw_dir: str,
    dataset_name: str,
    split: str,
    num_shards: int,
) -> None:
    if not shard_buffer:
        return

    if shard_idx not in writers:
        output_file = os.path.join(
            raw_dir,
            f"{dataset_name}-{split}-{shard_idx:05d}-of-{num_shards:05d}.parquet",
        )
        writers[shard_idx] = pq.ParquetWriter(output_file, TEXT_SCHEMA)

    table = pa.Table.from_arrays([pa.array(shard_buffer, type=pa.string())], schema=TEXT_SCHEMA)
    writers[shard_idx].write_table(table)
    shard_buffer.clear()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Dolma and export its text field to parquet shards."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="allenai/dolma3_dolmino_mix-10B-1025",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text/data/dolma/processed",
        help="Output root directory.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Unused in streaming mode; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional HF datasets cache dir.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not download from HF; use local cache only.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print N processed samples before writing parquet.",
    )
    parser.add_argument(
        "--parquet_shards",
        type=int,
        default=128,
        help="Number of parquet files to write in raw_parquet.",
    )
    parser.add_argument(
        "--preview_chars",
        type=int,
        default=500,
        help="Maximum characters shown per preview sample.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = os.path.join(args.output_dir, "raw_parquet")
    os.makedirs(raw_dir, exist_ok=True)
    for file_name in os.listdir(raw_dir):
        if file_name.endswith(".parquet"):
            os.remove(os.path.join(raw_dir, file_name))

    print(f"Resolving dataset snapshot: {args.dataset_id}")
    snapshot_dir = snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        cache_dir=args.hf_cache_dir,
        allow_patterns=["data/**/*.jsonl.zst"],
        local_files_only=args.local_files_only,
    )
    data_files = sorted(Path(snapshot_dir).glob("data/**/*.jsonl.zst"))
    if not data_files:
        raise FileNotFoundError(
            f"No data files found under snapshot: {snapshot_dir}/data/**/*.jsonl.zst"
        )
    print(f"Resolved {len(data_files)} compressed shards")

    num_shards = max(1, args.parquet_shards)
    dataset_name = args.dataset_id.split("/")[-1]
    print(f"Writing {num_shards} parquet shard(s) to: {raw_dir}")

    writers: dict[int, pq.ParquetWriter] = {}
    shard_buffers: dict[int, list[str]] = {idx: [] for idx in range(num_shards)}
    buffer_flush_size = 2048

    rows_seen = 0
    rows_written = 0
    preview_left = max(0, args.preview)
    text_column_missing = False
    skipped_files = 0
    preview_chars = max(50, args.preview_chars)
    row_pbar = tqdm(desc="Preparing Dolma rows", unit="rows")
    file_pbar = tqdm(total=len(data_files), desc="Reading Dolma files", unit="file")

    try:
        for data_file in data_files:
            file_pbar.update(1)
            try:
                ds = load_dataset(
                    "json",
                    data_files=str(data_file),
                    split="train",
                    streaming=True,
                    cache_dir=args.hf_cache_dir,
                )
                for example in ds:
                    rows_seen += 1
                    row_pbar.update(1)
                    if "text" not in example:
                        text_column_missing = True
                        continue

                    value = example["text"]
                    text = value if isinstance(value, str) else str(value)
                    text = text.strip()
                    if not text:
                        continue

                    if preview_left > 0:
                        print(f"\n--- sample {args.preview - preview_left} ---")
                        if len(text) > preview_chars:
                            print(f"{text[:preview_chars]} ... [truncated]")
                            print(f"(sample length: {len(text)} chars)")
                        else:
                            print(text)
                        preview_left -= 1

                    shard_idx = rows_written % num_shards
                    shard_buffers[shard_idx].append(text)
                    rows_written += 1

                    if len(shard_buffers[shard_idx]) >= buffer_flush_size:
                        _flush_buffer(
                            shard_idx=shard_idx,
                            shard_buffer=shard_buffers[shard_idx],
                            writers=writers,
                            raw_dir=raw_dir,
                            dataset_name=dataset_name,
                            split=args.split,
                            num_shards=num_shards,
                        )

                    if args.max_samples is not None and rows_written >= args.max_samples:
                        print(f"Reached max_samples={args.max_samples}")
                        break

                if args.max_samples is not None and rows_written >= args.max_samples:
                    break
            except Exception as exc:
                skipped_files += 1
                print(f"Skipping corrupt shard: {data_file.name} ({exc})")
                continue

        if text_column_missing:
            print("Warning: Some rows were missing `text` and were skipped.")

        for shard_idx in range(num_shards):
            _flush_buffer(
                shard_idx=shard_idx,
                shard_buffer=shard_buffers[shard_idx],
                writers=writers,
                raw_dir=raw_dir,
                dataset_name=dataset_name,
                split=args.split,
                num_shards=num_shards,
            )
    finally:
        row_pbar.close()
        file_pbar.close()
        for writer in writers.values():
            writer.close()

    if rows_written == 0:
        raise RuntimeError("No rows left after filtering empty text values.")

    print("Done.")
    print(f"Rows read: {rows_seen}")
    print(f"Rows written: {rows_written}")
    print(f"Skipped source files: {skipped_files}")
    print(f"Parquet shards written: {num_shards}")
    print(f"Parquet folder: {raw_dir}")


if __name__ == "__main__":
    main()
