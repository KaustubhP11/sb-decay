import argparse
import os
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize normalized OpenThoughts parquet with Datatrove."
    )
    parser.add_argument(
        "--input_parquet_dir",
        type=str,
        default="text/data/openthoughts/processed/raw_parquet",
        help="Directory containing normalized parquet files.",
    )
    parser.add_argument(
        "--output_tokenized_dir",
        type=str,
        default="text/data/openthoughts/processed/tokenized",
        help="Directory where Datatrove tokenized shards are written.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer to match training config.",
    )
    parser.add_argument(
        "--text_key",
        type=str,
        default="text",
        help="Text field in parquet.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=64,
        help="Datatrove tasks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Datatrove local workers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Tokenizer batch size.",
    )
    parser.add_argument(
        "--max_tokens_per_file",
        type=int,
        default=1_000_000_000,
        help="Shard cap used by DocumentTokenizer.",
    )
    parser.add_argument(
        "--clean_output",
        action="store_true",
        help="Remove existing tokenized output directory before writing new files.",
    )
    args = parser.parse_args()

    if args.clean_output and os.path.isdir(args.output_tokenized_dir):
        print(f"Cleaning output directory: {args.output_tokenized_dir}")
        shutil.rmtree(args.output_tokenized_dir)
    os.makedirs(args.output_tokenized_dir, exist_ok=True)
    parquet_files = sorted(Path(args.input_parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in input directory: {args.input_parquet_dir}"
        )

    effective_tasks = min(args.tasks, len(parquet_files))
    if effective_tasks < args.tasks:
        print(
            f"Capping tasks from {args.tasks} to {effective_tasks} "
            f"(number of parquet files)."
        )

    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer

    pipeline = [
        ParquetReader(
            data_folder=args.input_parquet_dir,
            glob_pattern="*.parquet",
            text_key=args.text_key,
        ),
        DocumentTokenizer(
            output_folder=args.output_tokenized_dir,
            tokenizer_name_or_path=args.tokenizer,
            batch_size=args.batch_size,
            max_tokens_per_file=args.max_tokens_per_file,
        ),
    ]

    print(f"Tokenizing from: {args.input_parquet_dir}")
    print(f"Writing tokenized shards to: {args.output_tokenized_dir}")
    print(f"Parquet files: {len(parquet_files)} | Tasks: {effective_tasks}")
    LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=effective_tasks,
        workers=args.workers,
    ).run()
    print("Tokenization complete.")


if __name__ == "__main__":
    main()
