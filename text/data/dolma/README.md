# Dolma preprocessing for SmolLM pretraining

This folder contains a full pipeline to prepare `allenai/dolma3_dolmino_mix-10B-1025` for Nanotron pretraining:

1. Download the dataset and keep its `text` field as-is.
2. Save normalized rows as parquet.
3. Tokenize parquet with Datatrove using `meta-llama/Llama-3.2-1B`.
4. Use the tokenized output path in a Stage-3 training config.

## 0) Install deps

From repo root:

```bash
python3 -m pip install -r text/data/dolma/requirements.txt
```

If you install manually, use:

```bash
python3 -m pip install -U pip
python3 -m pip install datasets pyarrow transformers datatrove
```

Do **not** install `datatrove[all]` unless you need all language-specific pipelines.
It pulls optional dependencies like `sudachipy` that require a Rust compiler on some systems.

If you already ran `datatrove[all]` and hit the Rust error, reset with:

```bash
python3 -m pip uninstall -y sudachipy sudachidict_core datatrove
python3 -m pip install datatrove
```

## 1) Download + normalize

```bash
python3 text/data/dolma/prepare_dolma.py \
  --dataset_id allenai/dolma3_dolmino_mix-10B-1025 \
  --split train \
  --output_dir text/data/dolma/processed \
  --num_proc 16 \
  --preview 3 \
  --parquet_shards 256
```

Notes:
- The script expects a `text` column and keeps it as-is (no prompt/answer reconstruction).
- For a quick smoke test, add `--max_samples 50000`.
- Use `--preview N` to print N transformed samples before writing parquet.
- Use `--parquet_shards N` to split output into multiple parquet files (recommended).
- If your cluster has a dedicated HF cache, pass `--hf_cache_dir /path/to/hf_cache`.

## 2) Tokenize for pretraining

```bash
python3 text/data/dolma/tokenize_dolma.py \
  --input_parquet_dir text/data/dolma/processed/raw_parquet \
  --output_tokenized_dir text/data/dolma/processed/tokenized \
  --tokenizer meta-llama/Llama-3.2-1B \
  --tasks 256 \
  --workers 64 \
  --clean_output
```

Tune `--tasks` and `--workers` for your machine/cluster. Set `--tasks` close to
the number of parquet shards for better parallelism.

## 3) Plug into Stage 3 config

Use either:
- `text/pretraining/smollm3/run_decay.yaml`
- `text/pretraining/smollm3/run_stable.yaml`

Both now point to a mixed Stage-3 setup with:
- `text/data/openthoughts/processed/tokenized`
- `text/data/dolma/processed/tokenized`

## 4) Required invariants

- `dataset_read_path`, `dataset_folder`, `dataset_weights` must have equal lengths.
- Weights should sum to ~`1.0`.
- Training tokenizer config must match tokenization tokenizer:
  - `meta-llama/Llama-3.2-1B`
  - `vocab_size: 128256`
