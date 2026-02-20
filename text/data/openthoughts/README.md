# OpenThoughts2-1M preprocessing for SmolLM pretraining

This folder contains a full pipeline to prepare `open-thoughts/OpenThoughts2-1M` for Nanotron pretraining:

1. Download and normalize the dataset to a single `text` field.
2. Save normalized rows as parquet.
3. Tokenize parquet with Datatrove using `meta-llama/Llama-3.2-1B`.
4. Use the tokenized output path in `run_decay.yaml` Stage 3.

## 0) Install deps

From repo root:

```bash
python3 -m pip install -r text/data/openthoughts/requirements.txt
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
python3 text/data/openthoughts/prepare_openthoughts.py \
  --dataset_id open-thoughts/OpenThoughts2-1M \
  --split train \
  --output_dir text/data/openthoughts/processed \
  --num_proc 16 \
  --preview 3 \
  --parquet_shards 128
```

Notes:
- For OpenThoughts2-1M, the script now builds each sample as:
  - `Question: <question>`
  - `Answer: <final assistant answer>`
- It keeps reasoning tokens and normalizes think tags to standard
  `<think>...</think>` (also handles non-standard `<<think>...<</think>` input).
- The script still has generic fallbacks for other dataset schemas.
- For a quick smoke test, add `--max_samples 50000`.
- Use `--preview N` to print N transformed samples before writing parquet.
- Use `--parquet_shards N` to split output into multiple parquet files (recommended).
- If your cluster has a dedicated HF cache, pass `--hf_cache_dir /path/to/hf_cache`.

## 2) Tokenize for pretraining

```bash
python3 text/data/openthoughts/tokenize_openthoughts.py \
  --input_parquet_dir text/data/openthoughts/processed/raw_parquet \
  --output_tokenized_dir text/data/openthoughts/processed/tokenized \
  --tokenizer meta-llama/Llama-3.2-1B \
  --tasks 128 \
  --workers 64 \
  --clean_output
```

Tune `--tasks` and `--workers` for your machine/cluster. Set `--tasks` close to
the number of parquet shards for better parallelism.

## 3) Plug into `run_decay.yaml` (Stage 3)

In the Stage-3 `dataset` block of `text/pretraining/smollm3/run_decay.yaml`:

- Add the original read path for traceability:
  - `dataset_read_path:`
    - `hf://datasets/open-thoughts/OpenThoughts2-1M`
- Add tokenized folder:
  - `dataset_folder:`
    - `text/data/openthoughts/processed/tokenized`
- Add matching weight:
  - `dataset_weights:`
    - `<your_weight>`

If Stage 3 is only OpenThoughts, use weight `1.0`.

## 4) Required invariants

- `dataset_read_path`, `dataset_folder`, `dataset_weights` must have equal lengths.
- Weights should sum to ~`1.0`.
- Training tokenizer config must match tokenization tokenizer:
  - `meta-llama/Llama-3.2-1B`
  - `vocab_size: 128256`
