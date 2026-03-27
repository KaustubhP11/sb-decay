# Tulu 3 SFT mixture preprocessing for SmolLM pretraining

This folder contains a full pipeline to prepare `allenai/tulu-3-sft-mixture` for Nanotron pretraining:

1. Download and normalize the dataset to a single `text` field.
2. Save normalized rows as parquet.
3. Tokenize parquet with Datatrove using `meta-llama/Llama-3.2-1B`.
4. Use the tokenized output path in your Nanotron dataset config.

## 0) Install deps

From repo root:

```bash
python3 -m pip install -r text/data/tulu/requirements.txt
```

If you install manually, use:

```bash
python3 -m pip install -U pip
python3 -m pip install datasets pyarrow transformers datatrove
```

## 1) Download + normalize

```bash
python3 text/data/tulu/prepare_tulu.py \
  --dataset_id allenai/tulu-3-sft-mixture \
  --split train \
  --output_dir text/data/tulu/processed \
  --num_proc 16 \
  --preview 3 \
  --parquet_shards 128
```

Notes:
- The script is schema-tolerant and prefers direct text fields when present.
- For chat-style rows, it serializes turns as `System:`, `User:`, `Assistant:`, etc.
- It also normalizes non-standard think tags like `<<think>...<</think>`.
- For a quick smoke test, add `--max_samples 50000`.
- If the dataset schema changes, you can force a field with `--text_field`.
- If your cluster has a dedicated HF cache, pass `--hf_cache_dir /path/to/hf_cache`.

## 2) Tokenize for pretraining

```bash
python3 text/data/tulu/tokenize_tulu.py \
  --input_parquet_dir text/data/tulu/processed/raw_parquet \
  --output_tokenized_dir text/data/tulu/processed/tokenized \
  --tokenizer meta-llama/Llama-3.2-1B \
  --tasks 128 \
  --workers 64 \
  --clean_output
```

Tune `--tasks` and `--workers` for your machine or cluster. Set `--tasks` close to
the number of parquet shards for better parallelism.

## 3) Plug into Nanotron

Add Tulu to your dataset block:

- `dataset_read_path:`
  - `hf://datasets/allenai/tulu-3-sft-mixture`
- `dataset_folder:`
  - `text/data/tulu/processed/tokenized`
- `dataset_weights:`
  - `<your_weight>`

Example mixed Stage 3 dataset block:

```yaml
dataset:
  dataset_read_path:
    - hf://datasets/open-thoughts/OpenThoughts2-1M
    - hf://datasets/allenai/dolma3_dolmino_mix-10B-1025
    - hf://datasets/allenai/tulu-3-sft-mixture
  dataset_folder:
    - /path/to/text/data/openthoughts/processed/tokenized
    - /path/to/text/data/dolma/processed/tokenized
    - /path/to/text/data/tulu/processed/tokenized
  dataset_weights:
    - 0.25
    - 0.65
    - 0.10
  pad_samples_to_global_batch_size: false
  return_positions: true
  token_size_in_bytes: 4
  tokenizer_name: meta-llama/Llama-3.2-1B
  use_old_brrr_dataloader: false
  vocab_size: 128256
```

## 4) Required invariants

- `dataset_read_path`, `dataset_folder`, `dataset_weights` must have equal lengths.
- Weights should sum to about `1.0`.
- Training tokenizer config must match tokenization tokenizer:
  - `meta-llama/Llama-3.2-1B`
  - `vocab_size: 128256`
