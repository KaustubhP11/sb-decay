import argparse
import os
import re
from typing import Any

from datasets import load_dataset


# OpenThoughts samples may use either <think>...</think> or <<think>...<</think>.
THINK_RE = re.compile(
    r"(?P<open><<think>|<think>).*?(?P<close><</think>|</think>)",
    flags=re.DOTALL | re.IGNORECASE,
)


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought content while preserving think tags."""
    if not text:
        return ""
    text = THINK_RE.sub(lambda m: f"{m.group('open')}{m.group('close')}", text)
    return text.strip()


def _extract_answer_from_conversation(conversation: Any) -> str:
    """Get the final assistant answer from OpenThoughts conversation format."""
    if not isinstance(conversation, list):
        return ""

    assistant_values: list[str] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        value = turn.get("value", "")
        if role == "assistant" and isinstance(value, str):
            cleaned = _strip_reasoning(value)
            if cleaned:
                assistant_values.append(cleaned)

    return assistant_values[-1] if assistant_values else ""


def _to_text(example: dict[str, Any], text_field: str | None) -> dict[str, str]:
    if text_field:
        value = example.get(text_field, "")
        return {"text": str(value) if value is not None else ""}

    # OpenThoughts-specific path: keep only Question + final Answer.
    question = example.get("question", "")
    if not isinstance(question, str):
        question = str(question) if question is not None else ""
    question = question.strip()

    answer = _extract_answer_from_conversation(
        example.get("conversation", example.get("conversations"))
    )
    if not answer:
        # Fallbacks if a sample is missing `conversation`.
        raw_answer = example.get("answer", "") or example.get("response", "") or example.get("output", "")
        answer = _strip_reasoning(str(raw_answer)) if raw_answer is not None else ""

    if question and answer:
        return {"text": f"Question: {question}\nAnswer: {answer}"}

    # Generic fallback for other datasets.
    # Prefer direct text fields when available.
    direct_candidates = [
        "text",
        "prompt",
        "completion",
        "response",
        "output",
        "answer",
        "instruction",
        "question",
    ]
    for key in direct_candidates:
        value = example.get(key, None)
        if isinstance(value, str) and value.strip():
            return {"text": value}

    # Handle chat-style lists of messages/conversations.
    for key in ["messages", "conversation", "conversations"]:
        value = example.get(key, None)
        if isinstance(value, list) and value:
            lines: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    role = item.get("role", "unknown")
                    content = item.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(str(x) for x in content)
                    lines.append(f"{role}: {content}".strip())
                else:
                    lines.append(str(item))
            text = "\n".join(lines).strip()
            if text:
                return {"text": text}

    # Last-resort fallback that keeps script robust.
    return {"text": str(example)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize open-thoughts/OpenThoughts2-1M into parquet."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="open-thoughts/OpenThoughts2-1M",
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
        default="text/data/openthoughts/processed",
        help="Output root directory.",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default=None,
        help="Optional source field to force as text.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Workers for map/filter.",
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = os.path.join(args.output_dir, "raw_parquet")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_id} [{args.split}]")
    ds = load_dataset(
        args.dataset_id,
        split=args.split,
        cache_dir=args.hf_cache_dir,
    )

    if args.max_samples is not None:
        max_samples = min(args.max_samples, len(ds))
        ds = ds.select(range(max_samples))
        print(f"Using max_samples={max_samples}")

    original_columns = ds.column_names
    print(f"Original columns: {original_columns}")

    ds = ds.map(
        lambda ex: _to_text(ex, args.text_field),
        remove_columns=original_columns,
        num_proc=args.num_proc,
        desc="Normalizing examples to text",
    )

    ds = ds.filter(
        lambda ex: len(ex["text"].strip()) > 0,
        num_proc=args.num_proc,
        desc="Dropping empty rows",
    )

    if args.preview and args.preview > 0:
        preview_count = min(args.preview, len(ds))
        print(f"Previewing {preview_count} processed samples:")
        for i in range(preview_count):
            print(f"\n--- sample {i} ---")
            print(ds[i]["text"])

    num_rows = len(ds)
    if num_rows == 0:
        raise RuntimeError("No rows left after preprocessing/filtering.")

    num_shards = max(1, min(args.parquet_shards, num_rows))
    dataset_name = args.dataset_id.split("/")[-1]
    print(f"Writing {num_shards} parquet shard(s) to: {raw_dir}")
    for shard_idx in range(num_shards):
        shard_ds = ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        if len(shard_ds) == 0:
            continue
        output_file = os.path.join(
            raw_dir,
            f"{dataset_name}-{args.split}-{shard_idx:05d}-of-{num_shards:05d}.parquet",
        )
        shard_ds.to_parquet(output_file)

    print("Done.")
    print(f"Rows written: {len(ds)}")
    print(f"Parquet shards written: {num_shards}")
    print(f"Parquet folder: {raw_dir}")


if __name__ == "__main__":
    main()
