import argparse
import os
import re
from typing import Any

from datasets import Dataset, load_dataset


THINK_OPEN_RE = re.compile(r"<<\s*think\s*>", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"<</\s*think\s*>", flags=re.IGNORECASE)


def _normalize_think_tags(text: str) -> str:
    if not text:
        return ""
    text = THINK_OPEN_RE.sub("<think>", text)
    text = THINK_CLOSE_RE.sub("</think>", text)
    return text.strip()


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_think_tags(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(_normalize_think_tags(text_value))
                    continue
                if item_type in {"text", "input_text", "output_text"}:
                    fallback = item.get("content", "")
                    if fallback:
                        parts.append(_normalize_think_tags(str(fallback)))
            else:
                rendered = _normalize_think_tags(str(item))
                if rendered:
                    parts.append(rendered)
        return "\n".join(part for part in parts if part).strip()
    return _normalize_think_tags(str(value))


def _normalize_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    role_map = {
        "human": "User",
        "user": "User",
        "prompt": "User",
        "instruction": "User",
        "assistant": "Assistant",
        "gpt": "Assistant",
        "model": "Assistant",
        "response": "Assistant",
        "system": "System",
        "developer": "System",
        "tool": "Tool",
    }
    return role_map.get(role, role.capitalize() if role else "Unknown")


def _format_turn(role: str, content: str) -> str:
    normalized_lines = [line.rstrip() for line in content.splitlines()]
    if not normalized_lines:
        return ""
    return "\n".join(f"{role}: {line}" for line in normalized_lines)


def _conversation_to_text(turns: Any) -> str:
    if not isinstance(turns, list):
        return ""

    lines: list[str] = []
    for turn in turns:
        if isinstance(turn, dict):
            role = _normalize_role(
                turn.get("role", turn.get("from", turn.get("speaker", turn.get("author", ""))))
            )
            content = _stringify_content(
                turn.get("content", turn.get("value", turn.get("text", turn.get("message", ""))))
            )
            if content:
                lines.append(_format_turn(role, content))
        else:
            content = _stringify_content(turn)
            if content:
                lines.append(content)
    return "\n".join(lines).strip()


def _to_text(example: dict[str, Any], text_field: str | None) -> dict[str, str]:
    if text_field:
        return {"text": _stringify_content(example.get(text_field, ""))}

    prompt = _stringify_content(example.get("prompt"))
    completion = _stringify_content(example.get("completion"))
    response = _stringify_content(example.get("response"))
    output = _stringify_content(example.get("output"))
    answer = _stringify_content(example.get("answer"))
    chosen = _stringify_content(example.get("chosen"))

    if prompt and (completion or response or output or answer or chosen):
        assistant_text = completion or response or output or answer or chosen
        return {"text": _format_turn("User", prompt) + "\n" + _format_turn("Assistant", assistant_text)}

    question = _stringify_content(example.get("question"))
    if question and (answer or response or output):
        assistant_text = answer or response or output
        return {"text": _format_turn("User", question) + "\n" + _format_turn("Assistant", assistant_text)}

    direct_candidates = [
        "text",
        "instruction",
    ]
    for key in direct_candidates:
        if key not in example:
            continue
        rendered = _stringify_content(example.get(key))
        if rendered:
            return {"text": rendered}

    for key in ["messages", "conversation", "conversations", "chat", "dialogue"]:
        if key not in example:
            continue
        rendered = _conversation_to_text(example.get(key))
        if rendered:
            return {"text": rendered}

    return {"text": _normalize_think_tags(str(example))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and normalize allenai/tulu-3-sft-mixture into parquet."
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="allenai/tulu-3-sft-mixture",
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
        default="text/data/tulu/processed",
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
        "--streaming",
        action="store_true",
        help="Load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print N processed samples before writing parquet.",
    )
    parser.add_argument(
        "--preview_chars",
        type=int,
        default=800,
        help="Maximum characters shown per preview sample.",
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
    for file_name in os.listdir(raw_dir):
        if file_name.endswith(".parquet"):
            os.remove(os.path.join(raw_dir, file_name))

    print(f"Loading dataset: {args.dataset_id} [{args.split}]")
    if args.streaming and args.max_samples is None:
        raise ValueError("--streaming requires --max_samples so the dataset can be materialized.")

    ds = load_dataset(
        args.dataset_id,
        split=args.split,
        cache_dir=args.hf_cache_dir,
        streaming=args.streaming,
    )

    if args.max_samples is not None:
        if args.streaming:
            ds = ds.take(args.max_samples)
            ds = list(ds)
        else:
            max_samples = min(args.max_samples, len(ds))
            ds = ds.select(range(max_samples))
            print(f"Using max_samples={max_samples}")

    if isinstance(ds, list):
        original_columns = list(ds[0].keys()) if ds else []
        mapped_rows = [_to_text(example, args.text_field) for example in ds]
        ds = Dataset.from_list([row for row in mapped_rows if row["text"].strip()])
    else:
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

    preview_chars = max(50, args.preview_chars)
    if args.preview and args.preview > 0:
        preview_count = min(args.preview, len(ds))
        print(f"Previewing {preview_count} processed samples:")
        for i in range(preview_count):
            text = ds[i]["text"]
            print(f"\n--- sample {i} ---")
            if len(text) > preview_chars:
                print(f"{text[:preview_chars]} ... [truncated]")
                print(f"(sample length: {len(text)} chars)")
            else:
                print(text)

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
    print(f"Rows written: {num_rows}")
    print(f"Parquet shards written: {num_shards}")
    print(f"Parquet folder: {raw_dir}")


if __name__ == "__main__":
    main()
