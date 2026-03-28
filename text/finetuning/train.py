# Code adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# and https://huggingface.co/blog/gemma-peft
import argparse
import inspect
import json
import multiprocessing
import os
import re

import math
import time
from transformers import TrainerCallback

import torch
import transformers
import yaml
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    logging,
    set_seed,
)
import wandb
from trl import SFTConfig, SFTTrainer


class ThroughputCallback(TrainerCallback):
    def __init__(self, per_device_train_batch_size, gradient_accumulation_steps, world_size, alpha=0.1):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.world_size = world_size
        self.alpha = alpha

        self.last_time = None
        self.last_step = None
        self.ema_samples_per_second = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_time = time.time()
        self.last_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if not state.is_world_process_zero:
            return

        now = time.time()
        current_step = state.global_step

        if self.last_time is None or self.last_step is None:
            self.last_time = now
            self.last_step = current_step
            return

        steps_delta = current_step - self.last_step
        time_delta = now - self.last_time

        if steps_delta <= 0 or time_delta <= 0:
            return

        global_batch_size = (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )
        samples_delta = steps_delta * global_batch_size
        inst_samples_per_second = samples_delta / time_delta

        if self.ema_samples_per_second is None:
            self.ema_samples_per_second = inst_samples_per_second
        else:
            self.ema_samples_per_second = (
                self.alpha * inst_samples_per_second
                + (1.0 - self.alpha) * self.ema_samples_per_second
            )

        eta_seconds = None
        if state.max_steps and state.max_steps > 0 and self.ema_samples_per_second > 0:
            remaining_steps = state.max_steps - current_step
            remaining_samples = remaining_steps * global_batch_size
            eta_seconds = remaining_samples / self.ema_samples_per_second

        logs["samples_per_second"] = round(inst_samples_per_second, 4)
        logs["ema_samples_per_second"] = round(self.ema_samples_per_second, 4)
        if eta_seconds is not None:
            logs["eta_seconds"] = int(eta_seconds)
            logs["eta_hours"] = round(eta_seconds / 3600.0, 3)

        self.last_time = now
        self.last_step = current_step

        if wandb.run is not None:
            wandb.log(logs, step=current_step)


THINK_OPEN_RE = re.compile(r"<<\s*think\s*>", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"<</\s*think\s*>", flags=re.IGNORECASE)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_json_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    value = str(value).strip()
    if not value:
        return None
    return json.loads(value)


def _load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")
    return data


def _map_yaml_keys_to_args(cfg):
    key_map = {
        "model_name_or_path": "model_id",
        "max_length": "max_seq_length",
        "per_device_train_batch_size": "micro_batch_size",
        "dataset_num_proc": "num_proc",
        "data_dir": "subset",
        "dtype": "dtype",
    }
    mapped = {}
    for key, value in cfg.items():
        mapped[key_map.get(key, key)] = value
    return mapped


def _filter_kwargs_for_init(cls, kwargs):
    valid_params = set(inspect.signature(cls.__init__).parameters.keys())
    valid_params.discard("self")
    return {k: v for k, v in kwargs.items() if k in valid_params and v is not None}


def _normalize_think_tags(text: str) -> str:
    if not text:
        return ""
    text = THINK_OPEN_RE.sub("<think>", text)
    text = THINK_CLOSE_RE.sub("</think>", text)
    return text.strip()


def _stringify_chat_content(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_think_tags(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(_normalize_think_tags(text_value))
                    continue
                fallback = item.get("content", "")
                if fallback:
                    parts.append(_normalize_think_tags(str(fallback)))
            else:
                rendered = _normalize_think_tags(str(item))
                if rendered:
                    parts.append(rendered)
        return "\n".join(part for part in parts if part).strip()
    return _normalize_think_tags(str(value))


def _normalize_chat_role(value):
    role = str(value or "").strip().lower()
    role_map = {
        "human": "user",
        "user": "user",
        "prompt": "user",
        "instruction": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "model": "assistant",
        "response": "assistant",
        "system": "system",
        "developer": "system",
        "tool": "tool",
    }
    return role_map.get(role, role if role else "user")


def _extract_chat_messages(example, preferred_field, role_key, content_key):
    candidate_fields = [preferred_field, "messages", "conversation", "conversations", "chat", "dialogue"]
    seen = set()
    for field in candidate_fields:
        if not field or field in seen:
            continue
        seen.add(field)
        turns = example.get(field)
        if not isinstance(turns, list) or not turns:
            continue

        messages = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = _normalize_chat_role(
                turn.get(role_key, turn.get("from", turn.get("speaker", turn.get("author", ""))))
            )
            content = _stringify_chat_content(
                turn.get(content_key, turn.get("value", turn.get("text", turn.get("message", ""))))
            )
            if content:
                messages.append({"role": role, "content": content})

        if messages:
            return messages
    return None


def _build_chat_messages(example, messages_field, role_key, content_key):
    messages = _extract_chat_messages(example, messages_field, role_key, content_key)
    if not messages:
        return {"messages": []}
    return {"messages": messages}


def _build_chat_text(example, tokenizer, messages_field, role_key, content_key, add_generation_prompt):
    messages = _extract_chat_messages(example, messages_field, role_key, content_key)
    if not messages:
        return {"text": ""}
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        ).strip()
    }


def _extract_answer_from_conversation(conversation):
    if not isinstance(conversation, list):
        return ""
    assistant_values = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        value = turn.get("value", "")
        if role == "assistant" and isinstance(value, str):
            cleaned = _normalize_think_tags(value)
            if cleaned:
                assistant_values.append(cleaned)
    return assistant_values[-1] if assistant_values else ""


def _build_qa_pair(example, question_field, answer_field):
    question = example.get(question_field, "")
    if not isinstance(question, str):
        question = str(question) if question is not None else ""
    question = question.strip()

    answer = example.get(answer_field, "")
    if not isinstance(answer, str):
        answer = str(answer) if answer is not None else ""
    answer = _normalize_think_tags(answer)

    if not answer:
        answer = _extract_answer_from_conversation(example.get("conversation", example.get("conversations")))
    if not answer:
        raw_answer = example.get("response", "") or example.get("output", "")
        answer = _normalize_think_tags(str(raw_answer)) if raw_answer is not None else ""

    prompt = f"Question: {question}\nAnswer:"
    completion = f" {answer}".strip() if answer else ""
    return {"prompt": prompt, "completion": completion}


def _tokenize_answer_only(example, tokenizer, max_seq_length):
    prompt = example["prompt"]
    completion = example["completion"]

    prompt_tok = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    full_tok = tokenizer(
        f"{prompt} {completion}".strip(),
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    input_ids = full_tok["input_ids"]
    attention_mask = full_tok["attention_mask"]
    prompt_len = min(sum(prompt_tok["attention_mask"]), sum(attention_mask))
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    labels = [
        label if mask == 1 else -100
        for label, mask in zip(labels, attention_mask)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_file", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = {}
    if pre_args.config_file:
        yaml_cfg = _map_yaml_keys_to_args(_load_yaml_config(pre_args.config_file))

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--tokenizer_id", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-smol")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", type=str2bool, default=False)
    parser.add_argument("--offline", type=str2bool, default=False)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--dataset_download_mode", type=str, default="reuse_dataset_if_exists")
    parser.add_argument("--dataset_text_field", type=str, default="content")
    parser.add_argument("--dataset_load_from_cache_file", type=str2bool, default=True)
    parser.add_argument("--answer_only_loss", type=str2bool, default=False)
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--apply_chat_template", type=str2bool, default=False)
    parser.add_argument("--chat_messages_field", type=str, default="messages")
    parser.add_argument("--chat_role_field", type=str, default="role")
    parser.add_argument("--chat_content_field", type=str, default="content")
    parser.add_argument("--chat_add_generation_prompt", type=str2bool, default=False)
    parser.add_argument("--chat_template", type=str, default=None)
    parser.add_argument("--assistant_only_loss", type=str2bool, default=False)

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)

    parser.add_argument("--use_bnb", type=str2bool, default=False)
    parser.add_argument("--use_lora", type=str2bool, default=False)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--lr_scheduler_kwargs", type=_parse_json_dict, default=None)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_smollm2_python")
    parser.add_argument("--num_proc", type=int, default=128)
    parser.add_argument("--push_to_hub", type=str2bool, default=False)
    parser.add_argument("--repo_id", type=str, default="SmolLM2-1.7B-finetune")
    parser.add_argument("--report_to", type=str, default="wandb")
    # wandb options
    parser.add_argument("--wandb_project", type=str, default="midtraining-sft",
                        help="WandB project name (overrides WANDB_PROJECT env var)")
    parser.add_argument("--wandb_entity", type=str, default="ponkshekaustubh11",
                        help="WandB entity (user or team) to log under")

    if yaml_cfg:
        parser.set_defaults(**yaml_cfg)

    args = parser.parse_args()
    if args.dtype:
        dtype = str(args.dtype).strip().lower()
        if dtype in {"bfloat16", "bf16"}:
            args.bf16 = True
        elif dtype in {"float16", "fp16"}:
            args.bf16 = False

    if args.config_file and "num_train_epochs" in yaml_cfg and "max_steps" not in yaml_cfg:
        args.max_steps = -1

    return args


def main(args):
    # config
    lora_config = None
    if args.use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    bnb_config = None
    if args.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # load model and dataset
    token = os.environ.get("HF_TOKEN", None)
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        token = None

    model_kwargs = {
        "quantization_config": bnb_config,
        # "device_map": "auto",
        "attention_dropout": args.attention_dropout,
    }
    if args.attn_implementation and args.attn_implementation.lower() != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id)
    if args.chat_template:
        tokenizer.chat_template = args.chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.answer_only_loss and args.apply_chat_template:
        raise ValueError("answer_only_loss and apply_chat_template are mutually exclusive.")
    if args.answer_only_loss and args.assistant_only_loss:
        raise ValueError("answer_only_loss and assistant_only_loss are mutually exclusive.")
    if args.apply_chat_template and not tokenizer.chat_template:
        raise ValueError(
            "Tokenizer has no chat template. Set --tokenizer_id to a tokenizer that defines one."
        )
    if args.apply_chat_template and args.streaming:
        raise ValueError("apply_chat_template does not currently support streaming datasets.")
    if args.assistant_only_loss and not args.apply_chat_template:
        raise ValueError("assistant_only_loss requires apply_chat_template with conversational data.")

    dataset_kwargs = {
        "split": args.split,
        "token": token,
        "num_proc": args.num_proc if args.num_proc or args.streaming else multiprocessing.cpu_count(),
        "streaming": args.streaming,
        "download_mode": args.dataset_download_mode,
    }
    if args.dataset_cache_dir:
        dataset_kwargs["cache_dir"] = args.dataset_cache_dir
    if args.subset:
        dataset_kwargs["data_dir"] = args.subset

    data = load_dataset(args.dataset_name, **dataset_kwargs)

    if args.use_lora:
        model = get_peft_model(model, lora_config)

    # set wandb project/entity via environment if provided, so Trainer/SFTTrainer
    # pick it up when they call wandb.init(). This avoids duplicate run creation.
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    state = PartialState()
    process_index = state.process_index
    transform_cache_dir = os.path.join(args.output_dir, ".hf_transforms", f"rank")
    os.makedirs(transform_cache_dir, exist_ok=True)
    tokenization_cache_tag = f"answer_only_v2_len{args.max_seq_length}_rightpad"

    world_size = state.num_processes
    throughput_callback = ThroughputCallback(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=world_size,
        alpha=0.9, # ema smoothing factor for eta
    )

    if args.answer_only_loss:
        def build_dataset():
            qa_data = data.map(
                lambda ex: _build_qa_pair(ex, args.question_field, args.answer_field),
                num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
                load_from_cache_file=args.dataset_load_from_cache_file,
                cache_file_name=os.path.join(transform_cache_dir, "qa_map.arrow"),
                desc="Building Question/Answer fields",
            )
            qa_data = qa_data.filter(
                lambda ex: len(ex["completion"].strip()) > 0,
                num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
                load_from_cache_file=args.dataset_load_from_cache_file,
                cache_file_name=os.path.join(transform_cache_dir, "qa_filter.arrow"),
                desc="Dropping empty answers",
            )
            train_data = qa_data.map(
                lambda ex: _tokenize_answer_only(ex, tokenizer, args.max_seq_length),
                remove_columns=qa_data.column_names,
                num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
                load_from_cache_file=args.dataset_load_from_cache_file,
                cache_file_name=os.path.join(transform_cache_dir, f"{tokenization_cache_tag}_map.arrow"),
                desc="Tokenizing with answer-only loss mask",
            )
            train_data = train_data.filter(
                lambda ex: any(label != -100 for label in ex["labels"]),
                num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
                load_from_cache_file=args.dataset_load_from_cache_file,
                cache_file_name=os.path.join(transform_cache_dir, f"{tokenization_cache_tag}_filter.arrow"),
                desc="Dropping rows with no completion tokens",
            )

            return train_data

        if process_index == 0:
            print("Processing dataset for answer-only loss...")
            train_data = build_dataset()

        state.wait_for_everyone()
        train_data = build_dataset()

        training_kwargs = dict(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            lr_scheduler_kwargs=args.lr_scheduler_kwargs,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=args.checkpointing_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to=args.report_to,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.repo_id,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            args=TrainingArguments(**_filter_kwargs_for_init(TrainingArguments, training_kwargs)),
        )
    else:
        train_data = data
        if args.apply_chat_template:
            transform_num_proc = args.num_proc if args.num_proc else multiprocessing.cpu_count()

            def build_chat_dataset():
                if args.assistant_only_loss:
                    mapped = data.map(
                        lambda ex: _build_chat_messages(
                            ex,
                            args.chat_messages_field,
                            args.chat_role_field,
                            args.chat_content_field,
                        ),
                        remove_columns=data.column_names,
                        num_proc=transform_num_proc,
                        load_from_cache_file=args.dataset_load_from_cache_file,
                        cache_file_name=os.path.join(transform_cache_dir, "chat_messages_map.arrow"),
                        desc="Normalizing conversational messages",
                    )
                    return mapped.filter(
                        lambda ex: len(ex["messages"]) > 0,
                        num_proc=transform_num_proc,
                        load_from_cache_file=args.dataset_load_from_cache_file,
                        cache_file_name=os.path.join(transform_cache_dir, "chat_messages_filter.arrow"),
                        desc="Dropping rows with empty messages",
                    )

                mapped = data.map(
                    lambda ex: _build_chat_text(
                        ex,
                        tokenizer,
                        args.chat_messages_field,
                        args.chat_role_field,
                        args.chat_content_field,
                        args.chat_add_generation_prompt,
                    ),
                    remove_columns=data.column_names,
                    num_proc=transform_num_proc,
                    load_from_cache_file=args.dataset_load_from_cache_file,
                    cache_file_name=os.path.join(transform_cache_dir, "chat_template_map.arrow"),
                    desc="Applying chat template",
                )
                return mapped.filter(
                    lambda ex: len(ex["text"].strip()) > 0,
                    num_proc=transform_num_proc,
                    load_from_cache_file=args.dataset_load_from_cache_file,
                    cache_file_name=os.path.join(transform_cache_dir, "chat_template_filter.arrow"),
                    desc="Dropping rows with empty chat text",
                )

            if process_index == 0:
                print("Processing dataset with chat template...")
                train_data = build_chat_dataset()

            state.wait_for_everyone()
            train_data = build_chat_dataset()

        # setup the trainer
        sft_kwargs = dict(
            dataset_text_field=(None if args.assistant_only_loss else args.dataset_text_field),
            dataset_num_proc=args.num_proc,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            lr_scheduler_kwargs=args.lr_scheduler_kwargs,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            output_dir=args.output_dir,
            optim="adamw_torch_fused" if not args.use_bnb else "paged_adamw_8bit",
            save_strategy="steps",
            save_steps=args.checkpointing_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to=args.report_to,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.repo_id,
            assistant_only_loss=args.assistant_only_loss,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_data,
            args=SFTConfig(**_filter_kwargs_for_init(SFTConfig, sft_kwargs)),
        )

    # launch

    trainer.add_callback(throughput_callback)

    print("Training...")
    trainer.train()
    print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
