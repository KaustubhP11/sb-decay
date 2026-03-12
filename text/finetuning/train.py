# Code adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# and https://huggingface.co/blog/gemma-peft
import argparse
import inspect
import json
import multiprocessing
import os
import re

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
from trl import SFTConfig, SFTTrainer

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
    )
    full_tok = tokenizer(
        f"{prompt} {completion}".strip(),
        add_special_tokens=True,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = full_tok["input_ids"]
    attention_mask = full_tok["attention_mask"]
    prompt_len = min(len(prompt_tok["input_ids"]), len(input_ids))
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len

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
        "device_map": {"": PartialState().process_index},
        "attention_dropout": args.attention_dropout,
    }
    if args.attn_implementation and args.attn_implementation.lower() != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    process_index = PartialState().process_index
    transform_cache_dir = os.path.join(args.output_dir, ".hf_transforms", f"rank{process_index}")
    os.makedirs(transform_cache_dir, exist_ok=True)

    if args.answer_only_loss:
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
            cache_file_name=os.path.join(transform_cache_dir, "tokenized_map.arrow"),
            desc="Tokenizing with answer-only loss mask",
        )
        train_data = train_data.filter(
            lambda ex: any(label != -100 for label in ex["labels"]),
            num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
            load_from_cache_file=args.dataset_load_from_cache_file,
            cache_file_name=os.path.join(transform_cache_dir, "tokenized_filter.arrow"),
            desc="Dropping rows with no completion tokens",
        )

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
        # setup the trainer
        sft_kwargs = dict(
            dataset_text_field=args.dataset_text_field,
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
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=args.checkpointing_steps,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to=args.report_to,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.repo_id,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=data,
            args=SFTConfig(**_filter_kwargs_for_init(SFTConfig, sft_kwargs)),
        )

    # launch
    print("Training...")
    trainer.train()
    print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
