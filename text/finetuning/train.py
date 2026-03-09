# Code adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# and https://huggingface.co/blog/gemma-peft
import argparse
import multiprocessing
import os
import re

import torch
import transformers
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--tokenizer_id", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="bigcode/the-stack-smol")
    parser.add_argument("--subset", type=str, default="data/python")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--streaming", type=str2bool, default=False)
    parser.add_argument("--dataset_text_field", type=str, default="content")
    parser.add_argument("--answer_only_loss", type=str2bool, default=False)
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--answer_field", type=str, default="answer")

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=str2bool, default=True)

    parser.add_argument("--use_bnb", type=str2bool, default=False)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_smollm2_python")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=str2bool, default=False)
    parser.add_argument("--repo_id", type=str, default="SmolLM2-1.7B-finetune")
    return parser.parse_args()


def main(args):
    # config
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id or args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        token=token,
        num_proc=args.num_proc if args.num_proc or args.streaming else multiprocessing.cpu_count(),
        streaming=args.streaming,
    )

    model = get_peft_model(model, lora_config)

    if args.answer_only_loss:
        qa_data = data.map(
            lambda ex: _build_qa_pair(ex, args.question_field, args.answer_field),
            num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
            desc="Building Question/Answer fields",
        )
        qa_data = qa_data.filter(
            lambda ex: len(ex["completion"].strip()) > 0,
            num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
            desc="Dropping empty answers",
        )
        train_data = qa_data.map(
            lambda ex: _tokenize_answer_only(ex, tokenizer, args.max_seq_length),
            remove_columns=qa_data.column_names,
            num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
            desc="Tokenizing with answer-only loss mask",
        )
        train_data = train_data.filter(
            lambda ex: any(label != -100 for label in ex["labels"]),
            num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
            desc="Dropping rows with no completion tokens",
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            args=TrainingArguments(
                per_device_train_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=args.warmup_steps,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                lr_scheduler_type=args.lr_scheduler_type,
                weight_decay=args.weight_decay,
                bf16=args.bf16,
                logging_strategy="steps",
                logging_steps=10,
                output_dir=args.output_dir,
                optim="paged_adamw_8bit",
                seed=args.seed,
                run_name=f"train-{args.model_id.split('/')[-1]}",
                report_to="wandb",
                push_to_hub=args.push_to_hub,
                hub_model_id=args.repo_id,
                remove_unused_columns=False,
            ),
        )
    else:
        # setup the trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=data,
            args=SFTConfig(
                dataset_text_field=args.dataset_text_field,
                dataset_num_proc=args.num_proc,
                max_seq_length=args.max_seq_length,
                per_device_train_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_steps=args.warmup_steps,
                max_steps=args.max_steps,
                learning_rate=args.learning_rate,
                lr_scheduler_type=args.lr_scheduler_type,
                weight_decay=args.weight_decay,
                bf16=args.bf16,
                logging_strategy="steps",
                logging_steps=10,
                output_dir=args.output_dir,
                optim="paged_adamw_8bit",
                seed=args.seed,
                run_name=f"train-{args.model_id.split('/')[-1]}",
                report_to="wandb",
                push_to_hub=args.push_to_hub,
                hub_model_id=args.repo_id,
            ),
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
