import argparse
import json
from typing import Any, Dict, List
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset, concatenate_datasets

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# -------------------------------
# Preprocess Function
# -------------------------------


def preprocess_function(
    examples: Dict[str, Any], tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """Preprocess the dataset examples using chat template or default formatting."""
    inputs = [
        apply_chat_template_if_available(
            tokenizer,
            {
                "instruction": examples["instruction"][i],
                "context": examples["context"][i],
                "answer": examples["answer"][i],
            },
        )[0]
        for i in range(len(examples["instruction"]))
    ]
    return {"text": inputs}


# -------------------------------
# Chat Template Helper
# -------------------------------


def apply_chat_template_if_available(
    tokenizer: AutoTokenizer, examples: Dict[str, Any]
) -> List[str]:
    """Apply chat template if available, otherwise use default formatting."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"{examples['context']}\n{examples['instruction']}",
                },
                {"role": "assistant", "content": examples["answer"]},
            ]
            tokenized_chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return tokenized_chat
        except Exception as e:
            print(
                f"Error applying chat template: {e}. Falling back to default formatting."
            )

    # Fallback to default formatting: Context first, then instruction.
    return [
        f"Context: {examples['context']}\n\n"
        f"Instruction: {examples['instruction']}\n\n"
        f"Answer: {examples['answer']}"
    ]


# -------------------------------
# Load and Preprocess Dataset
# -------------------------------


def load_jsonl_dataset(file_paths: List[str]) -> Dataset:
    datasets = [
        load_dataset("json", data_files=file_path)["train"] for file_path in file_paths
    ]
    return concatenate_datasets(datasets)


# -------------------------------
# Main Fine-Tuning Function
# -------------------------------


def main(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading dataset...")
    raw_datasets = load_jsonl_dataset(args.train_files)

    split_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets["train"]
    val_dataset = split_datasets["test"]

    print("Preprocessing datasets...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
    )

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_attention_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_train_dataset = tokenized_train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    tokenized_val_dataset = tokenized_val_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    data_collator = DataCollatorForCompletionOnlyLM("Answer:", tokenizer=tokenizer)

    print("Preparing training arguments...")
    # Parse any extra training arguments passed via CLI
    extra_training_args = (
        json.loads(args.training_kwargs) if args.training_kwargs else {}
    )

    # Initialize TrainingArguments with defaults and any extra arguments
    training_args = TrainingArguments(
        output_dir="finetuned_models",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=8e-6,
        logging_steps=1,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        gradient_accumulation_steps=2,
        optim="adafactor",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        seed=42,
        **extra_training_args,  # Add extra CLI-provided arguments
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete and model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a HF model with custom datasets."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="HuggingFaceTB/SmolLM-135M",
        help="Path to the pre-trained model or model identifier from huggingface.co.",
    )
    parser.add_argument(
        "--train_files",
        nargs="+",
        required=True,
        help="List of paths to JSONL training files.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--training_kwargs",
        type=str,
        default="{}",
        help="Additional TrainingArguments as a JSON string.",
    )

    args = parser.parse_args()
    main(args)
