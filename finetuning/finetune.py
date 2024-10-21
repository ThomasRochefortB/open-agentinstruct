import os
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# -------------------------------
# Define Special Tokens
# -------------------------------

SPECIAL_TOKENS = {
    "begin_of_text": "<|begin_of_text|>",
    "end_of_text": "<|end_of_text|>",
    "finetune_right_pad_id": "<|finetune_right_pad_id|>",
    "start_header_id": "<|start_header_id|>",
    "end_header_id": "<|end_header_id|>",
    "eom_id": "<|eom_id|>",
    "eot_id": "<|eot_id|>",
    "python_tag": "<|python_tag|>",
}

# -------------------------------
# Prompt Templates
# -------------------------------

def preprocess_function(examples: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
    inputs = []
    for i in range(len(examples["instruction"])):
        if prompt_template == "instruct":
            prompt = instruct_prompt(
                examples["instruction"][i],
                examples["context"][i],
                examples["answer"][i]
            )
        else:
            prompt = base_model_prompt(
                examples["instruction"][i],
                examples["context"][i],
                examples["answer"][i]
            )
        inputs.append(prompt)
    return {"text": inputs}

def instruct_prompt(instruction: str, context: str, answer: str) -> str:
    return (
        f"{SPECIAL_TOKENS['begin_of_text']}"
        f"{SPECIAL_TOKENS['start_header_id']}system{SPECIAL_TOKENS['end_header_id']}\n\n"
        f"You are a helpful assistant.\n"
        f"{SPECIAL_TOKENS['eot_id']}"
        f"{SPECIAL_TOKENS['start_header_id']}user{SPECIAL_TOKENS['end_header_id']}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Context: {context}\n"
        f"{SPECIAL_TOKENS['eot_id']}"
        f"{SPECIAL_TOKENS['start_header_id']}assistant{SPECIAL_TOKENS['end_header_id']}\n"
        f"{answer}\n"
        f"{SPECIAL_TOKENS['eot_id']}"
    )

def base_model_prompt(instruction: str, context: str, answer: str) -> str:
    return f"Instruction: {instruction}\n\nContext: {context}\n\nAnswer: {answer}"

# -------------------------------
# Load and Preprocess Dataset
# -------------------------------

def load_jsonl_dataset(file_paths: List[str]) -> Dataset:
    datasets = [load_dataset("json", data_files=file_path)["train"] for file_path in file_paths]
    return concatenate_datasets(datasets)

# -------------------------------
# Main Fine-Tuning Function
# -------------------------------

def freeze_model_except_last_layers(model, num_layers_to_train: int = 2):
    """
    Freezes all model parameters except the last few Transformer layers and the lm_head.
    """
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few transformer layers
    for layer in model.model.layers[-num_layers_to_train:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Ensure the final lm_head is trainable (for causal language modeling tasks)
    for param in model.lm_head.parameters():
        param.requires_grad = True

    print(f"Only the last {num_layers_to_train} transformer layers and lm_head are trainable.")

def main(model_name_or_path: str, train_files: List[str], prompt_template: str):
    output_dir = "llama-finetuned-with-context"
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    num_train_epochs = 1
    learning_rate = 2e-8
    max_seq_length = 2048

    print("Loading dataset...")
    raw_datasets = load_jsonl_dataset(train_files)
    split_datasets = raw_datasets.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets["train"]
    val_dataset = split_datasets["test"]

    print("Preprocessing datasets...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, prompt_template),
        batched=True, remove_columns=train_dataset.column_names
    )
    tokenized_val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, prompt_template),
        batched=True, remove_columns=val_dataset.column_names
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Freeze all layers except the last few and the lm_head
    # freeze_model_except_last_layers(model, num_layers_to_train=2)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], padding="max_length", truncation=True,
            max_length=max_seq_length, return_attention_mask=True
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

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        gradient_accumulation_steps=8,
        remove_unused_columns=True,
        optim="adafactor",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        save_total_limit=2,
        bf16=True,
        tf32=True,
        weight_decay=0.1,
        lr_scheduler_type="constant",
        seed=42,
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete and model saved.")

if __name__ == "__main__":
    model_name_or_path = "meta-llama/Llama-3.2-1B"
    train_files = ["data/generated_data/multiple_choice_question.jsonl", 
                   "data/generated_data/reading_comprehension.jsonl"]
    prompt_template = "base_model"

    main(model_name_or_path, train_files, prompt_template)

