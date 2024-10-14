import os
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset, concatenate_datasets

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

# Define Roles
ROLES = ["system", "user", "assistant", "ipython"]

# -------------------------------
# Load and Preprocess Dataset
# -------------------------------

def load_jsonl_dataset(file_paths: List[str]) -> Dataset:
    """
    Load one or more JSONL files into a Hugging Face Dataset.
    """
    datasets = []
    for file_path in file_paths:
        datasets.append(load_dataset("json", data_files=file_path)["train"])
    return concatenate_datasets(datasets)

def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert instruction-context-answer pairs into the desired prompt format.
    """
    inputs = []
    for instruction, context, answer in zip(examples["instruction"], examples["context"], examples["answer"]):
        prompt = (
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
        inputs.append(prompt)
    return {"text": inputs}

# -------------------------------
# Main Fine-Tuning Function
# -------------------------------

def main(model_name_or_path: str, train_files: List[str]):
    # -------------------------------
    # Configuration
    # -------------------------------
    output_dir = "llama-finetuned-with-context"
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    num_train_epochs = 3
    logging_steps = 100
    save_steps = 500
    eval_steps = 500
    learning_rate = 5e-5
    max_seq_length = 256  # Adjust based on GPU memory

    # -------------------------------
    # Load Dataset
    # -------------------------------
    print("Loading dataset...")
    raw_datasets = load_jsonl_dataset(train_files)
    print(f"Number of training examples: {len(raw_datasets)}")

    # -------------------------------
    # Preprocess Dataset
    # -------------------------------
    print("Preprocessing dataset...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets.column_names,
    )

    # -------------------------------
    # Initialize Tokenizer
    # -------------------------------
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Add special tokens to the tokenizer
    additional_special_tokens = list(SPECIAL_TOKENS.values())
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

    # Set the pad_token to your defined padding token
    tokenizer.pad_token = SPECIAL_TOKENS["finetune_right_pad_id"]

    # -------------------------------
    # Initialize Model
    # -------------------------------
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=False,  # Set to True if using bitsandbytes for 8-bit training
        # torch_dtype=torch.float16,
        device_map="auto",
    )

    # Resize token embeddings to accommodate new special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Set the pad_token_id in the model's configuration
    model.config.pad_token_id = tokenizer.pad_token_id

    # -------------------------------
    # Tokenize Dataset
    # -------------------------------
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_attention_mask=True,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing dataset...")
    tokenized_datasets = tokenized_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Verify tokenized dataset
    print("Columns after tokenization:", tokenized_datasets.column_names)
    print("Example tokenized data:", tokenized_datasets[0])

    # -------------------------------
    # Data Collator
    # -------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # -------------------------------
    # Training Arguments
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        gradient_accumulation_steps=4,
        optim="adafactor",
        gradient_checkpointing=True,
        save_total_limit=2,
        tf32=True,  # Enable mixed precision
        push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
        remove_unused_columns=False,  # Temporarily set to False
    )

    # -------------------------------
    # Initialize Trainer
    # -------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=None,  # Add eval_dataset if you have one
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # -------------------------------
    # Start Training
    # -------------------------------
    print("Starting training...")
    trainer.train()

    # -------------------------------
    # Save Model
    # -------------------------------
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete and model saved.")

if __name__ == "__main__":
    model_name_or_path = "meta-llama/Llama-3.2-1B"  # Replace with your LLaMA model path or Hugging Face model ID
    train_files = ["reading_comprehension.json"]  # List of paths to your training data JSON files
    main(model_name_or_path, train_files)
