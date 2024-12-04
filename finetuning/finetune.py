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
from datasets import load_dataset, Dataset, concatenate_datasets

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------------------
# Preprocess Function
# -------------------------------

def preprocess_function(
    examples: Dict[str, Any], tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """Preprocess the dataset examples using chat template."""
    inputs = []
    # Parse the string representation of the messages list if needed
    for messages_str in examples["messages"]:
        if isinstance(messages_str, str):
            messages = json.loads(messages_str)
        else:
            messages = messages_str
            
        # If messages is a single dict, wrap it in a list
        if isinstance(messages, dict):
            messages = [messages]
            
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs.append(chat_text)
    return {"text": inputs}

def load_jsonl_dataset(file_paths: List[str]) -> Dataset:
    datasets = []
    for file_path in file_paths:
        # Load the raw JSONL file
        messages_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # Parse the JSON array from each line
                    messages = json.loads(line.strip())
                    messages_data.append(messages)
        
        # Convert to Dataset format
        dataset = Dataset.from_dict({"messages": messages_data})
        datasets.append(dataset)
    
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

# -------------------------------
# Main Fine-Tuning Function
# -------------------------------

def main(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set ChatML template if not already set
    if not tokenizer.chat_template:
        chatml_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        tokenizer.chat_template = chatml_template

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

    print("Preparing training arguments...")
    extra_training_args = (
        json.loads(args.training_kwargs) if args.training_kwargs else {}
    )

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
        **extra_training_args,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_model",
        help="Directory where the model will be saved",
    )

    args = parser.parse_args()
    main(args)