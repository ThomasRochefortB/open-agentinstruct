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

from trl import SFTConfig, SFTTrainer

from datasets import load_dataset, Dataset, concatenate_datasets

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


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
        print("Setting ChatML template...")
        chatml_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        tokenizer.chat_template = chatml_template

    print("Loading dataset...")
    raw_datasets = load_dataset("json", data_files=args.train_files)

    split_datasets = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)

    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = split_datasets['train'].map(template_dataset, remove_columns=["messages"])
    test_dataset = split_datasets['test'].map(template_dataset, remove_columns=["messages"])
    


    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # device_map="auto",
    )
    if torch.backends.mps.is_available():
        model = model.to("mps")


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
        save_strategy="epoch",
        gradient_accumulation_steps=2,
        optim="adafactor",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        save_total_limit=2,
        # bf16=torch.cuda.is_bf16_supported(),
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        seed=42,
        **extra_training_args,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
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
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
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