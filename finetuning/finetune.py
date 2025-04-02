import argparse
import json
from typing import Any, Dict, List
import torch
import glob  # Import the glob module
import os # Import os module
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
)

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset, concatenate_datasets

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


# -------------------------------
# Main Fine-Tuning Function
# -------------------------------

def main(args):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Check and set pad token
    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id is not set. Setting it to eos_token_id. Consider adding a dedicated pad token for better performance with DataCollatorForCompletionOnlyLM.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif tokenizer.pad_token_id == tokenizer.eos_token_id:
        print("Warning: pad_token_id is equal to eos_token_id. Consider adding a dedicated pad token for better performance with DataCollatorForCompletionOnlyLM.")

    # Set padding side to left. Necessary for Completion Only LM
    tokenizer.padding_side = 'left'

    # Set ChatML template if not already set
    if not tokenizer.chat_template:
        print("Setting ChatML template...")
        chatml_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
        tokenizer.chat_template = chatml_template

    print("Loading dataset...")
    # Find all .jsonl files in the specified directory
    train_files_pattern = os.path.join(args.train_data_dir, "*.jsonl")
    train_files = glob.glob(train_files_pattern)
    if not train_files:
        raise ValueError(f"No .jsonl files found in directory: {args.train_data_dir}")
    print(f"Found training files: {train_files}")

    raw_datasets = load_dataset("json", data_files=train_files)

    # Split dataset - keep the original 'messages' column
    split_datasets = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_datasets['train']
    test_dataset = split_datasets['test']

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

    training_args = SFTConfig(
        output_dir="finetuned_models",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        learning_rate=5e-6,
        logging_steps=10,
        save_strategy="epoch",
        gradient_accumulation_steps=1,
        warmup_ratio=0.05,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        max_seq_length=args.max_seq_length,
        **extra_training_args,
    )
    
    # Define response template and instantiate collator
    # The template should match the start of the assistant's response in the ChatML format
    response_template = "<|im_start|>assistant\n"
    # We don't need instruction_template for ChatML format when using apply_chat_template
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # We don't need a formatting_func, SFTTrainer handles chat templates now
        data_collator=collator,        # Pass the custom collator
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
        "--train_data_dir",  # Changed argument name
        type=str,            # Changed type to string
        required=True,
        help="Directory containing the JSONL training files.", # Updated help text
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
        default="./finetuned_models",
        help="Directory where the model will be saved",
    )

    args = parser.parse_args()
    main(args)