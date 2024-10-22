import os
import json
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

def base_model_prompt(instruction: str, context: str) -> str:
    return f"Instruction: {instruction}\n\nContext: {context}\n\nAnswer:"

def main():
    # Set the path to the finetuned model and tokenizer
    model_path = "llama-finetuned-with-context"

    # Load the finetuned tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For generation with left padding

    # Load the finetuned model
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Use torch.float16 if bfloat16 is not supported
        device_map="auto",
    )
    model.eval()  # Set the model to evaluation mode

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load a sample question from the dataset
    data_file = "data/generated_data/multiple_choice_question.jsonl"
    with open(data_file, 'r') as f:
        # Read the first line as a sample
        line = f.readline()
        sample = json.loads(line)

    instruction = sample.get('instruction', '')
    context = sample.get('context', '')

    # Build the prompt using the same template used during fine-tuning
    prompt = base_model_prompt(instruction, context)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the model's answer
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract the answer (excluding the prompt)
    answer_start = len(prompt)
    generated_answer = generated_text[answer_start:].strip()

    # Print the prompt and the generated answer
    print("\nPrompt:")
    print(prompt)
    print("\nGenerated Answer:")
    print(generated_answer)

if __name__ == "__main__":
    main()
