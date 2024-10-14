import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from datetime import datetime
import argparse
import csv
import re
import pandas as pd
from typing import List, Dict, Union

class DROPTask:
    HISTORY = "history"
    LITERATURE = "literature"
    SCIENCE = "science"
    PHILOSOPHY_RELIGION = "philosophy_religion"
    MUSIC_THEATER_DANCE = "music_theater_dance"
    WORLD_HISTORY = "world_history"
    AMERICAN_HISTORY = "american_history"
    EUROPEAN_HISTORY = "european_history"
    OTHER = "other"

class DROPTemplate:
    @staticmethod
    def format_question(data: Dict, include_answer: bool = True) -> str:
        question = f"Passage: {data['passage']}\n\nQuestion: {data['question']}\n\n"
        if include_answer:
            answers = ', '.join(data['answers_spans']['spans'])
            question += f"Answer: {answers}\n\n"
        return question

    @staticmethod
    def generate_prompt(train_set: List[Dict], input: str, n_shots: int) -> str:
        prompt = "Answer the following questions based on the given passages. Provide a short, specific answer.\n\n"
        for i in range(n_shots):
            prompt += DROPTemplate.format_question(train_set[i], include_answer=True)
        prompt += input + "\nAnswer:"
        return prompt

def get_model_answer(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # Post-process the answer
    answer = answer.split('\n')[0]  # Take only the first line
    answer = re.sub(r'^Answer:\s*', '', answer)  # Remove "Answer:" prefix if present
    answer = re.findall(r'\d+(?:\.\d+)?|\w+', answer)  # Extract numbers and words
    if answer:
        return answer[0]  # Return the first extracted item
    return ""

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def normalize_answer(answer):
    answer = re.sub(r'[^\w\s]', '', answer.lower())
    return ' '.join(answer.split())

def evaluate_exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def run_drop_benchmark(model_name, num_samples=None, n_shots=5, verbose=False):
    model, tokenizer = load_model_and_tokenizer(model_name)
    dataset = load_dataset("drop", split="validation")
    
    if num_samples is not None:
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    
    train_set = load_dataset("drop", split="train")
    shots_set = []
    categories_seen = set()
    for data in train_set:
        category = data['section_id']
        if category not in categories_seen:
            categories_seen.add(category)
            shots_set.append(data)
            if len(shots_set) == n_shots:
                break
    
    correct = 0
    total = 0
    predictions = []
    
    for sample in tqdm(dataset, disable=not verbose):
        input_text = DROPTemplate.format_question(sample, include_answer=False)
        prompt = DROPTemplate.generate_prompt(shots_set, input_text, n_shots)
        
        model_answer = get_model_answer(model, tokenizer, prompt)
        ground_truth = sample['answers_spans']['spans']
        
        is_correct = any(evaluate_exact_match(model_answer, answer) for answer in ground_truth)
        if is_correct:
            correct += 1
        total += 1
        
        predictions.append({
            'Task': sample['section_id'],
            'Input': input_text,
            'Prediction': model_answer,
            'Correct': is_correct
        })
        
        if verbose:
            print(f"Question: {sample['question']}")
            print(f"Model answer: {model_answer}")
            print(f"Correct answers: {ground_truth}")
            print(f"Exact match: {is_correct}")
            print("---")
    
    accuracy = correct / total
    predictions_df = pd.DataFrame(predictions)
    task_scores = predictions_df.groupby('Task')['Correct'].mean().reset_index()
    task_scores.columns = ['Task', 'Score']
    
    return {
        "exact_match": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions_df,
        "task_scores": task_scores
    }

def write_results_to_csv(results):
    csv_path = os.path.join('benchmarks', 'benchmark_results.csv')
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['date', 'model_name', 'benchmark', 'setting', 'num_samples', 'metric', 'metric_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'date': results['date'],
            'model_name': results['model_name'],
            'benchmark': results['benchmark'],
            'setting': f"{results['n_shots']}-shot",
            'num_samples': results['num_samples'],
            'metric': 'exact_match',
            'metric_value': results['results']['exact_match']
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DROP benchmark")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Name of the model to benchmark")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate. If None, use entire validation set")
    parser.add_argument("--n_shots", type=int, default=5, help="Number of few-shot examples to use")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output during evaluation")
    args = parser.parse_args()

    print(f"Running DROP evaluation for {args.model_name}:")
    drop_results = run_drop_benchmark(args.model_name, args.num_samples, args.n_shots, verbose=args.verbose)
    print(f"DROP Exact Match Accuracy: {drop_results['exact_match']:.2%}")
    
    # Prepare results for CSV
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {
        'date': current_date,
        'model_name': args.model_name,
        'benchmark': "DROP",
        'n_shots': args.n_shots,
        'num_samples': args.num_samples if args.num_samples is not None else "full_validation_set",
        'results': drop_results
    }
    
    write_results_to_csv(results)
    
    print(f"\nResults have been written to {os.path.join('benchmarks', 'benchmark_results.csv')}")
    
    # Save detailed results
    drop_results['predictions'].to_csv(os.path.join('benchmarks', 'drop_predictions.csv'), index=False)
    drop_results['task_scores'].to_csv(os.path.join('benchmarks', 'drop_task_scores.csv'), index=False)
    print(f"Detailed results have been saved to 'benchmarks/drop_predictions.csv' and 'benchmarks/drop_task_scores.csv'")