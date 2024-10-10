import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
import os
from datetime import datetime
import argparse
import csv

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

def format_example(question, choices, answer=None):
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    if answer is not None:
        prompt += f"\nAnswer: {chr(65 + int(answer))}\n\n"
    else:
        prompt += "\nAnswer: "
    return prompt

def get_model_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1)
    
    answer = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip()
    answer = answer[0] if answer else ''
    return answer if answer in ['A', 'B', 'C', 'D'] else 'A'

def run_mmlu_benchmark(model_name, num_samples=None, few_shot=False, verbose=True):
    model, tokenizer = load_model_and_tokenizer(model_name)
    dataset = load_dataset("cais/mmlu", "all")
    
    dev_set = dataset["dev"]
    test_set = dataset["test"]
    
    if num_samples is not None:
        test_set = test_set.shuffle().select(range(min(num_samples, len(test_set))))
    
    dev_by_subject = defaultdict(list)
    for example in dev_set:
        dev_by_subject[example['subject']].append(example)
    
    correct = 0
    total = 0
    results_by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for sample in tqdm(test_set, disable=not verbose):
        question = sample['question']
        choices = sample['choices']
        correct_answer = sample['answer']
        subject = sample['subject']
        
        if few_shot:
            prompt = ""
            for few_shot_sample in dev_by_subject[subject]:
                prompt += format_example(few_shot_sample['question'], few_shot_sample['choices'], few_shot_sample['answer'])
            prompt += format_example(question, choices)
        else:
            prompt = format_example(question, choices)
        
        model_answer = get_model_answer(model, tokenizer, prompt)
        
        model_answer_index = ord(model_answer) - ord('A')
        correct_answer_index = int(correct_answer)
        
        if verbose:
            print(f"Subject: {subject}")
            print(f"Question: {question}")
            print(f"Model answer: {model_answer} ({model_answer_index}), Correct answer: {chr(65 + correct_answer_index)} ({correct_answer_index})")
            print(f"Choices: {choices}")
            print("---")
        
        if model_answer_index == correct_answer_index:
            correct += 1
            results_by_subject[subject]["correct"] += 1
        total += 1
        results_by_subject[subject]["total"] += 1
    
    accuracy = correct / total
   
    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

def write_results_to_csv(results):
    csv_path = os.path.join('benchmarks', 'benchmark_results.csv')
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['date', 'model_name', 'benchmark', 'setting', 'num_samples', 'metric', 'metric_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for setting in ['zero_shot', 'few_shot']:
            writer.writerow({
                'date': results['date'],
                'model_name': results['model_name'],
                'benchmark': results['benchmark'],
                'setting': setting,
                'num_samples': results['num_samples'],
                'metric': 'accuracy',
                'metric_value': results[setting]['overall_accuracy']
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU benchmark")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Name of the model to benchmark")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate. If None, use entire test set")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output during evaluation")
    args = parser.parse_args()

    print(f"Running zero-shot evaluation for {args.model_name}:")
    zero_shot_results = run_mmlu_benchmark(args.model_name, args.num_samples, few_shot=False, verbose=args.verbose)
    print(f"MMLU Zero-shot Accuracy: {zero_shot_results['overall_accuracy']:.2%}")
    
    print(f"\nRunning 5-shot evaluation for {args.model_name}:")
    few_shot_results = run_mmlu_benchmark(args.model_name, args.num_samples, few_shot=True, verbose=args.verbose)
    print(f"MMLU 5-shot Accuracy: {few_shot_results['overall_accuracy']:.2%}")
    
    # Prepare results for CSV
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {
        'date': current_date,
        'model_name': args.model_name,
        'benchmark': "MMLU",
        'num_samples': args.num_samples if args.num_samples is not None else "full_test_set",
        'zero_shot': zero_shot_results,
        'few_shot': few_shot_results
    }
    
    write_results_to_csv(results)
    
    print(f"\nResults have been written to {os.path.join('benchmarks', 'benchmark_results.csv')}")