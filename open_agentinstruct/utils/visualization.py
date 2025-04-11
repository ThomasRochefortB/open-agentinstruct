import json
import textwrap
import argparse
import sys
import random


def load_dataset(dataset_path):
    """Load a dataset from a JSON or JSONL file."""
    try:
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, "r") as f:
                dataset = [json.loads(line) for line in f]
        elif dataset_path.endswith(".json"):
            with open(dataset_path, "r") as f:
                dataset = json.load(f)
        else:
            print("Unsupported file format. Please use a .json or .jsonl file.")
            sys.exit(1)
        return dataset
    except FileNotFoundError:
        print(f"Dataset file not found at {dataset_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)


def filter_complete_samples(dataset):
    """Filter out incomplete samples that don't have both an answer and instruction with choices."""
    return [
        sample
        for sample in dataset
        if sample.get("answer") and "Choices" in sample.get("instruction", "")
    ]


def format_mcq(sample, wrap_width=80):
    """Format and return an MCQ sample as a string."""
    instruction = sample.get("instruction", "No instruction provided.")
    answer = sample.get("answer", "No answer provided.")
    agent = sample.get("agent", "No agent information")
    instruction_agent = sample.get(
        "instruction_agent", "No instruction agent information"
    )
    transformed_content = sample.get("transformed_content", "No transformed content")
    source = sample.get("source", "No source available")
    original_text = sample.get("original_text", "No original text available")

    # Split instruction into question and choices
    parts = instruction.split("Choices:", 1)
    question = parts[0].strip()
    choices = parts[1].strip() if len(parts) > 1 else "No choices provided"

    output = []
    output.append("Question:")
    output.append(textwrap.fill(question, width=wrap_width))
    output.append("\nChoices:")
    for choice in choices.split("\n"):
        output.append(textwrap.fill(choice.strip(), width=wrap_width))
    output.append(f"\nCorrect Answer: {answer}")
    output.append(f"\nRefinement Agent: {agent}")
    output.append(f"\nInstruction Agent: {instruction_agent}")
    if transformed_content and transformed_content != "``` \n```":
        output.append("\nTransformed Content:")
        output.append(textwrap.fill(transformed_content, width=wrap_width))
    output.append("\nOriginal Text:")
    output.append(textwrap.fill(original_text, width=wrap_width))
    output.append(f"\nSource: {source}")
    output.append("\n" + "-" * wrap_width + "\n")

    return "\n".join(output)


def visualize_dataset(dataset, num_samples=5, wrap_width=80, filter_incomplete=False):
    """
    Visualizes and returns samples from the dataset as a string.

    Args:
        dataset (list): List of dataset entries.
        num_samples (int): Number of samples to display.
        wrap_width (int): The width to wrap text for better readability.
        filter_incomplete (bool): Whether to filter out incomplete samples.

    Returns:
        str: The formatted output as a string.
    """
    if filter_incomplete:
        dataset = filter_complete_samples(dataset)

    total_samples = len(dataset)

    if total_samples == 0:
        return "No complete samples available in the dataset."

    sampled_indices = random.sample(
        range(total_samples), min(num_samples, total_samples)
    )
    output = [
        f"Displaying {len(sampled_indices)} random samples out of {total_samples}:\n"
    ]

    for i, idx in enumerate(sampled_indices):
        sample = dataset[idx]
        output.append(f"Sample {i + 1}:\n")
        output.append(format_mcq(sample, wrap_width))

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Visualize samples from a dataset.")
    parser.add_argument("dataset_path", help="Path to the dataset (.json or .jsonl).")
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to display."
    )
    parser.add_argument(
        "--wrap_width", type=int, default=80, help="Width to wrap text for readability."
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Only display complete samples with answers and choices.",
    )
    parser.add_argument("--output_file", type=str, help="Path to the output .txt file.")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    output = visualize_dataset(
        dataset,
        num_samples=args.num_samples,
        wrap_width=args.wrap_width,
        filter_incomplete=args.filter,
    )

    if args.output_file:
        try:
            with open(args.output_file, "w") as f:
                f.write(output)
            print(f"Output successfully written to {args.output_file}")
        except IOError as e:
            print(f"Error writing to file: {e}")
    else:
        print(output)


if __name__ == "__main__":
    main()
