import re
import json
import sys
from pathlib import Path
from typing import List


def extract_answer_letter(answer: str) -> str:
    """
    Extract the letter-based answer or raise a ValueError if not found.
    Assumes the answer is in the format: 'X) "Some verbose text..."'.
    """
    match = re.match(r"([A-Z])\)", answer.strip())
    if match:
        return match.group(1)  # Return the selected letter (e.g., 'D')
    else:
        raise ValueError("No valid letter-based answer found.")


def filter_and_extract_answers(input_file: str, output_file: str) -> None:
    """
    Read a .jsonl file, filter and extract valid answers,
    and write to a new .jsonl file with only the letter as the answer.

    Parameters:
    - input_file: Path to the original .jsonl file.
    - output_file: Path to the filtered .jsonl file.
    """
    valid_samples = []

    # Read and process the dataset
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            try:
                # Extract the letter-based answer
                letter = extract_answer_letter(sample["answer"])
                # Replace the answer with only the letter
                sample["answer"] = letter
                valid_samples.append(sample)
            except ValueError:
                pass  # Skip samples without a valid letter-based answer

    # Write the filtered dataset to a new .jsonl file
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")

    print(
        f"Filtered dataset saved to {output_file} with {len(valid_samples)} valid samples."
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_dataset.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Ensure input file exists
    if not Path(input_file).is_file():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Create the filtered dataset
    filter_and_extract_answers(input_file, output_file)
