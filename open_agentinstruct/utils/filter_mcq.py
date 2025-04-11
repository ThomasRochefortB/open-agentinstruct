import re
import json
import sys
from pathlib import Path


def extract_answer_letter(answer: str) -> str:
    """
    Extract the letter-based answer or return None if not found.
    Assumes the answer is in the format: 'X) "Some verbose text..."'.
    """
    match = re.match(r"([A-Z])\)", answer.strip())
    if match:
        return match.group(1)  # Return the letter (e.g., 'B')
    else:
        return None  # Indicate no valid letter was found


def has_question_choices(instruction: str) -> bool:
    """
    Check if the instruction contains question choices.
    Looks for the presence of 'A)' to 'D)' (or more) patterns.
    """
    return bool(re.search(r"\b[A-Z]\)", instruction))


def filter_and_extract_answers(input_file: str, output_file: str) -> None:
    """
    Filter dataset to remove samples without question choices,
    and extract valid answer letters if possible.

    Parameters:
    - input_file: Path to the original .jsonl file.
    - output_file: Path to the filtered .jsonl file.
    """
    valid_samples = []

    # Read and process the dataset
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            # Skip samples without question choices
            if not has_question_choices(sample.get("instruction", "")):
                continue

            # Attempt to extract the answer letter
            letter = extract_answer_letter(sample.get("answer", ""))

            if letter:
                # Replace answer with the extracted letter
                sample["answer"] = letter

            # Append the sample, even if letter extraction fails
            valid_samples.append(sample)

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
