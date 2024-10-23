import json
import textwrap


def visualize_dataset(dataset_path, num_samples=5, start_index=0, wrap_width=80):
    """
    Visualizes samples from the medicinal chemistry dataset.

    Args:
        dataset_path (str): Path to the JSON dataset file.
        num_samples (int): Number of samples to display.
        start_index (int): Starting index for displaying samples.
        wrap_width (int): The width to wrap text for better readability.

    Returns:
        None
    """
    # Load the dataset
    try:
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Dataset file not found at {dataset_path}")
        return
    except json.JSONDecodeError:
        print("Error decoding JSON. Ensure the dataset is in the correct format.")
        return

    total_samples = len(dataset)
    end_index = min(start_index + num_samples, total_samples)

    if start_index >= total_samples:
        print("Start index is beyond the dataset size.")
        return

    print(
        f"Displaying samples {start_index + 1} to {end_index} out of {total_samples}:\n"
    )

    for idx in range(start_index, end_index):
        sample = dataset[idx]
        instruction = sample.get("instruction", "No instruction provided.")
        response = sample.get("response", "No response provided.")

        print(f"Sample {idx + 1}:\n")
        print("Instruction:")
        print(textwrap.fill(instruction, width=wrap_width))
        print("\nResponse:")
        print(textwrap.fill(response, width=wrap_width))
        print("\n" + "-" * wrap_width + "\n")
