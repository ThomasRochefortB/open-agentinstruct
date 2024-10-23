# utils/get_datasets.py

from datasets import load_dataset


def load_dataset_from_hf(dataset_name, cache_dir):
    """
    Load a dataset from Hugging Face and cache it in the specified directory.
    """
    return load_dataset(dataset_name, cache_dir=cache_dir)


def download_datasets(dataset_names=None, cache_dir="data/seed_data"):
    """
    Download specified datasets from Hugging Face and cache them in the specified directory.
    If no dataset names are provided, download all predefined datasets.
    """
    if dataset_names is None:
        dataset_names = [
            "query-of-CC/Knowledge_Pile",
            "crumb/openstax-text",
            "math-ai/AutoMathText",
        ]

    seed_data = {}
    for name in dataset_names:
        seed_data[name] = load_dataset_from_hf(name, cache_dir)

    return seed_data
