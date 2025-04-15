![1-8d2861cb](https://github.com/user-attachments/assets/2717bf2a-8f6a-4043-9538-8b832118798c)
# open-agentinstruct

An open-source recreation of the [AgentInstruct](https://arxiv.org/pdf/2407.03502v1) agentic workflow.

`open-agentinstruct` is a project aimed at recreating the AgentInstruct agentic workflow. It supports any LiteLLM model to be used in the agentic synthetic data generation worflow. The AgentInstruct workflow involves three agentic step for synthetic data generation based on "seed" data:
- **Content Transformation**: Transforms text content using various agent configurations.
- **Instruction Generation**: Generates instructions based on transformed content.
- **Instruction Refinement**: Refines generated instructions to enhance complexity and challenge.

## Table of Contents
- [Supported tasks](#supported-tasks)
- [Supported seed datasets](#supported-seed-datasets)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example of generated data](#example-of-generated-data)
- [Project Structure](#project-structure)

## Supported tasks
The AgentInstruct paper implements the following tasks which are all implemented in open-agentinstruct:

- Reading Comprehension
- Open Domain Question Answering
- Text Modification
- Web Agent
- Brain Teaser
- Analytical Reasoning
- Multiple Choice Questions
- Data To Text
- Fermi
- Coding
- Text Extraction
- Text Classification
- Retrieval Augmented Generation
- Tool Use
- Creative Content Generation
- Few Shot Reasoning
- Conversation


## Supported seed datasets
- Any HF datasets:
    - The AgentInstruct paper uses the following:
        - [Knowledge Pile](https://huggingface.co/datasets/Query-of-CC/Knowledge_Pile)
        - [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText)
        - subset of [openstax](https://huggingface.co/datasets/crumb/openstax-text)
        - subset Apache 2.0 from [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean)
- Any set of user-provided seed `.pdf`s

## Features
- LiteLLM compatible LLMs
- Finetuning pipeline for llama3

## Installation

**Option 1: Install from PyPI (Recommended for users)**

Once the package is published, you can install it directly using pip:

```sh
pip install open-agentinstruct
```

**Option 2: Install from source (For developers)**

1.  Clone the repository:
    ```sh
    git clone https://github.com/ThomasRochefortB/open-agentinstruct.git
    cd open-agentinstruct
    ```

2.  Create a virtual environment (recommended):
    ```sh
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

3.  Install the package in editable mode along with development dependencies:
    ```sh
    pip install -e ".[dev]"
    ```

4.  Set up your API keys necessary to use the desired LiteLLM model(s):
    *   Create a `.env` file in the root directory (or wherever you run the command).
    *   Add your API key(s) to the `.env` file (the library uses `python-dotenv` to load them):
        ```dotenv
        # Example for OpenAI
        OPENAI_API_KEY=your_openai_api_key
        # Add other keys as needed (e.g., COHERE_API_KEY, ANTHROPIC_API_KEY)
        # ...
        ```

## Usage

The primary way to use the data generation workflow is through the command-line interface:

```sh
# Basic usage with a Hugging Face dataset
open-agentinstruct-generate --dataset-names <hf/datasetname> --task-name <your_task_name>

# Example: Generate reading comprehension data from the first 100 chunks of openstax
open-agentinstruct-generate --dataset-names "crumb/openstax-text" --task-name reading_comprehension --max-chunks 100

# Generate data for all tasks from the specified dataset, processing max 100 chunks, skipping refinement, including content
open-agentinstruct-generate --dataset-names "crumb/openstax-text:text:train:20000" --model gemini/gemini-2.0-flash --max-chunks 100 --output-dir ./output

# Example: Generate data for all tasks from a PDF directory, including original content
open-agentinstruct-generate --pdf-dir path/to/your/pdfs --all-tasks --include-content

# See all available options
open-agentinstruct-generate --help
```

Generated data will be saved to `./data/generated_data/<task_name>.jsonl` by default.
