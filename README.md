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
The AgentInstruct paper implements the following tasks which are not all implemented yet in open-agentinstruct:
| **AgentInstruct Task name**           | **Open-AgentInstruct** |
|:-------------------------------------|:----------------------:|
| **Reading Comprehension**             | :heavy_check_mark:     |
| **Open Domain Question Answering**    | :heavy_check_mark:     |
| **Text Modification**                 | :heavy_check_mark:     |
| **Web Agent**                         | :heavy_check_mark:     |
| **Brain Teaser**                      | :heavy_check_mark:     |
| **Analytical Reasoning**              | :heavy_check_mark:     |
| **Multiple Choice Questions**         | :heavy_check_mark:     |
| **Data To Text**                      | :heavy_check_mark:     |
| **Fermi**                             | :heavy_check_mark:     |
| **Coding**                            | :heavy_check_mark:     |
| **Text Extraction**                   | :heavy_check_mark:     |
| **Text Classification**               | :heavy_check_mark:     |
| **Retrieval Augmented Generation**    | :heavy_check_mark:     |
| **Tool Use**                          | :heavy_check_mark:     |
| **Creative Content Generation**       | :heavy_check_mark:     |
| **Few Shot Reasoning**                | :heavy_check_mark:     |
| **Conversation**                      | :heavy_check_mark:     |

<!-- What benchmarks will evaluate these:
- [MMLU](https://huggingface.co/datasets/cais/mmlu) (Multiple choice questions)
- [DROP](https://huggingface.co/datasets/ucinlp/drop) (Reading comprehension) -->

## Supported seed datasets
- Any HF datasets:
    - The AgentInstruct uses the following:
        - [Knowledge Pile](https://huggingface.co/datasets/Query-of-CC/Knowledge_Pile)
        - [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText)
        - subset of [openstax](https://huggingface.co/datasets/crumb/openstax-text)
        - subset Apache 2.0 from [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean)
- Any set of user-defined set of seed `.pdf`s






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

# Example: Generate data for all tasks from a PDF directory, including original content
open-agentinstruct-generate --pdf-dir path/to/your/pdfs --all-tasks --include-content

# See all available options
open-agentinstruct-generate --help
```

Generated data will be saved to `./data/generated_data/<task_name>.jsonl` by default.

*Note: The finetuning and benchmarking scripts (`finetuning/finetune.py`, `lm_eval.sh`) are currently outside the installable package structure. You would run them directly from your cloned repository if needed.*

## Example of generated data:
```json
{"instruction": "What is the primary message or theme of the passage regarding the immune system?", "answer": "The primary message of the passage is that the immune system, while important for protecting the body from diseases, has significant limitations and challenges. It highlights the difficulties in vaccine development, the potential for the immune system to fail in recognizing pathogens, and the negative outcomes associated with immune responses, suggesting that its effectiveness may not be as reliable as previously portrayed.", "context": "The immune system is a complex network that, while essential for protecting the body from infections and diseases, also has notable limitations and challenges. Its ability to respond to a wide range of pathogens is not always effective, as evidenced by the ongoing difficulties in developing vaccines for certain diseases, such as HIV and malaria. These challenges highlight that the immune system, despite its intended role as a defender of health, can sometimes fail to recognize harmful invaders or may overreact, leading to autoimmune diseases and allergies. \nMoreover, while advancements in immunology have been made, the progress is often slow and fraught with setbacks. The hygiene hypothesis, which suggests that exposure to various environmental factors can positively influence immune development, remains a topic of debate, raising questions about its universal applicability. In transplantation, the immune system's role is equally complex; while immunosuppressive therapies have improved transplant success rates, they also illustrate the immune system's struggle to accept necessary foreign elements, such as transplanted organs, which can lead to rejection.\nIn conclusion, while the immune system is a crucial component of our health, its limitations and the challenges it faces cannot be overlooked. Ongoing research is essential, but many questions remain unanswered, suggesting that the immune system's capabilities may not be as reliable as once thought. By critically examining the immune system's weaknesses, we can better understand the need for effective treatments and interventions that address its shortcomings.", "agent": "Refinement Round 2, Goal 2"}

```
## Project Structure

-   **`open_agentinstruct/`**: The main source code for the installable package.
    -   `__init__.py`: Makes the directory a Python package.
    -   `cli.py`: Contains the command-line interface logic (formerly `gen_data.py`), powered by `argparse`.
    -   `agents/`: Contains the core agent logic and configuration files.
        -   `__init__.py`
        -   `async_chat.py`: Handles asynchronous calls to LLMs.
        -   `content_transformation.py`: Implements the content transformation flow.
        -   `instruction_generation.py`: Implements the instruction generation flow.
        -   `instruction_refinement.py`: Implements the instruction refinement flow.
        -   `split_agents/`: Contains JSON configuration files defining agent prompts and parameters for different tasks.
    -   `utils/`: Contains utility functions.
        -   `__init__.py`
        -   `agent_utils.py`: Utilities related to agent configuration loading.
        -   `text_extraction.py`: Functions for extracting text from datasets and PDFs.
        -   *(Other utility modules)*
-   **`data/`**: Default directory for generated data and potentially seed data (excluded from package).
-   **`.cache/`**: Default directory for progress tracking files (excluded from package).
-   **`finetuning/`**: Example scripts for finetuning models (excluded from package).
-   **`tests/`**: Contains unit and integration tests (excluded from package). *(You should add tests here!)*
-   **`pyproject.toml`**: Defines build system requirements and project metadata (PEP 518, PEP 621).
-   **`setup.cfg`**: Provides detailed configuration for the `setuptools` build backend (dependencies, package discovery, entry points, package data).
-   **`README.md`**: This file.
-   **`LICENSE`**: Project license file.
-   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.
-   **`.github/workflows/`**: Contains GitHub Actions workflow files for CI/CD (testing, publishing).

Example execution commands from the root directory after installation:

```sh
# Generate data for all tasks from the specified dataset, processing max 100 chunks, skipping refinement, including content
open-agentinstruct-generate --dataset-names "crumb/openstax-text" --all-tasks --max-chunks 100 --skip-refinement --include-content
```
