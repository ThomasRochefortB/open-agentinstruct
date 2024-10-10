# open-agentinstruct [WIP]

An open-source recreation of the [AgentInstruct](https://arxiv.org/pdf/2407.03502v1) agentic workflow. [WIP]

Let's plan the implementation to recreate the results of the paper. They implement AgentInstruct flows for 17 capabilities. We will start with a subset of:
- Reading comprehension
- Multiple Choice Questions 

What benchmarks will evaluate these:
- [MMLU](https://huggingface.co/datasets/cais/mmlu) (multiple choice questions)
- DROP (Reading comprehension)
- MMLU-Pro (?Question mark)

### Which dataset will serve as seed data?
The paper uses:
- [Knowledge Pile](https://huggingface.co/datasets/Query-of-CC/Knowledge_Pile)
- [AutoMathText](https://huggingface.co/datasets/math-ai/AutoMathText)
- subset of [openstax](https://huggingface.co/datasets/crumb/openstax-text)
- subset Apache 2.0 from codeparrot/github-code-clean



### Which model will we support?
The paper uses Mistral-7b and compares to Mistral-7b instruct. To limit the hardware requirements at the start, we will use:

-  [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and compare to its [instruct version](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)


### TO-DO:


- [ ] Finetuning pipeline
- [ ] Benchmarking pipeline. We should be able to evaluate the performance of the - instruct baseline on the selected benchmarks



---
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

open-agentinstruct is a project aimed at recreating the AgentInstruct agentic workflow. It leverages OpenAI's GPT models to process and transform text, generate and refine instructions, and compile datasets.

## Features

- **Content Transformation**: Transforms text content using various agent configurations.
- **Instruction Generation**: Generates instructions based on transformed content.
- **Instruction Refinement**: Refines generated instructions to enhance complexity and challenge.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/open-agentinstruct.git
    cd open-agentinstruct
    ```

2. Create the environment using `micromamba`:
    ```sh
    micromamba create -f env.yml
    micromamba activate open-agentinstruct
    ```

3. Set up your OpenAI API key:
    - Create a `.env` file in the root directory.
    - Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY=your_openai_api_key
        ```

## Usage

1. Place your PDF file in the [`data/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fthomas.rochefort-be%2FDocuments%2FGitHub%2Fopen-agentinstruct%2Fdata%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22ef351880-0640-415a-beeb-5d7952bfe3c6%22%5D "/Users/thomas.rochefort-be/Documents/GitHub/open-agentinstruct/data/") directory.
2. Run the main script:
    ```sh
    python main.py
    ```

## Usage

1. Place your PDF file in the [`data/`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FUsers%2Fthomas.rochefort-be%2FDocuments%2FGitHub%2Fopen-agentinstruct%2Fdata%2F%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%2227982eb0-09d3-4bf9-b050-803333a264a4%22%5D "/Users/thomas.rochefort-be/Documents/GitHub/open-agentinstruct/data/") directory.
2. Run the main script:
    ```sh
    python main.py
    ```

## Project Structure
