# open-agentinstruct

An open-source recreation of the [AgentInstruct](https://arxiv.org/pdf/2407.03502v1) agentic workflow.

## Supported tasks
The AgentInstruct paper implements the following tasks which are not all implemented yet in open-agentinstruct:
|            **AgentInstruct Task name**           | **Open-AgentInstruct** |
|:----------------------------------:|:------------------------:|
| **Reading Comprehension**          |             :heavy_check_mark:             |
| **Open Domain Question Answering** |                          |
| **Text Modification**              |             :heavy_check_mark:              |
| **Web Agent**                      |                          |
| **Brain Teaser**                   |                          |
| **Analytical Reasoning**           |                :heavy_check_mark:           |
| **Multiple Choice Questions**      |              :heavy_check_mark:             |
| **Data To Text**                   |                          |
| **Fermi**                          |                          |
| **Coding**                         |                          |
| **Text Extraction**                |                          |
| **Text Classification**            |                          |
| **Retrieval Augmented Generation** |                          |
| **Tool Use**                       |                          |
| **Creative Content Generation**    |                          |
| **Few Shot Reasoning**             |                          |
| **Conversation**                   |                          |

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




<!-- ### Which model will we support?
The paper uses Mistral-7b and compares to Mistral-7b instruct. To limit the hardware requirements at the start, we will use:

-  [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and compare to its [instruct version](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) -->

<!-- ### Current results:
| **Benchmark** | **# shots** |     **Metric**     | **LLama 3.2 1B** | **Llama 3.2 1B-Instruct** | **Llama 3.2 3B** | **OpenOrca3** |
|:-------------:|:-----------:|:------------------:|:----------------:|:-------------------------:|:----------------:|:-------------:|
|      MMLU     |      5      | macro_avg/acc_char |       32.2       |            49.3           |       58.0       |               |
|      DROP     |      3      |         f1         |       28.0       |            N/A            |       45.2       |               |

 -->

---
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

`open-agentinstruct` is a project aimed at recreating the AgentInstruct agentic workflow. It supports any LiteLLM model to be used in the agentic synthetic data generation worflow.

## Features

- **Content Transformation**: Transforms text content using various agent configurations.
- **Instruction Generation**: Generates instructions based on transformed content.
- **Instruction Refinement**: Refines generated instructions to enhance complexity and challenge.
- Finetuning pipeline for llama3 

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/ThomasRochefortB/open-agentinstruct.git
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
4. Benchmarking requires the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library. To install please [follow this](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

## Usage

1. Run the main script:
    ```sh
    python gen_data.py --task-name "reading_comprehension" --dataset-name "<hf_dataset_path>"
    ```

## Project Structure

The main script to generate datasets using the open-agentinstruct workflow is in `gen_data.py`

The agents/  folder holds:
- Prompts for content_transformation and instruction_generation agents
- Code for content_transformation, instruction_generation and instruction_refinement

The benchmarks/ folder holds:
- The bash script using 