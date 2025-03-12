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
| **Fermi**                            | :heavy_check_mark:     |
| **Coding**                           |                        |
| **Text Extraction**                  |                        |
| **Text Classification**              |                        |
| **Retrieval Augmented Generation**   |                        |
| **Tool Use**                         |                        |
| **Creative Content Generation**      | :heavy_check_mark:     |
| **Few Shot Reasoning**               | :heavy_check_mark:     |
| **Conversation**                     | :heavy_check_mark:     |

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



## Features
- LiteLLM compatible LLMs
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

3. Set up your API keys necessary to use the desired LiteLLM model:
    - Create a `.env` file in the root directory.
    - Add your API key to the `.env` file:
        ```
        OPENAI_API_KEY=your_openai_api_key
        COHERE_API_KEY=your_cohere_api_key
        ANTHROPIC_API_KEY=your_anthropic_key
        ...
        ```
4. Benchmarking requires the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) library. To install please [follow this](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

## Usage

1. Run the main script:
- To run on a HF dataset
    ```sh
    python gen_data.py --dataset_name <hf/datasetname>
    ```
- To run on a set of seed `.pdf`s
    ```sh
    python gen_data.py --pdf-dir data/seed_data/
    ```
2. Train a model on the generated dataset:
    ```sh
    python finetuning/finetune.py 
    ```
3. Benchmark your finetuned model:
    ```sh
   lm_eval --model hf \
    --model_args pretrained=<your finetuned model path> \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda \
    --batch_size auto \
    --output_path results
    ```

## Example of generated data:
```json
{"instruction": "What is the primary message or theme of the passage regarding the immune system?", "answer": "The primary message of the passage is that the immune system, while important for protecting the body from diseases, has significant limitations and challenges. It highlights the difficulties in vaccine development, the potential for the immune system to fail in recognizing pathogens, and the negative outcomes associated with immune responses, suggesting that its effectiveness may not be as reliable as previously portrayed.", "context": "The immune system is a complex network that, while essential for protecting the body from infections and diseases, also has notable limitations and challenges. Its ability to respond to a wide range of pathogens is not always effective, as evidenced by the ongoing difficulties in developing vaccines for certain diseases, such as HIV and malaria. These challenges highlight that the immune system, despite its intended role as a defender of health, can sometimes fail to recognize harmful invaders or may overreact, leading to autoimmune diseases and allergies. \nMoreover, while advancements in immunology have been made, the progress is often slow and fraught with setbacks. The hygiene hypothesis, which suggests that exposure to various environmental factors can positively influence immune development, remains a topic of debate, raising questions about its universal applicability. In transplantation, the immune system's role is equally complex; while immunosuppressive therapies have improved transplant success rates, they also illustrate the immune system's struggle to accept necessary foreign elements, such as transplanted organs, which can lead to rejection.\nIn conclusion, while the immune system is a crucial component of our health, its limitations and the challenges it faces cannot be overlooked. Ongoing research is essential, but many questions remain unanswered, suggesting that the immune system's capabilities may not be as reliable as once thought. By critically examining the immune system's weaknesses, we can better understand the need for effective treatments and interventions that address its shortcomings.", "agent": "Refinement Round 2, Goal 2"}

```
## Project Structure

- The main script to generate datasets using the open-agentinstruct workflow is in `gen_data.py`

- The `agents/` folder holds: 
    - `content_gen_agents.json` : The prompts defining the content transformation agents for the various tasks
    - `instruction_gen_agents.json` : The prompts defining the instruction generation agents for the various tasks
    - `content_transformation.py` : The code for the content transformation agent 
    - `instruction_generation.py` : The code for the instruction generation agent
    - `instruction_refinement.py` : The code for the instruction refinement agent

- The `utils/` folder holds:

- The `data/` folder holds:
