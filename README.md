# open-agentinstruct

An open-source recreation of the AgentInstruct agentic workflow.

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
