# agentinstruct_medicinal_chemistry.py

import re
import openai
from dotenv import load_dotenv  
import os

# Load environment variables from .env file
load_dotenv()
# Access the API key using the variable name defined in the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import any additional modules if necessary
from utils.text_extraction import extract_text_chunks_from_pdf

from agents.base_agent import BaseAgent
from agents.instruction_generation import generate_instructions
from agents.instruction_refinement import refine_instructions

import json
import openai
from dotenv import load_dotenv  
import os

# Load environment variables from .env file
load_dotenv()
# Access the API key using the variable name defined in the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

# Open and load the JSON data from the file
with open('agents/agent_configs.json', 'r') as json_file:
    agent_configs = json.load(json_file)
# Main functions
def content_transformation_flow(text):
    transformed_contents = []
    for config in agent_configs:
        agent = BaseAgent(
            name=config['name'],
            system_prompt=config['system_prompt'],
            user_prompt_template=config['user_prompt_template']
        )
        agent_output = agent.process(text)
        if agent_output:
            transformed_contents.extend(agent_output)
    return transformed_contents


def generate_responses(instructions):
    dataset = []
    for instr in instructions:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": instr}
            ]
        )
        answer = response.choices[0].message.content
        dataset.append({'instruction': instr, 'response': answer})
    return dataset


def main(pdf_path):
    # Step 1: Extract text chunks
    text_chunks = extract_text_chunks_from_pdf(pdf_path, pages_per_chunk=5)
    
    dataset = []
    
    for chunk_index, text in enumerate(text_chunks[0:1]):
        print(f"Processing chunk {chunk_index + 1}/{len(text_chunks)}...")
        
        # Step 2: Improved Content Transformation
        transformed_contents = content_transformation_flow(text)
        
        # Step 3: Seed Instruction Generation
        instructions = generate_instructions(transformed_contents)
        
        # Step 4: Instruction Refinement
        refined_instructions = refine_instructions(instructions)
        
        # Step 5: Dataset Compilation
        chunk_dataset = generate_responses(refined_instructions)
        
        dataset.extend(chunk_dataset)
    
    # Save the complete dataset to a file
    import json
    with open('medicinal_chemistry_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=4)
    
    print("Dataset generation complete!")

# Example usage
if __name__ == "__main__":
    pdf_path = 'data/textbook2.pdf'  # Replace with your PDF file path
    main(pdf_path)