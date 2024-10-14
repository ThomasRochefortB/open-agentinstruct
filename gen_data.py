import asyncio
import concurrent.futures
import json
from agents.instruction_generation import generate_instructions
from agents.instruction_refinement import refine_instructions
from agents.content_transformation import content_transformation_flow
from utils.agent_utils import load_agent_configs
from utils.text_extraction import extract_text_chunks_from_dataset
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

chosen_task = 'reading_comprehension'
async def process_chunk(chunk_index, text, content_agents, instruction_agents, debug, semaphore):
    async with semaphore:
        print(f"Processing chunk {chunk_index + 1}...")

        loop = asyncio.get_running_loop()

        try:
            # Run synchronous functions in an executor
            transformed_contents = await loop.run_in_executor(
                None, content_transformation_flow, text, content_agents, debug
            )

            instruction_answer_pairs = await loop.run_in_executor(
                None, generate_instructions, transformed_contents, instruction_agents, debug
            )

            # Step 4: Instruction Refinement Flow (asynchronous)
            refined_pairs = await refine_instructions(instruction_answer_pairs, max_rounds=2)

            return refined_pairs
        except Exception as e:
            print(f"Error processing chunk {chunk_index + 1}: {e}")
            return []

async def main():
    # Set the debug flag
    debug = True  # Set to True for debug mode, False for normal execution

    # Load agent configurations
    content_agents, instruction_agents = load_agent_configs(chosen_task)

    # Extract text chunks
    text_chunks = extract_text_chunks_from_dataset(
        dataset_name='crumb/openstax-text',
        split='train',
        text_field='text',
        chunk_size=5000,
        use_samples=False,
    )

    # Limit the number of text chunks in debug mode
    chunks_to_process = text_chunks[16000:16005] if debug else text_chunks

    dataset = []
    semaphore = asyncio.Semaphore(5)  # Adjust concurrency limit as needed

    tasks = [
        process_chunk(index, text, content_agents, instruction_agents, debug, semaphore)
        for index, text in enumerate(chunks_to_process)
    ]

    # Run all tasks concurrently
    all_refined_pairs = await asyncio.gather(*tasks)

    # Flatten the list of lists and extend the dataset
    for refined_pairs in all_refined_pairs:
        dataset.extend(refined_pairs)

    # Save the complete dataset to a file
    try:
        with open('{}.json'.format(chosen_task), 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    existing_data.extend(dataset)

    with open('{}.json'.format(chosen_task), 'w') as f:
        json.dump(existing_data, f, indent=4)

    print("Dataset generation complete!")

if __name__ == '__main__':
    asyncio.run(main())
