import argparse
import asyncio
import json
import os
from agents.instruction_generation import generate_instructions
from agents.instruction_refinement import refine_instructions
from agents.content_transformation import content_transformation_flow
from utils.agent_utils import load_agent_configs
from utils.text_extraction import extract_text_chunks_from_dataset
from dotenv import load_dotenv
import nest_asyncio
import signal
from agents.async_chat import async_chat_completion  # Import the function
from functools import partial

nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process dataset with specified task.")
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4o-mini",  # Default model
    help="The model to use for LLM completions."
)
parser.add_argument(
    "--dataset-name",
    type=str,
    default=os.getenv("DATASET_NAME", "crumb/openstax-text"),
    help="The name of the dataset to process. Can also be set via the DATASET_NAME environment variable."
)
parser.add_argument(
    "--task-name",
    type=str,
    default=os.getenv("TASK_NAME", "reading_comprehension"),
    help="The name of the task. Can also be set via the TASK_NAME environment variable."
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode to limit the number of chunks processed."
)
args = parser.parse_args()

# Create a partial function with the model set
from functools import partial
async_chat_with_model = partial(async_chat_completion, model=args.model)

# Define file paths
os.makedirs("./data/generated_data", exist_ok=True)  # Ensure the generated data folder exists
os.makedirs(".cache", exist_ok=True)  # Ensure the .cache folder exists

# Use both task name and dataset name for progress tracking
DATA_FILE = f'./data/generated_data/{args.task_name}.jsonl'
PROGRESS_FILE = f'.cache/{args.task_name}_{args.dataset_name.replace("/", "_")}_progress.json'

async def process_chunk(chunk_index, text, content_agents, instruction_agents, debug, semaphore, queue, async_chat_completion):
    async with semaphore:
        print(f"Processing chunk {chunk_index + 1}...")

        try:
            # Pass async_chat_completion to the agent functions
            transformed_contents = await content_transformation_flow(text, content_agents, async_chat_completion, debug)
            instruction_answer_pairs = await generate_instructions(transformed_contents, instruction_agents, async_chat_completion, debug)
            refined_pairs = await refine_instructions(instruction_answer_pairs, async_chat_completion, max_rounds=2)

            await queue.put(refined_pairs)
            await queue.put({'processed_chunk': chunk_index})

        except Exception as e:
            print(f"Error processing chunk {chunk_index + 1}: {e}")
            await queue.put({'error': {'chunk': chunk_index, 'error': str(e)}})

async def writer(queue):
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                processed_chunks = set(json.load(f))
            except json.JSONDecodeError:
                processed_chunks = set()
    else:
        processed_chunks = set()

    with open(DATA_FILE, 'a') as data_f, open(PROGRESS_FILE, 'w') as progress_f:
        while True:
            item = await queue.get()
            if item is None:
                break

            if isinstance(item, list):
                for pair in item:
                    data_f.write(json.dumps(pair) + '\n')
                data_f.flush()
            elif isinstance(item, dict):
                if 'processed_chunk' in item:
                    processed_chunks.add(item['processed_chunk'])
                elif 'error' in item:
                    print(f"Error in chunk {item['error']['chunk'] + 1}: {item['error']['error']}")
            queue.task_done()

        progress_f.write(json.dumps(list(processed_chunks)))
        progress_f.flush()

async def main(async_chat_completion):
    content_agents, instruction_agents = load_agent_configs(args.task_name)

    text_chunks = extract_text_chunks_from_dataset(
        dataset_name=args.dataset_name,
        split='train',
        text_field='text',
        chunk_size=20000,
        use_samples=False,
    )

    print(f"Extracted {len(text_chunks)} text chunks from the dataset '{args.dataset_name}'.")

    # Use debug mode to limit the chunks processed
    chunks_to_process = text_chunks[1000:1010] if args.debug else text_chunks

    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                processed_chunks = set(json.load(f))
            except json.JSONDecodeError:
                processed_chunks = set()
    else:
        processed_chunks = set()

    tasks_to_create = [
        (index, text) for index, text in enumerate(chunks_to_process)
        if index not in processed_chunks
    ]

    if not tasks_to_create:
        print("All chunks have been processed.")
        return

    semaphore = asyncio.Semaphore(1)
    queue = asyncio.Queue()

    writer_task = asyncio.create_task(writer(queue))
    tasks = [
        asyncio.create_task(
            process_chunk(index, text, content_agents, instruction_agents, args.debug, semaphore, queue, async_chat_completion)
        )
        for index, text in tasks_to_create
    ]

    def shutdown():
        print("Received stop signal. Cancelling tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        asyncio.create_task(queue.put(None))

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        print("Tasks have been cancelled.")
    finally:
        await queue.put(None)
        await writer_task

    print(f"{args.task_name.capitalize()} task processing complete!")

if __name__ == '__main__':
    asyncio.run(main(async_chat_with_model))
