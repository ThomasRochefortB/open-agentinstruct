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

nest_asyncio.apply()

# Load environment variables
load_dotenv()

chosen_task = 'multiple_choice_question'
DATA_FILE = f'{chosen_task}.jsonl'
PROGRESS_FILE = f'{chosen_task}_progress.json'

async def process_chunk(chunk_index, text, content_agents, instruction_agents, debug, semaphore, queue):
    async with semaphore:
        print(f"Processing chunk {chunk_index + 1}...")

        try:
            # Directly await the async content_transformation_flow
            transformed_contents = await content_transformation_flow(text, content_agents, debug)

            # Directly await the async generate_instructions
            instruction_answer_pairs = await generate_instructions(transformed_contents, instruction_agents, debug)

            # Step 4: Instruction Refinement Flow (asynchronous)
            refined_pairs = await refine_instructions(instruction_answer_pairs, max_rounds=2)

            # Put the result into the queue for writing
            await queue.put(refined_pairs)

            # Update progress
            await queue.put({'processed_chunk': chunk_index})

        except Exception as e:
            print(f"Error processing chunk {chunk_index + 1}: {e}")
            # Optionally, you can log errors to a separate file or queue
            await queue.put({'error': {'chunk': chunk_index, 'error': str(e)}})


async def writer(queue):
    # Load processed chunks
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                processed_chunks = set(json.load(f))
            except json.JSONDecodeError:
                processed_chunks = set()
    else:
        processed_chunks = set()

    # Open both files in append mode, using 'with' to ensure they close properly
    with open(DATA_FILE, 'a') as data_f, open(PROGRESS_FILE, 'a') as progress_f:
        while True:
            item = await queue.get()
            if item is None:
                # Sentinel received, terminate writer
                break

            if isinstance(item, list):
                # This is refined_pairs to append as a JSON line
                for pair in item:
                    data_f.write(json.dumps(pair) + '\n')  # Writing each item as JSON line
                data_f.flush()  # Ensure data is written to disk immediately
            elif isinstance(item, dict):
                if 'processed_chunk' in item:
                    # Log the processed chunk index
                    progress_f.write(json.dumps(item['processed_chunk']) + '\n')
                    progress_f.flush()  # Ensure data is written to disk immediately
                elif 'error' in item:
                    # Handle errors (optional)
                    print(f"Error in chunk {item['error']['chunk'] + 1}: {item['error']['error']}")
            queue.task_done()


async def main():
    # Set the debug flag
    debug = False  # Set to True for debug mode, False for normal execution

    # Load agent configurations
    content_agents, instruction_agents = load_agent_configs(chosen_task)

    # Extract text chunks
    text_chunks = extract_text_chunks_from_dataset(
        dataset_name='crumb/openstax-text',
        split='train',
        text_field='text',
        chunk_size=20000,
        use_samples=False,
    )
    
    print(f"Extracted {len(text_chunks)} text chunks from the dataset.")

    # Limit the number of text chunks in debug mode
    chunks_to_process = text_chunks[1000:1010] if debug else text_chunks

    # Load processed chunks to avoid reprocessing
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            try:
                processed_chunks = set(json.load(f))
            except json.JSONDecodeError:
                processed_chunks = set()
    else:
        processed_chunks = set()

    # Filter out already processed chunks
    tasks_to_create = [
        (index, text) for index, text in enumerate(chunks_to_process)
        if index not in processed_chunks
    ]

    if not tasks_to_create:
        print("All chunks have been processed.")
        return

    dataset = []
    semaphore = asyncio.Semaphore(1)  # Adjust concurrency limit as needed
    queue = asyncio.Queue()

    # Start the writer coroutine
    writer_task = asyncio.create_task(writer(queue))

    # Create processing tasks
    tasks = [
        process_chunk(index, text, content_agents, instruction_agents, debug, semaphore, queue)
        for index, text in tasks_to_create
    ]

    # Handle graceful shutdown
    def shutdown():
        print("Received stop signal. Cancelling tasks...")
        for task in tasks:
            task.cancel()
        asyncio.create_task(queue.put(None))  # Signal writer to terminate

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    try:
        # Run all processing tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        print("Tasks have been cancelled.")
    finally:
        # Signal the writer to finish
        await queue.put(None)
        await writer_task

    print("Dataset generation complete!")

if __name__ == '__main__':
    asyncio.run(main())
