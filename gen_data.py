import argparse
import asyncio
import json
import os
from agents.instruction_generation import generate_instructions
from agents.instruction_refinement import refine_instructions
from agents.content_transformation import content_transformation_flow
from utils.text_extraction import (
    extract_text_chunks_from_dataset,
    extract_text_chunks_from_pdf,
)
from utils.agent_utils import load_agent_configs

from dotenv import load_dotenv
import nest_asyncio
import signal
from agents.async_chat import async_chat_completion
from functools import partial
from pathlib import Path
import random

nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Process dataset or PDFs with specified task."
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4o-mini",
    help="The model to use for LLM completions.",
)
parser.add_argument(
    "--dataset-name",
    type=str,
    help="The name of the Hugging Face dataset to process.",
)
parser.add_argument(
    "--pdf-dir",
    type=str,
    help="Directory containing PDF files to process.",
)
parser.add_argument(
    "--pdf-engine",
    type=str,
    default="pdfminer",
    choices=["pdfminer", "pymupdf", "unstructured"],
    help="The PDF extraction engine to use.",
)
parser.add_argument(
    "--task-name",
    type=str,
    default=os.getenv("TASK_NAME", "reading_comprehension"),
    help="The name of the task. Can also be set via the TASK_NAME environment variable.",
)
parser.add_argument(
    "--content-agent-config",
    type=str,
    default="agents/content_gen_agents.json",
    help="Path to the content generation agents configuration file.",
)
parser.add_argument(
    "--instruction-agent-config",
    type=str,
    default="agents/instruction_gen_agents.json",
    help="Path to the instruction generation agents configuration file.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode to limit the number of chunks processed.",
)
parser.add_argument(
    "--random",
    action="store_true",
    help="Process chunks in random order by shuffling them before processing.",
)
parser.add_argument(
    "--skip-refinement",
    action="store_true",
    help="Skip the instruction refinement step.",
)
parser.add_argument(
    "--include-content",
    action="store_true",
    help="Include the transformed content in the output JSON for validation.",
)
args = parser.parse_args()

if not args.dataset_name and not args.pdf_dir:
    parser.error("Either --dataset-name or --pdf-dir must be specified")

async_chat_with_model = partial(async_chat_completion, model=args.model)

# Define file paths
os.makedirs("./data/generated_data", exist_ok=True)
os.makedirs(".cache", exist_ok=True)

source_name = args.dataset_name if args.dataset_name else Path(args.pdf_dir).name
DATA_FILE = f"./data/generated_data/{args.task_name}.jsonl"
PROGRESS_FILE = (
    f'.cache/{args.task_name}_{source_name.replace("/", "_")}_progress.jsonl'
)


def get_text_chunks():
    if args.dataset_name:
        return extract_text_chunks_from_dataset(
            dataset_name=args.dataset_name,
            split="train",
            text_field="text",
            chunk_size=20000,
            use_samples=False,
        )
    else:
        all_chunks = []
        pdf_dir = Path(args.pdf_dir)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            print(f"Processing PDF: {pdf_file}")
            chunks = extract_text_chunks_from_pdf(
                str(pdf_file), engine=args.pdf_engine, use_images=False
            )
            chunks = [(str(pdf_file), chunk) for chunk in chunks]
            all_chunks.extend(chunks)

        return all_chunks


async def process_chunk(
    chunk_index,
    chunk_data,
    content_agents,
    instruction_agents,
    one_shot_example,
    debug,
    semaphore,
    queue,
    async_chat_completion,
):
    async with semaphore:
        print(f"Processing chunk {chunk_index}...")
        try:
            if isinstance(chunk_data, tuple):
                source_file, text = chunk_data
            else:
                source_file = args.dataset_name
                text = chunk_data

            transformed_contents = await content_transformation_flow(
                text, content_agents, async_chat_completion, debug
            )

            instruction_answer_pairs = await generate_instructions(
                transformed_contents,
                instruction_agents,
                one_shot_example,
                async_chat_completion,
                text,  # Pass the original text
                debug,
            )

            if not args.skip_refinement:
                instruction_answer_pairs = await refine_instructions(
                    instruction_answer_pairs, async_chat_completion, max_rounds=1
                )

            for pair in instruction_answer_pairs:
                pair["source"] = source_file

            await queue.put((chunk_index, instruction_answer_pairs))
            await queue.put({"processed_chunk": chunk_index})

        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {e}")
            await queue.put({"error": {"chunk": chunk_index, "error": str(e)}})

async def writer(queue):
    processed_chunks = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                index = int(line.strip())
                processed_chunks.add(index)

    with open(DATA_FILE, "a") as data_f, open(PROGRESS_FILE, "a") as progress_f:
        while True:
            item = await queue.get()
            if item is None:
                break

            if isinstance(item, tuple):
                chunk_index, refined_pairs = item
                for pair in refined_pairs:
                    data_f.write(json.dumps(pair) + "\n")
                data_f.flush()
            elif isinstance(item, dict):
                if "processed_chunk" in item:
                    chunk_index = item["processed_chunk"]
                    progress_f.write(f"{chunk_index}\n")
                    progress_f.flush()
                    processed_chunks.add(chunk_index)
                elif "error" in item:
                    print(
                        f"Error in chunk {item['error']['chunk']}: {item['error']['error']}"
                    )
            queue.task_done()


async def main(async_chat_completion):
    content_agents, instruction_agents, one_shot_example = load_agent_configs(
        args.content_agent_config, args.instruction_agent_config, args.task_name
    )

    text_chunks = get_text_chunks()
    source_type = "dataset" if args.dataset_name else "PDF directory"
    print(f"Extracted {len(text_chunks)} text chunks from the {source_type}.")

    # Create a list of (original_index, chunk) pairs
    chunks_with_indices = list(enumerate(text_chunks))

    # Shuffle chunks if --random is specified
    if args.random:
        random.shuffle(chunks_with_indices)
        print("Shuffled the text chunks for random processing order.")

    # Process 10 chunks if --debug is specified
    if args.debug:
        chunks_with_indices = chunks_with_indices[:10]

    processed_chunks = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                index = int(line.strip())
                processed_chunks.add(index)

    tasks_to_create = [
        (index, chunk)
        for index, chunk in chunks_with_indices
        if index not in processed_chunks
    ]

    if not tasks_to_create:
        print("All chunks have been processed.")
        return

    semaphore = asyncio.Semaphore(10)
    queue = asyncio.Queue()

    writer_task = asyncio.create_task(writer(queue))
    tasks = [
        asyncio.create_task(
            process_chunk(
                index,
                chunk,
                content_agents,
                instruction_agents,
                one_shot_example,
                args.debug,
                semaphore,
                queue,
                async_chat_completion,
            )
        )
        for index, chunk in tasks_to_create
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


if __name__ == "__main__":
    asyncio.run(main(async_chat_with_model))
