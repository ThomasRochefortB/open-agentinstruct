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
    default="gemini/gemini-2.0-flash-lite",
    help="The model to use for LLM completions. This information will be included in each generated sample.",
)
parser.add_argument(
    "--dataset-names",
    type=str,
    nargs="+",
    help="List of Hugging Face datasets to process. Format: 'dataset_name' or 'dataset_name:text_field'.",
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
    "--agent-config-path",
    type=str,
    default="agents/split_agents",
    help="Path to the base directory containing agent configuration files.",
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
    help="Include the original text and transformed content in the output JSON. If not specified, these fields will be excluded to save space.",
)
parser.add_argument(
    "--max-chunks",
    type=int,
    default=0,
    help="Maximum number of input chunks to process. Set to 0 to process all chunks.",
)
parser.add_argument(
    "--text-fields",
    type=str,
    nargs="+",
    default=["text"],
    help="Field names containing text in HuggingFace datasets. You can provide a single field name to use for all datasets or one field name per dataset in the same order.",
)
parser.add_argument(
    "--all-tasks",
    action="store_true",
    help="Run the script for all available tasks instead of just one.",
)
args = parser.parse_args()

if not args.dataset_names and not args.pdf_dir:
    parser.error("Either --dataset-names or --pdf-dir must be specified")

async_chat_with_model = partial(async_chat_completion, model=args.model)

# Define file paths
os.makedirs("./data/generated_data", exist_ok=True)
os.makedirs(".cache", exist_ok=True)

source_name = args.dataset_names[0] if args.dataset_names else Path(args.pdf_dir).name
DATA_FILE = f"./data/generated_data/{args.task_name}.jsonl"
PROGRESS_FILE = (
    f'.cache/{args.task_name}_{source_name.replace("/", "_")}_progress.jsonl'
)


def get_text_chunks():
    if args.dataset_names:
        all_chunks = []
        
        # Process dataset names and extract text fields if provided in the format "dataset_name:text_field"
        datasets_and_fields = []
        for dataset_spec in args.dataset_names:
            if ":" in dataset_spec:
                # Split "dataset_name:text_field" into separate parts
                parts = dataset_spec.split(":", 1)
                dataset_name = parts[0]
                text_field = parts[1]
                datasets_and_fields.append((dataset_name, text_field))
            else:
                # No text field specified, use the default text field (from args.text_fields)
                dataset_name = dataset_spec
                datasets_and_fields.append((dataset_name, None))
        
        # For any datasets without a specified text field, use the provided --text-fields
        text_fields = args.text_fields
        if len(text_fields) == 1:
            default_text_field = text_fields[0]
        else:
            default_text_field = "text"
        
        for idx, (dataset_name, specified_field) in enumerate(datasets_and_fields):
            # Use the specified field if provided, otherwise fall back to the text_fields argument
            if specified_field:
                text_field = specified_field
            elif idx < len(text_fields):
                text_field = text_fields[idx]
            else:
                text_field = default_text_field
                
            print(f"Processing dataset: {dataset_name} using text field: {text_field}")
            
            try:
                chunks = extract_text_chunks_from_dataset(
                    dataset_name=dataset_name,
                    split="train",
                    text_field=text_field,
                    chunk_size=20000,
                    use_samples=False,
                )
                chunks = [(dataset_name, chunk) for chunk in chunks]
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {e}")
                
        return all_chunks
    elif args.pdf_dir:
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
                source_file = args.dataset_names[0] if args.dataset_names else args.pdf_dir
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
                # Add model information to each pair
                pair["model"] = args.model

            await queue.put((chunk_index, instruction_answer_pairs))
            await queue.put({"processed_chunk": chunk_index})

        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {e}")
            await queue.put({"error": {"chunk": chunk_index, "error": str(e)}})

async def writer(queue):
    """
    Legacy writer function for backward compatibility.
    This is replaced by task_writer in the process_task function.
    """
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
                    # Format as chat with roles
                    chat_messages = [
                        {"role": "user", "content": pair.get("instruction", "")},
                        {"role": "assistant", "content": pair.get("answer", "")}
                    ]
                    
                    # Add metadata if needed
                    if args.include_content and ("transformed_content" in pair or "original_text" in pair):
                        metadata = {k: v for k, v in pair.items() 
                                 if k not in ["instruction", "answer"]}
                        chat_data = {"messages": chat_messages, "metadata": metadata}
                        data_f.write(json.dumps(chat_data) + "\n")
                    else:
                        # Just write the messages array directly
                        data_f.write(json.dumps(chat_messages) + "\n")
                        
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

async def process_task(task_name, async_chat_completion):
    """Process a single task."""
    # Set the task name for the current execution
    args.task_name = task_name

    # Update the file paths for this task
    data_file = f"./data/generated_data/{task_name}.jsonl"
    progress_file = (
        f'.cache/{task_name}_{source_name.replace("/", "_")}_progress.jsonl'
    )
    
    print(f"\n{'='*50}")
    print(f"Processing task: {task_name}")
    print(f"{'='*50}\n")
    
    try:
        content_agents, instruction_agents, one_shot_example = load_agent_configs(
            args.agent_config_path, task_name
        )

        # Get new text chunks for each task to ensure different samples
        text_chunks = get_text_chunks()
        source_type = "dataset" if args.dataset_names else "PDF directory"
        print(f"Extracted {len(text_chunks)} text chunks from the {source_type}.")

        # Create a list of (original_index, chunk) pairs
        chunks_with_indices = list(enumerate(text_chunks))

        # Shuffle chunks if --random is specified, with a new random seed for each task
        if args.random:
            # Use task name as part of the random seed to ensure different shuffles for each task
            task_specific_seed = random.randint(0, 10000) + hash(task_name) % 10000
            task_rng = random.Random(task_specific_seed)
            task_rng.shuffle(chunks_with_indices)
            print(f"Shuffled the text chunks for task {task_name} with seed {task_specific_seed}.")

        # Process 10 chunks if --debug is specified
        if args.debug:
            chunks_with_indices = chunks_with_indices[:10]
        # Limit to max-chunks if specified and not in debug mode
        elif args.max_chunks > 0:
            chunks_with_indices = chunks_with_indices[:args.max_chunks]
            print(f"Limited processing to {args.max_chunks} chunks as requested.")

        processed_chunks = set()
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                for line in f:
                    index = int(line.strip())
                    processed_chunks.add(index)

        tasks_to_create = [
            (index, chunk)
            for index, chunk in chunks_with_indices
            if index not in processed_chunks
        ]

        if not tasks_to_create:
            print(f"All chunks for task {task_name} have been processed.")
            return

        semaphore = asyncio.Semaphore(10)
        queue = asyncio.Queue()

        # Create a specialized writer function for this task
        async def task_writer(q):
            with open(data_file, "a") as data_f, open(progress_file, "a") as progress_f:
                while True:
                    item = await q.get()
                    if item is None:
                        break

                    if isinstance(item, tuple):
                        chunk_index, refined_pairs = item
                        for pair in refined_pairs:
                            # Format as chat with roles
                            chat_messages = [
                                {"role": "user", "content": pair.get("instruction", "")},
                                {"role": "assistant", "content": pair.get("answer", "")}
                            ]
                            
                            # Add metadata if needed
                            if args.include_content and ("transformed_content" in pair or "original_text" in pair):
                                # Extract metadata (excluding instruction/answer which are now in the messages)
                                metadata = {k: v for k, v in pair.items() 
                                         if k not in ["instruction", "answer"]}
                                chat_data = {"messages": chat_messages, "metadata": metadata}
                                data_f.write(json.dumps(chat_data) + "\n")
                            else:
                                # Just write the messages array directly like in convert_to_chat.py
                                data_f.write(json.dumps(chat_messages) + "\n")
                                
                        data_f.flush()
                    elif isinstance(item, dict):
                        if "processed_chunk" in item:
                            chunk_index = item["processed_chunk"]
                            progress_f.write(f"{chunk_index}\n")
                            progress_f.flush()
                        elif "error" in item:
                            print(
                                f"Error in chunk {item['error']['chunk']}: {item['error']['error']}"
                            )
                    q.task_done()

        writer_task = asyncio.create_task(task_writer(queue))
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

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            print(f"Tasks for {task_name} have been cancelled.")
            raise  # Re-raise the exception
        finally:
            await queue.put(None)
            await writer_task

        print(f"{task_name.capitalize()} task processing complete!")
    except Exception as e:
        print(f"Error processing task {task_name}: {e}")
    
    return tasks  # Return tasks for potential cancellation from the main function

async def main(async_chat_completion):
    """Main function to run the data generation process."""
    # Single signal handler for immediate exit
    def shutdown():
        print("\nReceived interrupt - exiting immediately!")
        os._exit(1)  # Force exit the process

    # Register signal handlers at start
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    if args.all_tasks:
        # Get all available task names by looking at instruction JSON files
        task_names = []
        instruction_dir = os.path.join(args.agent_config_path, "instruction_generation")
        for filename in os.listdir(instruction_dir):
            if filename.endswith("_instruction.json"):
                task_name = filename.replace("_instruction.json", "")
                task_names.append(task_name)
        
        print(f"Found {len(task_names)} available tasks: {', '.join(task_names)}")
        
        # Process each task sequentially
        for task_name in task_names:
            await process_task(task_name, async_chat_completion)
        
        print("\nAll tasks processing completed or interrupted!")
    else:
        # Process a single task
        task_tasks = await process_task(args.task_name, async_chat_completion)
        if task_tasks:
            print("\nAll tasks processing completed or interrupted!")


if __name__ == "__main__":
    asyncio.run(main(async_chat_with_model))
