# open_agentinstruct/cli.py

import argparse
import asyncio
import json
import os
import sys  # Added for main_wrapper

# Use relative imports within the package
from .agents.instruction_generation import generate_instructions
from .agents.instruction_refinement import refine_instructions
from .agents.content_transformation import content_transformation_flow
from .utils.text_extraction import (
    extract_text_chunks_from_dataset,
    extract_text_chunks_from_pdf,
)
from .utils.agent_utils import load_agent_configs
from .agents.async_chat import async_chat_completion

from dotenv import load_dotenv
import nest_asyncio
import signal
from functools import partial
from pathlib import Path
import random
import pkg_resources  # To find package data

nest_asyncio.apply()

# Load environment variables from CWD or parent directories
load_dotenv()

# --- Global variable for args ---
# This is generally not ideal, but refactoring the whole script
# to pass args explicitly is a larger change.
# Consider refactoring later if needed.
args = None

# --- Global variable to track active tasks for cancellation ---
active_tasks = set()


def setup_parser():
    """Sets up the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate instruction-following data from datasets or PDFs using agent workflows."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OAI_MODEL", "gpt-4o-mini"),  # Changed default, use env var
        help="The model identifier to use for LLM completions (e.g., 'openai/gpt-4o-mini', 'gemini/gemini-1.5-flash-latest'). Loaded from OAI_MODEL env var if set.",
    )
    parser.add_argument(
        "--dataset-names",
        type=str,
        nargs="+",
        help="List of Hugging Face datasets to process. Format: 'name' or 'name:text_field' or 'name:text_field:splits' or 'name:text_field:splits:min_chars'.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        help="Directory containing PDF files to process.",
    )
    # Removed --pdf-engine argument
    parser.add_argument(
        "--task-name",
        type=str,
        default=os.getenv("OAI_TASK_NAME", "reading_comprehension"),  # Use env var
        help="The name of the task to generate data for. Loaded from OAI_TASK_NAME env var if set.",
    )
    parser.add_argument(
        "--agent-config-path",
        type=str,
        default=None,  # Default to None, will find package resource path later
        help="Path to the directory containing agent configuration JSON files. Defaults to bundled configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/generated_data",
        help="Directory to save the generated .jsonl files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Directory to save progress files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process only 10 chunks).",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Process chunks in random order.",
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Skip the instruction refinement step.",
    )
    parser.add_argument(
        "--include-content",
        action="store_true",
        help="Include original text and transformed content in output JSON (increases file size).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Maximum number of input chunks to process per task (0 means all).",
    )
    parser.add_argument(
        "--text-fields",
        type=str,
        nargs="+",
        default=["text"],
        help="Field names for text in HuggingFace datasets (one per dataset or one for all).",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run for all available task configurations found in agent-config-path.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of chunks to process concurrently.",
    )
    return parser


def get_default_agent_config_path():
    """Gets the path to the bundled agent configurations within the package."""
    try:
        # This assumes setup.cfg correctly includes the package data
        return pkg_resources.resource_filename(
            "open_agentinstruct", "agents/split_agents"
        )
    except Exception as e:
        print(
            f"Warning: Could not find bundled agent configurations via pkg_resources: {e}"
        )
        # Fallback if running from source without proper installation?
        # This might be fragile.
        script_dir = Path(__file__).parent
        fallback_path = script_dir / "agents" / "split_agents"
        if fallback_path.is_dir():
            print(
                f"Warning: Falling back to relative path for agent configs: {fallback_path}"
            )
            return str(fallback_path)
        else:
            raise FileNotFoundError(
                "Could not locate agent configuration directory."
            ) from e


def get_text_chunks(current_args):
    """Fetches text chunks based on arguments."""
    if current_args.dataset_names:
        all_chunks = []
        # (Dataset processing logic remains largely the same as gen_data.py)
        # ... (Copy dataset processing logic from gen_data.py lines 134-195 here) ...
        # Process dataset names and extract text fields and splits if provided
        datasets_and_fields = []
        for dataset_spec in current_args.dataset_names:
            parts = dataset_spec.split(":")  # Split on all colons
            dataset_name = parts[0]
            text_field = parts[1] if len(parts) > 1 else None

            # Parse multiple splits if provided (comma-separated)
            if len(parts) > 2:
                splits = [s.strip() for s in parts[2].split(",")]
            else:
                splits = ["train"]  # Default to "train" if not specified

            # Parse minimum characters if provided
            min_chars = None
            if len(parts) > 3:
                try:
                    min_chars = int(parts[3])
                except ValueError:
                    print(
                        f"Warning: Invalid minimum character count '{parts[3]}' for dataset {dataset_name}. Using no minimum."
                    )

            datasets_and_fields.append((dataset_name, text_field, splits, min_chars))

        # For any datasets without a specified text field, use the provided --text-fields
        text_fields = current_args.text_fields
        if len(text_fields) == 1:
            default_text_field = text_fields[0]
        else:
            default_text_field = "text"

        for idx, (dataset_name, specified_field, splits, min_chars) in enumerate(
            datasets_and_fields
        ):
            # Use the specified field if provided, otherwise fall back to the text_fields argument
            if specified_field:
                text_field = specified_field
            elif idx < len(text_fields):
                text_field = text_fields[idx]
            else:
                text_field = default_text_field

            print(
                f"Processing dataset: {dataset_name} using text field: {text_field}, splits: {splits}"
            )

            for split in splits:
                try:
                    # Call function from .utils
                    chunks = extract_text_chunks_from_dataset(
                        dataset_name=dataset_name,
                        split=split,
                        text_field=text_field,
                        use_samples=True,
                        min_chars=min_chars,
                    )
                    chunks = [(dataset_name, chunk) for chunk in chunks]
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing dataset {dataset_name} split {split}: {e}")

        return all_chunks
    elif current_args.pdf_dir:
        all_chunks = []
        pdf_dir = Path(current_args.pdf_dir)
        if not pdf_dir.is_dir():
            print(f"Error: PDF directory not found: {pdf_dir}")
            return []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"Warning: No PDF files found in {pdf_dir}")
            return []

        for pdf_file in pdf_files:
            print(f"Processing PDF: {pdf_file}")
            try:
                # Call function from .utils - removed engine parameter
                chunks = extract_text_chunks_from_pdf(str(pdf_file))
                chunks = [(str(pdf_file.name), chunk) for chunk in chunks]
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing PDF {pdf_file}: {e}")

        return all_chunks
    else:
        # This case should be caught by parser error, but added for safety
        return []


async def process_chunk(
    chunk_index,
    chunk_data,
    content_agents,
    instruction_agents,
    one_shot_example,
    current_args,
    semaphore,
    queue,
    async_chat_fn,  # Pass the partial function
):
    """Processes a single chunk of text."""
    async with semaphore:
        print(
            f"Processing chunk {chunk_index + 1}... (# Tasks: {len(active_tasks)})"  # Add active task count
        )
        try:
            source_name = "unknown"
            text = ""
            if isinstance(chunk_data, tuple):
                source_name, text = chunk_data
            else:
                # Fallback logic, try to determine source based on args
                if current_args.dataset_names:
                    # Simplification: Assume first dataset if multiple specified without tuple source
                    source_name = current_args.dataset_names[0]
                elif current_args.pdf_dir:
                    source_name = f"PDFs in {current_args.pdf_dir}"
                text = chunk_data  # Assume chunk_data is the text itself

            if not text:
                print(f"Skipping chunk {chunk_index + 1} due to empty text content.")
                await queue.put({"skipped_chunk": chunk_index})
                return

            # Use relative imports for agent flows
            transformed_contents = await content_transformation_flow(
                text, content_agents, async_chat_fn, current_args.debug
            )

            instruction_answer_pairs = await generate_instructions(
                transformed_contents,
                instruction_agents,
                one_shot_example,
                async_chat_fn,
                text,  # Pass the original text
                current_args.debug,
            )

            if not current_args.skip_refinement:
                refined_pairs = await refine_instructions(
                    instruction_answer_pairs, async_chat_fn, max_rounds=1
                )
            else:
                refined_pairs = instruction_answer_pairs  # Skip refinement

            output_pairs = []
            for pair in refined_pairs:
                # Ensure basic structure even if refinement failed
                final_pair = {
                    "instruction": pair.get("instruction", ""),
                    "answer": pair.get("answer", ""),
                    "source": source_name,
                    "model": current_args.model,
                }
                # Conditionally add original/transformed content
                if current_args.include_content:
                    final_pair["original_text"] = text
                    # Include transformed content if available and differs
                    if (
                        "transformed_content" in pair
                        and pair["transformed_content"] != text
                    ):
                        final_pair["transformed_content"] = pair["transformed_content"]
                    # Include agent info if available
                    if "agent" in pair:
                        final_pair["agent"] = pair["agent"]
                output_pairs.append(final_pair)

            if not output_pairs:
                print(
                    f"No instruction/answer pairs generated for chunk {chunk_index + 1}."
                )
                await queue.put({"empty_chunk": chunk_index})
                return

            await queue.put((chunk_index, output_pairs))
            await queue.put(
                {"processed_chunk": chunk_index}
            )  # Signal successful processing

        except asyncio.CancelledError:
            print(f"Chunk {chunk_index + 1} processing was cancelled.")
            await queue.put(
                {"cancelled_chunk": chunk_index}
            )  # Optionally signal cancellation
            # Re-raise the cancellation error so gather knows
            raise
        except Exception as e:
            print(f"Error processing chunk {chunk_index + 1}: {e}")
            # Log error details if possible (e.g., using traceback module)
            await queue.put({"error": {"chunk": chunk_index, "error": str(e)}})
        finally:
            # Ensure task is removed from active set upon completion or cancellation
            active_tasks.discard(asyncio.current_task())
            print(
                f"Chunk {chunk_index + 1} finished. (# Tasks: {len(active_tasks)})"
            )  # Add active task count


async def process_task(task_name, current_args, async_chat_fn):
    """Processes a single data generation task."""
    global active_tasks  # Modify global set
    print(f"\n{'='*50}")
    print(f"Processing task: {task_name}")
    print(f"{'='*50}\n")

    # Determine file paths for this task
    output_file = Path(current_args.output_dir) / f"{task_name}.jsonl"
    # Determine source name for progress file (handle potential None)
    source_id = "unknown_source"
    if current_args.dataset_names:
        source_id = current_args.dataset_names[0].replace("/", "_")
    elif current_args.pdf_dir:
        source_id = Path(current_args.pdf_dir).name
    progress_file = (
        Path(current_args.cache_dir) / f"{task_name}_{source_id}_progress.jsonl"
    )

    # Ensure directories exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load agent configs using relative import
        content_agents, instruction_agents, one_shot_example = load_agent_configs(
            current_args.agent_config_path, task_name
        )
    except FileNotFoundError as e:
        print(
            f"Error loading agent configurations for task '{task_name}' from '{current_args.agent_config_path}': {e}"
        )
        print("Please ensure the agent config path is correct and JSON files exist.")
        return  # Skip this task
    except Exception as e:
        print(
            f"Unexpected error loading agent configurations for task '{task_name}': {e}"
        )
        return  # Skip this task

    # Get text chunks for this task
    text_chunks = get_text_chunks(current_args)
    if not text_chunks:
        print(f"No text chunks found for task '{task_name}'. Skipping.")
        return

    source_type = "dataset(s)" if current_args.dataset_names else "PDF directory"
    print(
        f"Extracted {len(text_chunks)} text chunks from the {source_type} for task '{task_name}'."
    )

    # Create list of (original_index, chunk_data) pairs
    chunks_with_indices = list(enumerate(text_chunks))

    # Shuffle if requested
    if current_args.random:
        task_specific_seed = random.randint(0, 10000) + hash(task_name) % 10000
        task_rng = random.Random(task_specific_seed)
        task_rng.shuffle(chunks_with_indices)
        print(f"Shuffled chunks for task {task_name} with seed {task_specific_seed}.")

    # Apply debug/max limits
    if current_args.debug:
        chunks_with_indices = chunks_with_indices[:10]
        print("Debug mode: Processing only first 10 chunks.")
    elif current_args.max_chunks > 0:
        chunks_with_indices = chunks_with_indices[: current_args.max_chunks]
        print(f"Processing a maximum of {current_args.max_chunks} chunks.")

    # Load processed chunks from progress file
    processed_indices = set()
    if progress_file.exists():
        try:
            with open(progress_file, "r") as f:
                for line in f:
                    try:
                        index = int(line.strip())
                        processed_indices.add(index)
                    except ValueError:
                        print(
                            f"Warning: Skipping invalid line in progress file {progress_file}: {line.strip()}"
                        )
            print(
                f"Resuming task '{task_name}'. Found {len(processed_indices)} already processed chunks."
            )
        except Exception as e:
            print(f"Warning: Could not read progress file {progress_file}: {e}")

    tasks_to_create = [
        (index, chunk)
        for index, chunk in chunks_with_indices
        if index not in processed_indices
    ]

    if not tasks_to_create:
        print(f"All required chunks for task '{task_name}' are already processed.")
        return  # No tasks needed for this run
    else:
        print(f"Creating {len(tasks_to_create)} processing tasks for '{task_name}'.")

    semaphore = asyncio.Semaphore(current_args.concurrency)
    queue = asyncio.Queue()

    # Define the writer task specific to this process_task call
    async def task_writer(q, out_f_path, prog_f_path):
        processed_count = 0
        error_count = 0
        empty_count = 0
        skipped_count = 0
        cancelled_count = 0  # Track cancellations

        try:
            # Open files in append mode
            with open(out_f_path, "a") as data_f, open(prog_f_path, "a") as progress_f:
                while True:
                    item = await q.get()
                    if item is None:  # Sentinel value to stop
                        break

                    if isinstance(item, tuple):
                        _, generated_pairs = item
                        for pair_data in generated_pairs:
                            # Write the final pair structure
                            metadata = {
                                k: v
                                for k, v in pair_data.items()
                                if k not in ["instruction", "answer"]
                            }
                            chat_data = {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": pair_data.get("instruction", ""),
                                    },
                                    {
                                        "role": "assistant",
                                        "content": pair_data.get("answer", ""),
                                    },
                                ],
                                "metadata": metadata,
                            }
                            data_f.write(json.dumps(chat_data) + "\n")
                        data_f.flush()  # Flush after writing pairs for a chunk
                    elif isinstance(item, dict):
                        if "processed_chunk" in item:
                            chunk_index = item["processed_chunk"]
                            progress_f.write(f"{chunk_index}\n")
                            progress_f.flush()  # Flush after writing progress
                            processed_count += 1
                        elif "error" in item:
                            print(
                                f"Logged error for chunk {item['error']['chunk'] + 1}: {item['error']['error']}"
                            )
                            error_count += 1
                            # Optionally write errors to a separate log file
                        elif "empty_chunk" in item:
                            empty_count += 1
                        elif "skipped_chunk" in item:
                            skipped_count += 1
                        elif "cancelled_chunk" in item:  # Handle cancellation signal
                            cancelled_count += 1

                    q.task_done()  # Mark task as done in the queue
        except asyncio.CancelledError:
            print(f"Writer task for '{task_name}' was cancelled.")
            raise  # Propagate cancellation
        except Exception as e:
            print(f"FATAL ERROR in task_writer for task '{task_name}': {e}")
            # Handle writer error (e.g., log, raise)
        finally:
            print(
                f"Writer finished for task '{task_name}'. Summary: Processed={processed_count}, Errors={error_count}, Empty={empty_count}, Skipped={skipped_count}, Cancelled={cancelled_count}"
            )

    # Create the writer task
    writer_task = asyncio.create_task(task_writer(queue, output_file, progress_file))
    active_tasks.add(writer_task)  # Track writer task

    # Create processing tasks
    processing_tasks = []
    for index, chunk_data in tasks_to_create:
        task = asyncio.create_task(
            process_chunk(
                index,  # Pass original index
                chunk_data,
                content_agents,
                instruction_agents,
                one_shot_example,
                current_args,  # Pass args
                semaphore,
                queue,
                async_chat_fn,  # Pass partial chat function
            )
        )
        processing_tasks.append(task)
        active_tasks.add(task)  # Add processing task to global set

    # Wait for all processing tasks to complete or be cancelled
    try:
        await asyncio.gather(
            *processing_tasks, return_exceptions=False
        )  # Don't return exceptions here, they are handled in process_chunk
    except asyncio.CancelledError:
        print(f"Processing tasks for {task_name} gather loop cancelled.")
        # The shutdown handler already cancelled individual tasks
    finally:
        # Signal writer to finish after all tasks are done or cancelled
        await queue.put(None)
        # Wait for writer task to finish handling remaining queue items or be cancelled
        try:
            await writer_task
        except asyncio.CancelledError:
            print(
                f"Writer task for {task_name} did not finish cleanly after cancellation signal."
            )
        finally:
            active_tasks.discard(writer_task)  # Remove writer task from active set

    print(f"Task '{task_name}' processing finished.")


async def main_async(current_args):
    """Asynchronous part of the main execution."""
    global args, active_tasks  # Allow modification of global args and task set
    args = current_args

    # Resolve agent config path
    if args.agent_config_path is None:
        args.agent_config_path = get_default_agent_config_path()
        print(f"Using bundled agent configurations from: {args.agent_config_path}")
    else:
        print(f"Using custom agent configurations from: {args.agent_config_path}")
        if not Path(args.agent_config_path).is_dir():
            print(
                f"Error: Custom agent config path is not a valid directory: {args.agent_config_path}"
            )
            sys.exit(1)

    # Check output and cache directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    # Create the partial function for chat completion with the specified model
    async_chat_fn = partial(async_chat_completion, model=args.model)

    # --- Signal Handling ---
    loop = asyncio.get_running_loop()
    shutdown_requested = asyncio.Event()  # Event to signal shutdown

    async def shutdown_handler(sig):
        if shutdown_requested.is_set():
            print("Shutdown already requested. Force exiting...")
            # Optionally add a small delay then force exit if tasks don't stop
            # await asyncio.sleep(5)
            # os._exit(1) # Force exit if needed
            return

        print(f"\nReceived signal {sig}. Initiating graceful shutdown...")
        shutdown_requested.set()  # Signal that shutdown is in progress

        # Cancel all tracked active tasks (processing and writer)
        cancelled_count = 0
        tasks_to_cancel = list(
            active_tasks
        )  # Copy set to avoid modification during iteration
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
                cancelled_count += 1

        print(
            f"Cancellation requests sent to {cancelled_count} active tasks. Waiting for completion..."
        )
        # Allow the main loop to handle cancellations and cleanup

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            # Use lambda to prevent the handler from receiving the loop argument
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(shutdown_handler(s))
            )
        except NotImplementedError:
            # Windows might not support add_signal_handler
            print(
                f"Warning: Signal handling for {sig} not fully supported on this platform."
            )
            # Consider alternative shutdown mechanisms if needed

    # --- Task Processing ---
    main_processing_task = None
    try:
        # Wrap the main task processing in a single task to manage cancellation
        async def run_all_tasks():
            if args.all_tasks:
                # Discover task names from agent config directory
                instruction_dir = (
                    Path(args.agent_config_path) / "instruction_generation"
                )
                if not instruction_dir.is_dir():
                    print(
                        f"Error: Instruction generation config directory not found: {instruction_dir}"
                    )
                    return  # Exit this async function

                task_names = []
                for filename in os.listdir(instruction_dir):
                    if filename.endswith("_instruction.json"):
                        task_name = filename.replace("_instruction.json", "")
                        task_names.append(task_name)

                if not task_names:
                    print(
                        f"Error: No task configuration files (*_instruction.json) found in {instruction_dir}"
                    )
                    return

                print(f"Found {len(task_names)} tasks: {', '.join(task_names)}")

                # Process tasks sequentially
                for task_name in task_names:
                    if shutdown_requested.is_set():
                        print(f"Skipping task {task_name} due to shutdown request.")
                        break
                    await process_task(task_name, args, async_chat_fn)

                print("\nAll tasks processing completed or were interrupted.")
            else:
                # Process a single specified task
                if not shutdown_requested.is_set():
                    await process_task(args.task_name, args, async_chat_fn)
                    print("\nSingle task processing completed or was interrupted.")
                else:
                    print(f"Skipping task {args.task_name} due to shutdown request.")

        # Start the main processing task
        main_processing_task = asyncio.create_task(run_all_tasks())
        active_tasks.add(main_processing_task)  # Track main task itself
        await main_processing_task  # Wait for it to complete or be cancelled

    except asyncio.CancelledError:
        print("Main execution task was cancelled.")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
        # Log traceback here if possible
    finally:
        # Ensure main task is removed from active set
        if main_processing_task:
            active_tasks.discard(main_processing_task)

        # Wait briefly for remaining tasks to finish cancelling if shutdown was requested
        if shutdown_requested.is_set():
            print("Allowing a moment for tasks to finalize cancellation...")
            # Give cancelled tasks a chance to finish their finally blocks
            await asyncio.sleep(1)  # Adjust sleep time if needed

            # Check if any tasks are still somehow running (shouldn't happen often)
            still_running = [t for t in active_tasks if not t.done()]
            if still_running:
                print(
                    f"Warning: {len(still_running)} tasks did not complete shutdown cleanly."
                )

        # Remove signal handlers AFTER the main task processing logic
        print("Removing signal handlers...")
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.remove_signal_handler(sig)
            except (
                NotImplementedError,
                ValueError,
            ):  # ValueError if handler not registered
                pass  # Ignore if not supported or already removed
        print("Cleanup finished. Exiting.")


def main_wrapper():
    """Entry point for console script.
    Parses args and runs the async main function.
    """
    parser = setup_parser()
    cli_args = parser.parse_args()

    # Basic validation
    if not cli_args.dataset_names and not cli_args.pdf_dir:
        parser.error("Either --dataset-names or --pdf-dir must be specified.")

    # Run the async main function
    try:
        # Use the default event loop runner which handles KeyboardInterrupt better
        asyncio.run(main_async(cli_args), debug=False)  # Set debug=False for production
    except KeyboardInterrupt:
        # This might still catch if the signal handlers don't fully work
        # or if the interrupt happens before the loop starts properly.
        print(
            "\nInterrupted by user (KeyboardInterrupt) before or during loop start. Exiting forcefully."
        )
        sys.exit(1)
    except asyncio.CancelledError:
        # This can happen if the main_async task itself gets cancelled externally
        print("\nMain async task cancelled externally. Exiting.")
        sys.exit(1)


# This block is only executed when the script is run directly (e.g., python open_agentinstruct/cli.py)
# It won't run when imported or called via the entry point.
if __name__ == "__main__":
    main_wrapper()
