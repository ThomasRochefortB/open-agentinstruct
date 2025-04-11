import re
from io import StringIO
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from datasets import load_dataset
from typing import Dict, Optional

# Configure logging - Assuming logging is configured elsewhere or remove if not needed
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


def extract_text_chunks_from_pdf(pdf_path, pages=None):
    """
    Extracts text from a PDF file using pdfminer and splits it into chunks (one page per chunk),
    with optional page range.

    Args:
        pdf_path (str): Path to the PDF file.
        pages (list of int, optional): A list of page numbers to extract. Pages are 1-indexed.

    Returns:
        List[str]: A list of text chunks (one per page).
    """
    text_chunks = []
    resource_manager = PDFResourceManager()
    laparams = LAParams()

    try:
        with open(pdf_path, "rb") as file:
            for page_number, page in enumerate(PDFPage.get_pages(file), start=1):
                # Skip pages not in the specified list
                if pages and page_number not in pages:
                    continue

                output_string = StringIO()
                device = TextConverter(
                    resource_manager, output_string, laparams=laparams
                )
                interpreter = PDFPageInterpreter(resource_manager, device)
                interpreter.process_page(page)
                page_text = output_string.getvalue()
                device.close()
                output_string.close()

                # Clean up the text and add if not empty
                page_text = re.sub(r"\s+", " ", page_text).strip()
                if page_text:
                    text_chunks.append(page_text)
    except Exception as e:
        # Consider logging the error instead of just printing
        print(f"Error processing PDF {pdf_path} with pdfminer: {e}")
        # Depending on desired behavior, you might want to re-raise the exception
        # or return an empty list / partial results

    return text_chunks


def parse_instruction_answer_pairs(text):
    """
    Parses text to extract instruction and answer pairs, handling multiple pairs and various formats.
    """
    # Remove any markdown formatting or unnecessary characters
    text = re.sub(r"[*_]{1,2}", "", text).strip()

    # Remove code block markers if present
    text = re.sub(r"```[^\n]*\n?", "", text).strip()

    # Split text into potential pairs
    pairs = []

    # Find all instruction-answer blocks
    blocks = re.split(r"(?=Instruction:)", text)

    for block in blocks:
        if not block.strip():
            continue

        # Match instruction and answer within each block
        instruction_match = re.search(
            r"Instruction:\s*(.*?)(?=Answer:|$)", block, re.DOTALL | re.IGNORECASE
        )
        answer_match = re.search(
            r"Answer:\s*(.*?)(?=Instruction:|$)", block, re.DOTALL | re.IGNORECASE
        )

        if instruction_match and answer_match:
            instruction = instruction_match.group(1).strip()
            answer = answer_match.group(1).strip()

            # Validate the pair
            if instruction and answer:
                pairs.append({"instruction": instruction, "answer": answer})
        else:
            print(f"Failed to parse block:\n{block}")

    if not pairs:
        print("No valid instruction-answer pairs found in:\n{text[:200]}...")

    return pairs


def parse_modified_output(text: str) -> Optional[Dict]:
    """
    Parses modified instruction and answer from the text with specific prefixes,
    but removes these prefixes from the final output.

    Args:
        text (str): The text containing modified instruction and answer

    Returns:
        Optional[Dict]: Dictionary with 'instruction' and 'answer' keys, or None if parsing fails
    """
    try:
        # Remove any markdown formatting and normalize whitespace
        text = re.sub(r"[*_]{1,2}", "", text).strip()
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n+", "\n", text).strip()

        # Patterns to match modified instruction and answer sections
        instruction_pattern = r"Modified Instruction:\s*(.*?)(?=Modified Answer:|$)"
        answer_pattern = r"Modified Answer:\s*(.*?)(?=Modified Instruction:|$)"

        instruction_match = re.search(instruction_pattern, text, re.DOTALL)
        answer_match = re.search(answer_pattern, text, re.DOTALL)

        if instruction_match and answer_match:
            instruction = instruction_match.group(1).strip()
            answer = answer_match.group(1).strip()

            # Verify that we have actual content
            if instruction and answer:
                return {"instruction": instruction, "answer": answer}

        # If the first attempt fails, try to parse as multiple choice question
        question_pattern = r"Modified Instruction:\s*(.*?)(?:Choices:|(?=\s*[A-D]\)))"
        choices_pattern = r"(?:Choices:\s*|\n\s*)((?:[A-D]\).*?\n?)+)"

        question_match = re.search(question_pattern, text, re.DOTALL)
        choices_match = re.search(choices_pattern, text, re.DOTALL)

        if question_match and choices_match:
            instruction = question_match.group(1).strip()
            choices = choices_match.group(1).strip()

            if instruction and choices:
                return {
                    "instruction": f"{instruction}\nChoices:\n{choices}",
                    "answer": choices,  # or extract specific answer if indicated
                }

        print("Failed to parse modified instruction and answer sections.")
        return None

    except Exception as e:
        print(f"Error parsing modified output: {str(e)}")
        return None


def extract_text_chunks_from_dataset(
    dataset_name,
    split="train",
    text_field="text",
    chunk_size=1000,
    dataset_kwargs={},
    use_samples=False,
    min_chars=None,
):
    """
    Loads a HuggingFace dataset and extracts text chunks or individual samples.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Which split to use ('train', 'validation', 'test'). Default is 'train'.
        text_field (str): The field containing the text. Default is 'text'.
        chunk_size (int): The number of characters per chunk when not using samples. Default is 1000.
        dataset_kwargs (dict): Additional keyword arguments to pass to `load_dataset`.
        use_samples (bool): If True, return individual text samples from the dataset. If False, concatenate text and split into chunks.
        min_chars (int, optional): If specified and use_samples is True, concatenate samples until they reach this minimum character count.

    Returns:
        list: A list of text chunks or individual samples.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split, **dataset_kwargs)

    if use_samples:
        # Return individual text samples
        text_contents = [item[text_field] for item in dataset if text_field in item]

        # If min_chars is specified, combine samples until they meet the minimum length
        if min_chars and min_chars > 0:
            print(
                f"Using minimum character count: {min_chars} for dataset {dataset_name}"
            )
            combined_chunks = []
            current_chunk = ""

            for text in text_contents:
                if len(current_chunk) >= min_chars:
                    combined_chunks.append(current_chunk)
                    current_chunk = text
                else:
                    # Add a space between concatenated texts
                    if current_chunk:
                        current_chunk += " " + text
                    else:
                        current_chunk = text

            # Add the last chunk if it's not empty
            if current_chunk:
                combined_chunks.append(current_chunk)

            return combined_chunks

        return text_contents
    else:
        # Concatenate all the text fields into one large string
        all_text = " ".join(
            [item[text_field] for item in dataset if text_field in item]
        )

        # Split the text into chunks
        text_chunks = [
            all_text[i : i + chunk_size] for i in range(0, len(all_text), chunk_size)
        ]
        return text_chunks
