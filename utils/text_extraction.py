import re
from io import StringIO
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import pymupdf4llm
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from datasets import load_dataset
import re


def extract_text_chunks_from_pdf(
    pdf_path, engine="pdfminer", use_images=False, pages=None
):
    """
    Extracts text from a PDF file and splits it into chunks (one page per chunk), with optional page range.

    Args:
        pdf_path (str): Path to the PDF file.
        engine (str): The engine to use for extraction. Options: 'pdfminer', 'pymupdf', 'unstructured'.
        use_images (bool): Whether to include images in markdown extraction for PyMuPDF.
        pages (list of int, optional): A list of page numbers to extract. Pages are 1-indexed.

    Returns:
        List[str]: A list of text chunks (one per page or chunk).
    """
    text_chunks = []

    if engine == "pdfminer":
        # PDFMiner extraction
        resource_manager = PDFResourceManager()
        laparams = LAParams()

        with open(pdf_path, "rb") as file:
            for page_number, page in enumerate(PDFPage.get_pages(file), start=1):
                # Skip pages not in the range
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

                # Clean up the text (optional)
                page_text = re.sub(r"\s+", " ", page_text)
                text_chunks.append(page_text)

    elif engine == "pymupdf":
        # PyMuPDF extraction
        if pages is None:
            pages = []  # Extract all pages if no specific pages are mentioned

        markdown_chunks = pymupdf4llm.to_markdown(
            pdf_path, page_chunks=True, write_images=use_images, pages=pages
        )

        for page_dict in markdown_chunks:
            # Extract the 'text' key which contains the main page content
            page_text = page_dict.get("text", "")
            page_text = re.sub(r"\s+", " ", page_text)  # Optional: Clean up the text
            text_chunks.append(page_text)

    elif engine == "unstructured":
        # Unstructured extraction
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # Use high-resolution strategy
            extract_images_in_pdf=False,
            extract_image_block_types=[
                "Image",
                "Table",
            ],  # Optional: Include images and tables
            extract_image_block_to_payload=False,
            extract_image_block_output_dir="./",  # Optional: Specify where to save extracted images
        )

        # Chunking the extracted elements by title (uses a heuristic to create chunks based on document structure)
        chunks = chunk_by_title(elements)

        # Combine the chunk content into text format
        for chunk in chunks:
            chunk_text = str(chunk)  # Convert chunk content to string
            chunk_text = re.sub(r"\s+", " ", chunk_text)  # Clean up the text
            text_chunks.append(chunk_text)

    else:
        raise ValueError(
            "Unsupported engine. Choose either 'pdfminer', 'pymupdf', or 'unstructured'."
        )

    return text_chunks


def parse_instruction_answer_pairs(text):
    lines = text.strip().split("\n")
    instruction = ""
    answer = ""
    capturing_instruction = False
    capturing_answer = False

    for line in lines:
        line = line.strip()
        if line.startswith("Instruction:") or line.startswith("Refined Instruction:"):
            capturing_instruction = True
            capturing_answer = False
            # Remove both possible prefixes
            instruction = line.split(":", 1)[1].strip()
        elif line.startswith("Answer:") or line.startswith("Refined Answer:"):
            capturing_instruction = False
            capturing_answer = True
            answer = line.split(":", 1)[1].strip()
        else:
            if capturing_instruction:
                instruction += " " + line.strip()
            elif capturing_answer:
                answer += " " + line.strip()

    if instruction and answer:
        return [{"instruction": instruction.strip(), "answer": answer.strip()}]
    else:
        return []


def parse_modified_triple(text):
    # Remove any markdown formatting (e.g., **, __)
    text = re.sub(r"[*_]{1,2}", "", text)

    # Normalize the text by removing extra whitespace
    text = re.sub(r"\r\n", "\n", text)  # Replace carriage returns
    text = re.sub(r"\n+", "\n", text).strip()  # Remove extra newlines

    # Use regular expressions to match labels with optional whitespace
    pattern = r"Modified Passage:\s*(.*?)\nModified Question:\s*(.*?)\nModified Answer:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        modified_passage = match.group(1).strip()
        modified_question = match.group(2).strip()
        modified_answer = match.group(3).strip()
        return {
            "instruction": modified_question,
            "answer": modified_answer,
            "context": modified_passage,
        }
    else:
        print("Failed to parse using regex.")
        return None


def extract_text_chunks_from_dataset(
    dataset_name,
    split="train",
    text_field="text",
    chunk_size=1000,
    dataset_kwargs={},
    use_samples=False,
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

    Returns:
        list: A list of text chunks or individual samples.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split, **dataset_kwargs)

    if use_samples:
        # Return individual text samples
        text_contents = [item[text_field] for item in dataset if text_field in item]
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
