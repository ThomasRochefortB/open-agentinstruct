import re
from io import StringIO
from pdfminer.high_level import extract_text
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import pymupdf4llm

import re
from io import StringIO
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import pymupdf4llm

def extract_text_chunks_from_pdf(pdf_path, engine="pdfminer", use_images=False, pages=None):
    """
    Extracts text from a PDF file and splits it into chunks (one page per chunk), with optional page range.
    
    Args:
        pdf_path (str): Path to the PDF file.
        engine (str): The engine to use for extraction. Options: 'pdfminer', 'pymupdf'.
        use_images (bool): Whether to include images in markdown extraction for PyMuPDF.
        pages (list of int, optional): A list of page numbers to extract. Pages are 1-indexed.
        
    Returns:
        List[str]: A list of text chunks (one per page).
    """
    text_chunks = []
    
    if engine == "pdfminer":
        # PDFMiner extraction
        resource_manager = PDFResourceManager()
        laparams = LAParams()

        with open(pdf_path, 'rb') as file:
            for page_number, page in enumerate(PDFPage.get_pages(file), start=1):
                # Skip pages not in the range
                if pages and page_number not in pages:
                    continue

                output_string = StringIO()
                device = TextConverter(resource_manager, output_string, laparams=laparams)
                interpreter = PDFPageInterpreter(resource_manager, device)
                interpreter.process_page(page)
                page_text = output_string.getvalue()
                device.close()
                output_string.close()

                # Clean up the text (optional)
                page_text = re.sub(r'\s+', ' ', page_text)
                text_chunks.append(page_text)

    elif engine == "pymupdf":
        # PyMuPDF extraction
        if pages is None:
            pages = []  # Extract all pages if no specific pages are mentioned

        markdown_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, write_images=use_images, pages=pages)

        for page_dict in markdown_chunks:
            # Extract markdown text from the current page
            page_text = page_dict.get('text', '')
            page_text = re.sub(r'\s+', ' ', page_text)  # Optional: Clean up the text
            text_chunks.append(page_text)

    else:
        raise ValueError("Unsupported engine. Choose either 'pdfminer' or 'pymupdf'.")

    return text_chunks




def parse_instruction_answer_pairs(text):
    pairs = []
    lines = text.strip().split('\n')
    instruction = ''
    answer = ''
    capturing_instruction = False
    capturing_answer = False

    for line in lines:
        if line.strip().startswith('Instruction:'):
            capturing_instruction = True
            capturing_answer = False
            instruction = line.replace('Instruction:', '').strip()
        elif line.strip().startswith('Answer:'):
            capturing_instruction = False
            capturing_answer = True
            answer = line.replace('Answer:', '').strip()
        else:
            if capturing_instruction:
                instruction += ' ' + line.strip()
            elif capturing_answer:
                answer += ' ' + line.strip()

        if instruction and answer:
            pairs.append({'instruction': instruction.strip(), 'answer': answer.strip()})
            instruction = ''
            answer = ''
            capturing_instruction = False
            capturing_answer = False

    return pairs


def parse_refined_instruction_answer(text):
    refined_instruction = ''
    refined_answer = ''
    lines = text.strip().split('\n')
    capturing_instruction = False
    capturing_answer = False

    for line in lines:
        if line.strip().startswith('Refined Instruction:'):
            capturing_instruction = True
            capturing_answer = False
            refined_instruction = line.replace('Refined Instruction:', '').strip()
        elif line.strip().startswith('Refined Answer:'):
            capturing_instruction = False
            capturing_answer = True
            refined_answer = line.replace('Refined Answer:', '').strip()
        else:
            if capturing_instruction:
                refined_instruction += ' ' + line.strip()
            elif capturing_answer:
                refined_answer += ' ' + line.strip()

    if refined_instruction and refined_answer:
        return {'instruction': refined_instruction.strip(), 'answer': refined_answer.strip()}
    else:
        return None
