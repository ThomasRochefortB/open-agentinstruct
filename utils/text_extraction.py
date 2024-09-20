import re
import pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    import pdfminer.high_level
    text = pdfminer.high_level.extract_text(pdf_path)
    # Optional: Clean up the text
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_text_chunks_from_pdf(pdf_path, pages_per_chunk=5):
    """
    Extracts text from a PDF file and splits it into chunks.
    
    Args:
        pdf_path (str): Path to the PDF file.
        pages_per_chunk (int): Number of pages to include in each chunk.
        
    Returns:
        List[str]: A list of text chunks.
    """
   
    
    resource_manager = PDFResourceManager()
    laparams = LAParams()
    text_chunks = []
    total_pages = 0
    current_chunk_pages = []
    current_page_number = 0
    
    with open(pdf_path, 'rb') as file:
        for page in PDFPage.get_pages(file):
            current_page_number += 1
            output_string = StringIO()
            device = TextConverter(resource_manager, output_string, laparams=laparams)
            interpreter = PDFPageInterpreter(resource_manager, device)
            interpreter.process_page(page)
            page_text = output_string.getvalue()
            device.close()
            output_string.close()
            
            current_chunk_pages.append(page_text)
            
            if current_page_number % pages_per_chunk == 0:
                chunk_text = '\n'.join(current_chunk_pages)
                # Optional: Clean up the text
                chunk_text = re.sub(r'\s+', ' ', chunk_text)
                text_chunks.append(chunk_text)
                current_chunk_pages = []
        
        # Add any remaining pages as a chunk
        if current_chunk_pages:
            chunk_text = '\n'.join(current_chunk_pages)
            chunk_text = re.sub(r'\s+', ' ', chunk_text)
            text_chunks.append(chunk_text)
    
    return text_chunks