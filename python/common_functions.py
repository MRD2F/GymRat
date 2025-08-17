from dotenv import load_dotenv
import os
def get_private_keys(key_name):
    load_dotenv()
    key = os.getenv(key_name)
    if not key:
        raise ValueError(f"Missing key for {key_name} in .env") 
    return key

from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter

def extract_pages(input_pdf, output_pdf, start_page, end_page):
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    for page_num in range(start_page-1, end_page):  # PyPDF2 is 0-indexed
        writer.add_page(reader.pages[page_num])
    with open(output_pdf, "wb") as f:
        writer.write(f)

# # Paths
# file_path = Path("../data/books/rebuilding_milo.pdf")
# temp_path = Path("../data/books/rebuilding_milo_subset_p90to140.pdf")

# # Extract only pages 2–5
# extract_pages(file_path, temp_path, 90, 140)