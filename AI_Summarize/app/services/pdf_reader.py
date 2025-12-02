# app/services/pdf_reader.py
# VERSI FINAL – HANYA TEKS, 100% JALAN DI KOMPUTER KAMU SEKARANG

from pypdf import PdfReader
from typing import List, Dict

def extract_content_from_pdf(pdf_path: str) -> List[Dict]:
    reader = PdfReader(pdf_path)
    contents = []
    
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            contents.append({
                "type": "text",
                "content": text.strip()
            })
    
    return contents