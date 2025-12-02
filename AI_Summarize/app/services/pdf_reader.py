
from pypdf import PdfReader
from typing import List, Dict
import io
import os


try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except Exception:
    FITZ_AVAILABLE = False

def extract_content_from_pdf(pdf_path: str) -> List[Dict]:
    contents = []
    reader = PdfReader(pdf_path)

    
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            contents.append({"type": "text", "content": text.strip()})

    
    if FITZ_AVAILABLE and os.path.getsize(pdf_path) < 30 * 1024 * 1024:  
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(len(doc), 20)): 
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=180)
                img_data = pix.tobytes("png")
                from PIL import Image
                img = Image.open(io.BytesIO(img_data))
                contents.append({"type": "image", "image": img})
            doc.close()
        except Exception as e:
            print(f"Gambar gagal diekstrak: {e}")  

    return contents
