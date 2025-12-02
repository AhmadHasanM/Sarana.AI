import google.generativeai as genai
import os
from dotenv import load_dotenv
from app.services.prompt_builder import build_prompt

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")

genai.configure(api_key=API_KEY)


def run_summarization_with_pdf(pdf_bytes: bytes):
    # Ambil prompt dari prompt builder
    msgs = build_prompt("Berikut PDF yang ingin dianalisis:")

    system_prompt = msgs[0].content
    human_prompt = msgs[1].content

    # Tambahkan konfigurasi model
    generation_config = {
        "temperature": 0.1,   # Lebih konsisten
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 2048
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config
    )

    response = model.generate_content(
        [
            {"text": system_prompt},
            {"text": human_prompt},
            {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": pdf_bytes
                }
            }
        ],
        safety_settings="BLOCK_NONE"
    )

    return response.text
