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

    system_prompt, human_prompt = build_prompt()

    generation_config = {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 4096
    }

    # SANGAT disarankan untuk PDF: gemini-1.5-flash
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=generation_config
    )

    response = model.generate_content(
        [
            {"text": system_prompt.content},
            {"text": human_prompt.content},
            {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": pdf_bytes
                }
            }
        ],
        safety_settings="BLOCK_NONE"
    )

    # Antisipasi finish_reason=2
    if not response.candidates or not response.candidates[0].content:
        raise ValueError(
            "Model tidak menghasilkan output. finish_reason="
            + str(response.candidates[0].finish_reason)
        )

    return response.text
