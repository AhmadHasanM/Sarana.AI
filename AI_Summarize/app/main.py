from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.services.summarizer import run_summarization_with_pdf
import shutil
import tempfile
import os

app = FastAPI(
    title="PDF Summarizer",
    description="Upload PDF lalu sistem akan menganalisis isi (teks + gambar + tabel).",
    version="2.0.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
def homepage():
    template_path = "app/templates/index.html"

    if not os.path.exists(template_path):
        raise HTTPException(500, detail="Template index.html tidak ditemukan.")

    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File harus berformat PDF.")

    try:
        # Simpan PDF ke file temporary
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            pdf_path = tmp.name

        # Baca PDF sebagai bytes
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Summarize langsung dari PDF (multimodal)
        summary = run_summarization_with_pdf(pdf_bytes)

        return {
            "filename": file.filename,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses PDF: {str(e)}"
        )

    finally:
        # Hapus file temp
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except:
            pass
