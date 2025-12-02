from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import os

from .services.pdf_reader import extract_content_from_pdf
from .services.summarizer import summarize_pdf
from .models.responses import SummaryResponse

app = FastAPI(title="AI PDF Summarizer - LangChain + Gemini")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse({"error": "Hanya file PDF yang diperbolehkan"}, status_code=400)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        contents = extract_content_from_pdf(file_path)
        image_count = sum(1 for c in contents if c["type"] == "image")

        summary = await summarize_pdf(contents)

        os.remove(file_path) 

        return JSONResponse({
            "summary": summary,
            "page_count": len([c for c in contents if c["type"] == "text"]),
            "processed_images": image_count
        })

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({"error": str(e)}, status_code=500)
