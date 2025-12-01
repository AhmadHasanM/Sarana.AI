# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Sarana.AI – Chat dengan PDF", layout="centered")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PG_URI = os.getenv("PG_URI")

if not GEMINI_API_KEY or not PG_URI:
    st.error("GEMINI_API_KEY atau PG_URI tidak ditemukan di .env")
    st.stop()

COLLECTION_NAME = "streamlit_pdf_collection"
LLM_MODEL = "gemini-2.0-flash"

# ------------------- FUNGSI -------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    # Simpan temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Load & chunk
    loader = PyPDFLoader(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(loader.load())

    # Vector store
    embeddings = get_embeddings()
    db = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=PG_URI,
        pre_delete_collection=True
    )
    os.unlink(tmp_path)  # hapus file temp
    return db

def get_qa_chain(db):
    llm = GoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY, temperature=0.3)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 6})
    )

# ------------------- UI -------------------
st.title("Sarana.AI – Chat dengan PDF Kamu")
st.markdown("Upload PDF → otomatis dirangkum → tanya apa saja sepuasnya!")

uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")

if uploaded_file:
    with st.spinner("Membaca & mengolah PDF..."):
        db = process_pdf(uploaded_file)
        qa = get_qa_chain(db)
        st.success(f"PDF **{uploaded_file.name}** berhasil di-load! ({uploaded_file.size//1024} KB)")

    # Auto summary
    with st.spinner("Membuat ringkasan otomatis..."):
        summary_prompt = """Kamu adalah asisten peneliti senior yang teliti dan berpengalaman.
        Buat ringkasan yang sangat lengkap, terstruktur, dan enak dibaca dalam bahasa Indonesia yang natural.
        Gunakan struktur:
        1. Judul & info umum
        2. Latar belakang
        3. Tujuan
        4. Metode singkat
        5. Temuan utama (bullet)
        6. Kesimpulan
        7. Rekomendasi
        8. Take-home message dalam 1 kalimat.

        Tulis seperti sedang menjelaskan ke kolega yang cerdas tapi sibuk."""
        summary = qa.run(summary_prompt)
        st.subheader("Ringkasan Otomatis")
        st.write(summary)

    st.markdown("---")
    st.subheader("Tanya Apa Saja tentang PDF Ini")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ketik pertanyaanmu di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                answer = qa.run(prompt)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Silakan upload PDF untuk memulai")
    st.markdown(
        """
        **Fitur:**
        - Upload PDF (tanpa batas ukuran wajar)
        - Ringkasan otomatis berkualitas tinggi
        - Chat interaktif sepuasnya
        - Embedding lokal (gratis selamanya)
        - Data disimpan sementara di PostgreSQL
        """
    )