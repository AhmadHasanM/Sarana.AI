import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import tempfile

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Sarana.AI – Chat dengan PDF", layout="centered")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PG_URI = os.getenv("PG_URI")

if not GEMINI_API_KEY or not PG_URI:
    st.error("GEMINI_API_KEY atau PG_URI tidak ditemukan di .env")
    st.stop()

COLLECTION_NAME = "streamlit_pdf_collection"
LLM_MODEL = "gemini-2.0-flash"

# ------------------------------------------------
# FUNGSI
# ------------------------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(uploaded_file):
    # Simpan ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Load dan pecah per bagian
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Vectorstore
    embeddings = get_embeddings()

    db = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=PG_URI,
        pre_delete_collection=True
    )

    os.unlink(tmp_path)
    return db

def get_qa_chain(db):
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 6})
    )

# ------------------------------------------------
# UI
# ------------------------------------------------
st.title("Sarana.AI – Chat dengan PDF Kamu")
st.write("Upload PDF lalu dapatkan ringkasan otomatis dan bisa tanya jawab.")

uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")

if uploaded_file:

    with st.spinner("Memproses PDF..."):
        db = process_pdf(uploaded_file)
        qa = get_qa_chain(db)

    st.success(f"PDF {uploaded_file.name} berhasil diproses.")

    # Ringkasan otomatis
    with st.spinner("Membuat ringkasan..."):
        summary_prompt = """
        Buat ringkasan lengkap, terstruktur, dan jelas berdasarkan isi dokumen.
        Format:
        1. Judul dan informasi umum
        2. Latar belakang
        3. Tujuan
        4. Metode
        5. Temuan utama
        6. Kesimpulan
        7. Rekomendasi
        8. Ringkasan 1 kalimat
        """
        summary = qa.run(summary_prompt)

    st.subheader("Ringkasan Otomatis")
    st.write(summary)

    st.markdown("---")
    st.subheader("Tanya Apa Saja Mengenai PDF Ini")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Tulis pertanyaanmu di sini.")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                answer = qa.run(prompt)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload PDF untuk mulai.")
