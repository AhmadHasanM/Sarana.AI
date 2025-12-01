from dotenv import load_dotenv
load_dotenv()

import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------- CONFIG -------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di .env")

PG_URI = os.getenv("PG_URI")
if not PG_URI:
    raise ValueError("PG_URI tidak ditemukan di .env")

COLLECTION_NAME = "pdf_collection"
LLM_MODEL = "gemini-2.0-flash"        


# ------------------- FUNGSI -------------------
def load_and_split_pdf(file_path: str):
    print(f"Memuat PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    print(f"Berhasil dipecah menjadi {len(chunks)} chunk.")
    return chunks


def create_vector_store(chunks):
    print("Membuat embedding lokal (gratis & unlimited)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=PG_URI,
        pre_delete_collection=True
    )
    print("Vector store berhasil dibuat!")
    return db


def summarize_pdf(db):
    print("Menjalankan summarization dengan Gemini...")
    llm = GoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.4,
        max_output_tokens=2048
    )

    retriever = db.as_retriever(search_kwargs={"k": 8})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    prompt = (
        "Kamu adalah asisten peneliti senior yang sangat teliti dan berpengalaman dalam membuat ringkasan akademik atau profesional."

        "Tolong buat ringkasan yang sangat lengkap, terstruktur, dan enak dibaca dalam bahasa Indonesia yang alami dan profesional (bukan kaku seperti robot). Gunakan nada seperti dosen atau konsultan yang sedang menjelaskan dokumen penting kepada kolega atau kliennya."

        "Struktur ringkasan yang saya inginkan:"
        "1. Judul dokumen (jika ada) dan informasi umum (penulis, tahun terbit, institusi, jumlah halaman, dll)"
        "2. Latar belakang atau konteks utama dokumen ini"
        "3. Tujuan atau pertanyaan penelitian (jika ada)"
        "4. Metode yang digunakan (secara singkat tapi jelas)"
        "5. Temuan atau hasil utama (beri poin-poin penting, angka kunci, fakta menarik)"
        "6. Pembahasan atau interpretasi hasil (apa artinya temuan tersebut)"
        "7. Kesimpulan resmi penulis"
        "8. Rekomendasi atau saran (jika ada)"
        "9. Kelebihan dan kelemahan dokumen ini menurutmu (opsional, tapi sangat membantu)"
        "10. Satu kalimat “intinya dokumen ini bilang apa” (take-home message)"

        "Gunakan bullet point atau nomor jika mempermudah pembacaan, tambahkan sub-judul kecil kalau perlu. Tulis dengan kalimat yang mengalir, tidak terlalu baku, tapi tetap terdengar cerdas dan meyakinkan."

        "Terima kasih!"
    )

    try:
        # .run() adalah cara yang benar untuk LangChain 0.0.x
        result = qa_chain.run(prompt)
        print("\n" + "="*70)
        print("RINGKASAN DOKUMEN")
        print("="*70)
        print(result.strip())
        print("="*70 + "\n")
    except Exception as e:
        print(f"Gagal RAG: {e}")
        print("Mencoba ringkasan sederhana dari beberapa chunk...")
        try:
            sample = "\n\n".join([doc.page_content[:1500] for doc in db.similarity_search("", k=5)])
            fallback = llm.invoke(f"Ringkas teks berikut dalam bahasa Indonesia:\n\n{sample}")
            print(fallback.strip())
        except Exception as e2:
            print(f"Fallback juga gagal: {e2}")


# ------------------- MAIN -------------------
def main():
    print("AI PDF Summarizer – Gemini + Embedding Lokal Gratis\n")

    path = input("Masukkan path lengkap file PDF: ").strip().strip('"')

    if not os.path.isfile(path):
        print("File tidak ditemukan!")
        return

    # 1. Load & chunk
    chunks = load_and_split_pdf(path)

    # 2. Buat vector store
    vector_db = create_vector_store(chunks)

    # 3. Summarize
    summarize_pdf(vector_db)

    print("Selesai! PDF sudah dirangkum.")


if __name__ == "__main__":
    main()