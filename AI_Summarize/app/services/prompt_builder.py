# app/services/prompt_builder.py

from langchain_core.messages import SystemMessage, HumanMessage

def build_prompt():
    system = SystemMessage(
        content=(
            "Kamu adalah analis dokumen profesional. "
            "Tugasmu adalah membaca, memahami, dan meringkas isi PDF yang dilampirkan."
        )
    )

    human = HumanMessage(
        content=(
            "Tolong analisis PDF terlampir dan berikan ringkasan dengan struktur berikut:\n"
            "1. Ringkasan Umum\n"
            "2. Poin-poin Penting\n"
            "3. Insight Penting\n"
            "4. Analisis Tabel (jika ada)\n"
            "5. Analisis Gambar/Diagram (jika ada)\n\n"
            "Jika tidak ada tabel atau gambar, sebutkan tidak ada."
        )
    )

    return [system, human]
