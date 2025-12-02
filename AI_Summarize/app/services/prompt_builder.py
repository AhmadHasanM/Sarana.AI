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
            "Berikan ringkasan lengkap dari dokumen berikut dalam bahasa Indonesia.\n\n"
            "Gunakan struktur output berikut:\n"
            "1. **Ringkasan Umum:** (3–5 kalimat ringkasan utama)\n"
            "2. **Poin-poin Penting:** (bullet points berisi informasi terpenting)\n"
            "3. **Insight Penting:** (analisis/interpretasi dari isi dokumen)\n"
            "4. **Analisis Tabel:** (jika ada tabel, jelaskan isinya dan maknanya)\n"
            "5. **Analisis Gambar/Diagram:** (jika ada gambar, jelaskan apa yang ditampilkan dan insightnya)\n\n"
            "Jika tidak ada tabel atau gambar, cukup tulis: 'Tidak ada tabel/gambar yang ditemukan.'\n\n"
        
        )
    )

    return [system, human]
