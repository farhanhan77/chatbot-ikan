import os
import streamlit as st
import PyPDF2
import chromadb
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# KONFIGURASI & KONSTANTA
# ─────────────────────────────────────────────
# GANTI DENGAN INI:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "dokumen", "penyakit_ikan.pdf")
CHROMA_PATH     = ".chroma_db"
COLLECTION_NAME = "penyakit_ikan"
CHUNK_SIZE      = 700
CHUNK_OVERLAP   = 100
TOP_K           = 3
GEMINI_MODEL    = "gemini-3.6-flash"

# ─────────────────────────────────────────────
# FUNGSI: BACA PDF
# ─────────────────────────────────────────────
def read_pdf(pdf_path: str) -> str:
    """Ekstrak seluruh teks dari file PDF."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except FileNotFoundError:
        st.error(f"File PDF tidak ditemukan di: `{pdf_path}`")
    return text


# ─────────────────────────────────────────────
# FUNGSI: INISIALISASI VECTOR DB
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat & mengindeks dokumen penyakit ikan...")
def init_vector_db() -> chromadb.Collection:
    """
    Baca PDF → chunking → simpan ke ChromaDB.
    Menggunakan @st.cache_resource agar proses ini hanya berjalan sekali.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Hapus koleksi lama jika ada agar dokumen ter-refresh saat restart
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    raw_text = read_pdf(PDF_PATH)
    if not raw_text.strip():
        st.warning(
            ""
            ""
        )
        return collection

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(raw_text)

    # Masukkan ke ChromaDB
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection


# ─────────────────────────────────────────────
# FUNGSI: RETRIEVAL — ambil Top-K chunk relevan
# ─────────────────────────────────────────────
def retrieve_context(collection: chromadb.Collection, query: str) -> str:
    """Cari TOP_K potongan teks paling relevan dari ChromaDB."""
    if collection.count() == 0:
        return ""
    results = collection.query(
        query_texts=[query],
        n_results=min(TOP_K, collection.count()),
    )
    chunks = results.get("documents", [[]])[0]
    return "\n\n".join(chunks)


# ─────────────────────────────────────────────
# FUNGSI: GENERASI — kirim ke Gemini API
# ─────────────────────────────────────────────
def generate_answer(api_key: str, context: str, question: str) -> str:
    """Bangun prompt RAG lalu kirim ke Gemini, kembalikan teks jawaban."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    system_prompt = f"""Kamu adalah asisten ahli kesehatan dan perawatan ikan.
Gunakan referensi dari potongan dokumen berikut untuk menjawab pertanyaan pengguna:

--- KONTEKS RELEVAN DARI DOKUMEN ---
{context if context else "Tidak ada konteks dokumen yang tersedia."}
--- AKHIR KONTEKS ---

Aturan:
1. Utamakan informasi dari KONTEKS RELEVAN di atas.
2. Jika topik perawatan/penyakit tidak ada pada konteks, gunakan pengetahuan umum pemeliharaan/budidaya ikan yang praktis dan tepat.
3. Jawab dengan ringkas dan langsung ke solusi/penanganan.

Pertanyaan pengguna: {question}"""

    response = model.generate_content(system_prompt)
    return response.text


# ─────────────────────────────────────────────
# UI — STREAMLIT
# ─────────────────────────────────────────────
def main():
    # ── Konfigurasi halaman ──────────────────
    st.set_page_config(
        page_title="Chatbot Penyakit Ikan",
        page_icon="🐟",
        layout="centered",
    )

    # ── Header ──────────────────────────────
    st.title("🐟 Chatbot Penanganan Penyakit Ikan")
    st.caption(
        "Didukung oleh Google Gemini + RAG (ChromaDB) · "
        "Tanyakan seputar penyakit, gejala, dan penanganan ikan budidaya."
    )
    st.divider()

    # ── Sidebar — API Key & Info ─────────────
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Masukkan API Key Anda...",
            help="Dapatkan API Key gratis di https://aistudio.google.com/",
        )

        st.divider()
        st.markdown("### 📖 Cara Penggunaan")
        st.markdown(
            """
1. Masukkan **API Key** Gemini di atas.
2. Letakkan file `penyakit_ikan.pdf` di folder `dokumen/`.
3. Ketik pertanyaan di kolom chat.

**Contoh pertanyaan:**
- *Bagaimana cara mengobati bintik putih pada ikan gurame?*
- *Apa gejala penyakit KHV pada ikan koi?*
- *Cara mencegah jamur pada ikan lele?*
            """
        )

        st.divider()
        st.markdown("### 📂 Status Dokumen")
        if os.path.exists(PDF_PATH):
            st.success(f"✅ `penyakit_ikan.pdf` ditemukan")
        else:
            st.warning(f"⚠️ PDF belum ada di `dokumen/`")

        # Tombol reset percakapan
        st.divider()
        if st.button("🗑️ Reset Percakapan", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Inisialisasi session state ───────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Inisialisasi Vector DB (cached) ─────
    collection = init_vector_db()

    # ── Tampilkan riwayat chat ───────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🐟"):
            st.markdown(msg["content"])

    # ── Input pengguna ───────────────────────
    user_input = st.chat_input("Tanyakan tentang penyakit atau perawatan ikan...")

    if user_input:
        # Validasi API Key
        if not api_key:
            st.warning("⚠️ Masukkan Google Gemini API Key di sidebar terlebih dahulu.")
            st.stop()

        # Simpan & tampilkan pesan user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Retrieval + Generation
        with st.chat_message("assistant", avatar="🐟"):
            with st.spinner("Mencari informasi & menyusun jawaban..."):
                try:
                    context = retrieve_context(collection, user_input)
                    answer  = generate_answer(api_key, context, user_input)
                except Exception as e:
                    answer = f"❌ Terjadi kesalahan: {e}"

            st.markdown(answer)

            # Tampilkan konteks sumber (opsional, collapsible)
            if context:
                with st.expander("📄 Lihat konteks dokumen yang digunakan"):
                    st.text(context)

        # Simpan jawaban ke riwayat
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
