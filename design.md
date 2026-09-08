# System Architecture & Design Document
## Simple Fish Care & Disease Management Chatbot (Streamlit + ChromaDB + Gemini)

---

## 1. Executive Summary

Proyek ini bertujuan untuk membangun **Chatbot Sederhana Penanganan Penyakit & Perawatan Ikan** menggunakan bahasa pemrograman **Python**, framework web **Streamlit**, basis data vektor **ChromaDB**, dan Large Language Model (LLM) **Google Gemini API**. 

Sistem ini menerapkan arsitektur **Retrieval-Augmented Generation (RAG)** sederhana. Pendekatan RAG memungkinkan chatbot memberikan jawaban yang sangat relevan, akurat, dan berbasis dokumen ilmiah/teknis (PDF penyakit ikan) yang disediakan oleh pengguna, sekaligus meminimalkan *hallucination* dan menghemat penggunaan *token* API.

---

## 2. High-Level System Architecture

Sistem terbagi menjadi 3 lapisan utama (3-Tier Concept):
1. **Presentation Layer (Frontend)**: Tampilan antarmuka berbasis **Streamlit** (UI/UX Chatbot, Sidebar API Key management, status loader).
2. **Retrieval & Processing Layer (RAG Engine)**:
   - **Document Ingestion**: Membaca file `.pdf` menggunakan `PyPDF2`.
   - **Chunking Engine**: Memotong dokumen teks tebal menjadi segmen-segmen kecil menggunakan `RecursiveCharacterTextSplitter`.
   - **Vector Database**: Menyimpan dan mencari potongan teks relevan menggunakan `ChromaDB`.
3. **Generation Layer (AI Model)**: Mengirimkan konteks relevan beserta pertanyaan *user* ke **Google Gemini API (`gemini-2.5-flash`)** untuk membuat jawaban akhir.

### Diagrams / Flow Diagrams

```
+-----------------------------------------------------------------------+
|                           USER INTERFACE                              |
|                   (Streamlit Web Application)                         |
+-----------------------------------+-----------------------------------+
                                    |
            1. User Prompt          |  2. Pass Context & Prompt
                                    v
+-----------------------------------+-----------------------------------+
|                        RAG & VECTOR ENGINE                            |
|                                                                       |
|  [PDF File] ---> (PyPDF2) ---> [Chunking] ---> (ChromaDB Vector Store) |
|                                                       |               |
|                                         3. Similarity | Top-3 Chunks  |
|                                            Search     v               |
+-------------------------------------------------------+---------------+
                                                        |
                                         4. Enriched Context + Prompt
                                                        v
+-------------------------------------------------------+---------------+
|                         GENERATIVE AI ENGINE                          |
|                       (Google Gemini API)                             |
+-----------------------------------------------------------------------+
```

---

## 3. Data Flow & Processing Pipeline

### Phase 1: Ingestion & Vectorization (On Startup / First Run)
1. **PDF Reading**: File `penyakit_ikan.pdf` diakses dari folder `dokumen/`.
2. **Text Extraction**: PyPDF2 mengekstrak seluruh teks per halaman.
3. **Recursive Chunking**: Teks dipecah dengan aturan:
   - `chunk_size`: 700 karakter
   - `chunk_overlap`: 100 karakter
4. **Vector Indexing**: Potongan teks dimasukkan ke dalam koleksi `penyakit_ikan` pada ChromaDB lokal (`./.chroma_db`).

### Phase 2: Query & Retrieval Pipeline (Per User Message)
1. **User Query**: Pengguna mengirimkan pertanyaan (misal: *"Bagaimana cara mengobati bintik putih pada ikan gurame?"*).
2. **Vector Query**: ChromaDB melakukan penelusuran kemiripan teks (*similarity search*) dan mengembalikan **Top-3 chunks** terbaik.
3. **Prompt Augmentation**: System prompt dibentuk dengan menggabungkan konteks RAG dan pertanyaan pengguna.
4. **LLM Generation**: Gemini API memproses prompt dan mengembalikan jawaban kontekstual.
5. **UI Rendering**: Hasil ditampilkan di jendela obrolan Streamlit.

---

## 4. Component Specification & Technology Stack

| Komponen | Teknologi / Library | Fungsi Utama |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Bahasa pemrograman utama |
| **Frontend Framework** | Streamlit | Antarmuka interaktif chat & sidebar |
| **PDF Parser** | PyPDF2 | Ekstraksi teks dari file dokumen PDF |
| **Text Splitter** | LangChain Text Splitters | Pemotongan teks (*chunking*) berbasis aturan |
| **Vector Database** | ChromaDB | Penyimpanan *embeddings* & kueri pencarian vektor |
| **LLM Provider** | Google Generative AI (`gemini-2.5-flash`) | Pengolahan bahasa alami & pembentukan jawaban |

---

## 5. File & Directory Structure

```text
chatbot-ikan/
├── .chroma_db/             # Folder penyimpanan persistensi Vector DB (dibuat otomatis)
├── dokumen/                # Folder penyimpanan file dokumen referensi
│   └── penyakit_ikan.pdf   # File utama sumber pengetahuan penyakit
├── app.py                  # Entrypoint aplikasi Streamlit & logika RAG
├── design.md               # Dokumentasi arsitektur & desain sistem
└── requirements.txt        # Daftar dependency project
```

---

## 6. Prompt Engineering & System Persona

Untuk memastikan respon chatbot konsisten, terstruktur, dan akurat, *system prompt* berikut diterapkan pada setiap kueri:

```text
Kamu adalah asisten ahli kesehatan dan perawatan ikan.
Gunakan referensi dari potongan dokumen berikut untuk menjawab pertanyaan pengguna:

--- KONTEKS RELEVAN DARI DOKUMEN ---
{relevant_context}
--- AKHIR KONTEKS ---

Aturan:
1. Utamakan informasi dari KONTEKS RELEVAN di atas.
2. Jika topik perawatan/penyakit tidak ada pada konteks, gunakan pengetahuan umum pemeliharaan/budidaya ikan yang praktis dan tepat.
3. Jawab dengan ringkas dan langsung ke solusi/penanganan.
```

---

## 7. Future Enhancements & Scalability

1. **Google Embedding Integration**: Mengintegrasikan `text-embedding-004` dari Google untuk peningkatan presisi pencarian vektor semantic.
2. **Multi-Document Support**: Fitur unggah PDF dinamis melalui UI Streamlit untuk mendukung lebih banyak jenis dokumen.
3. **Interactive Control**: Menambahkan tombol reset database pada sidebar untuk pembaharuan indeks dokumen secara cepat.
