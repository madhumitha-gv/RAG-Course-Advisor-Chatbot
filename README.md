# 🎓 Luddy Course Advisor

AI-powered course recommender chatbot for graduate students at the **Luddy School of Informatics, Computing, and Engineering**, Indiana University Bloomington.

Uses **Retrieval-Augmented Generation (RAG)** with hybrid retrieval — combining **BM25 lexical search** with **FAISS semantic search** — over 10 official program handbooks to deliver grounded, citation-backed advising responses.

## Architecture

```
User Query
    │
    ├──→ BM25 (lexical) ──→ Top-K results ──┐
    │                                         ├──→ Reciprocal Rank Fusion ──→ Top-K fused
    └──→ FAISS (semantic) ──→ Top-K results ──┘              │
                                                              ▼
                                                    Context + System Prompt
                                                              │
                                                              ▼
                                                     Mistral Small API
                                                              │
                                                              ▼
                                                   Answer with Citations
```

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| PDF Extraction | `pdfplumber` | Preserves tables & structured layouts in academic handbooks |
| Embeddings | `all-MiniLM-L6-v2` | Fast, CPU-friendly, strong performance on short text |
| Vector Search | `FAISS` (IndexFlatIP) | Direct score access needed for RRF fusion |
| Lexical Search | `BM25Okapi` | Exact match for course codes, policy terms |
| Fusion | Reciprocal Rank Fusion | Rank-based, no score normalization needed |
| LLM | Mistral Small | Strong instruction-following for grounded QA |
| Frontend | Streamlit | Rapid chat UI with expandable source citations |

## Programs Covered

- Computer Science (PhD, MS, Accelerated MS)
- Data Science — Residential (MS)
- Data Science — Online (MS)
- Human-Computer Interaction Design (MS)
- Informatics (MS, PhD)
- Information & Library Science (MS, PhD)
- Intelligent Systems Engineering (MS, PhD)
- Secure Computing (MS)

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/madhumitha-gv/luddy-course-advisor.git
cd luddy-course-advisor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env.example .env
# Edit .env and add your Mistral API key
```

### 3. Add Handbooks

Place all 10 PDF handbooks in `data/raw/`. See `src/config.py` → `PDF_SOURCES` for expected filenames. Rename your files to match or update the config.

### 4. Build Pipeline

```bash
# Step 1: Extract text from PDFs
python -m src.extract

# Step 2: Chunk documents
python -m src.chunk

# Step 3: Build FAISS + BM25 indexes
python -m src.index_builder
```

### 5. Run the App

```bash
streamlit run app.py
```

## Project Structure

```
luddy-course-advisor/
├── data/
│   ├── raw/                  # Place handbook PDFs here
│   └── processed/            # Extracted chunks (auto-generated)
├── indexes/                  # FAISS + BM25 indexes (auto-generated)
├── src/
│   ├── config.py             # Paths, models, hyperparameters
│   ├── extract.py            # Phase 1 – PDF text extraction
│   ├── chunk.py              # Phase 2 – Section-based chunking
│   ├── index_builder.py      # Phase 3 – FAISS + BM25 index creation
│   ├── retriever.py          # Phase 4 – Hybrid retrieval + RRF
│   └── chain.py              # Phase 5 – Mistral RAG chain
├── app.py                    # Phase 6 – Streamlit chat interface
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Example Queries

- "What are the core courses for the Data Science MS?"
- "Can I take an Informatics elective as a CS PhD student?"
- "What are the prerequisite requirements for DSCI-D590?"
- "Compare the thesis vs non-thesis track for Informatics MS"
- "When is the qualifying exam for Computer Science PhD?"
