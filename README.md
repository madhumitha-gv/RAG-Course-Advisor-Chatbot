# 🎓 RAG Course Advisor Chatbot

AI-powered course recommender chatbot for graduate students at the **Luddy School of Informatics, Computing, and Engineering**, Indiana University Bloomington.

Uses **Retrieval-Augmented Generation (RAG)** with hybrid retrieval — combining **BM25 lexical search** with **FAISS semantic search** — over 10 official program handbooks to deliver grounded, citation-backed advising responses via a Streamlit chat interface.

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
                                                   Answer with Source Citations
```

**Why hybrid retrieval?** Neither retriever alone handles the full query spectrum. BM25 excels at exact matches — a student typing `DSCI-D590` needs lexical precision. FAISS captures semantic similarity — a question about "machine learning electives" should surface courses described as "predictive modeling" or "statistical learning." Reciprocal Rank Fusion merges both ranked lists without requiring score normalization across different scales.

## Tech Stack

| Component | Tool | Justification |
|-----------|------|---------------|
| PDF Extraction | `pdfplumber` | Preserves tables and structured layouts common in academic handbooks |
| Embeddings | `all-MiniLM-L6-v2` | 80MB, CPU-friendly, strong performance on short-text retrieval |
| Vector Search | `FAISS` (IndexFlatIP) | Raw score access required for RRF fusion; no persistence overhead |
| Lexical Search | `BM25Okapi` | Exact-match retrieval for course codes, policy terms, proper nouns |
| Fusion | Reciprocal Rank Fusion | Rank-based fusion — no score normalization needed between retrievers |
| LLM | Mistral Small (via API) | Strong instruction-following for grounded, context-only QA |
| Chunking | Section-based + size fallback | Preserves semantic coherence of handbook sections |
| Frontend | Streamlit | Chat UI with expandable source citations |

## Programs Covered

10 official Luddy graduate handbooks across 8 programs:

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
git clone https://github.com/madhumitha-gv/RAG-Course-Advisor-Chatbot.git
cd RAG-Course-Advisor-Chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

```bash
cp .env.example .env
# Edit .env and add your Mistral API key (https://console.mistral.ai)
```

### 3. Add Handbooks

Place all 10 PDF handbooks in `data/raw/`. See `src/config.py` → `PDF_SOURCES` for expected filenames.

### 4. Build the Pipeline

Run each phase sequentially — each saves its output for the next:

```bash
# Phase 1: Extract text from PDFs → data/processed/extracted_docs.json
python -m src.extract

# Phase 2: Chunk into retrieval units → data/processed/chunks.json
python -m src.chunk

# Phase 3: Build FAISS + BM25 indexes → indexes/
python -m src.index_builder
```

### 5. Test Retrieval (Optional)

```bash
# Runs test queries through hybrid retriever
python -m src.retriever

# Runs a full RAG query through Mistral
python -m src.chain
```

### 6. Launch the App

```bash
streamlit run app.py
```

## Project Structure

```
RAG-Course-Advisor-Chatbot/
├── data/
│   ├── raw/                  # Handbook PDFs (10 files)
│   └── processed/            # Extracted text + chunks (auto-generated)
├── indexes/                  # FAISS + BM25 indexes (auto-generated)
├── src/
│   ├── config.py             # Paths, model names, hyperparameters
│   ├── extract.py            # Phase 1 – PDF text extraction (pdfplumber)
│   ├── chunk.py              # Phase 2 – Section-based chunking with junk filtering
│   ├── index_builder.py      # Phase 3 – FAISS + BM25 dual index build
│   ├── retriever.py          # Phase 4 – Hybrid retrieval with RRF fusion
│   └── chain.py              # Phase 5 – Mistral RAG chain with grounded prompting
├── app.py                    # Phase 6 – Streamlit chat interface
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Example Queries

- "What are the core courses for the Data Science MS?"
- "Can I take an Informatics elective as a CS PhD student?"
- "Compare the thesis vs non-thesis track for Informatics MS"
- "When is the qualifying exam for Computer Science PhD?"
- "What machine learning courses are available as electives?"
- "What are the credit hour requirements for ISE?"
