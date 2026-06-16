"""
Central configuration for the Luddy Course Advisor.
All paths, model names, and hyperparameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR = ROOT_DIR / "indexes"

# Create dirs if they don't exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── PDF Source Mapping ──────────────────────────────────
# Maps a short program key to its PDF filename.
# Drop your PDFs into data/raw/ and update filenames here.
PDF_SOURCES = {
    "cs_grad":          {"file": "cs-graduate-student-handbook.pdf",     "program": "Computer Science",              "level": "PhD, MS, Accelerated MS"},
    "ds_residential":   {"file": "ds-residential-ms-handbook.pdf",      "program": "Data Science (Residential)",     "level": "MS"},
    "ds_online":        {"file": "ds-online-masters-26-27.pdf",         "program": "Data Science (Online)",          "level": "MS"},
    "hcid":             {"file": "hcid-program-handbook.pdf",           "program": "Human-Computer Interaction Design", "level": "MS"},
    "info_ms":          {"file": "info-ms-guidebook.pdf",               "program": "Informatics",                   "level": "MS"},
    "info_phd":         {"file": "info-phd-guidebook.pdf",              "program": "Informatics",                   "level": "PhD"},
    "ils_ms":           {"file": "ils-masters-student-handbook.pdf",    "program": "Information & Library Science",  "level": "MS"},
    "ils_phd":          {"file": "phd-information-science-handbook.pdf","program": "Information & Library Science",  "level": "PhD"},
    "ise":              {"file": "ise-graduate-handbook.pdf",           "program": "Intelligent Systems Engineering","level": "MS, PhD"},
    "secure_computing": {"file": "ms-secure-computing-handbook.pdf",   "program": "Secure Computing",              "level": "MS"},
}

# ── Model Config ────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ── Chunking Hyperparameters ────────────────────────────
MAX_CHUNK_SIZE = 800       # chars – ceiling for a single chunk
CHUNK_OVERLAP = 100        # chars – overlap when splitting oversized sections

# ── Retrieval Hyperparameters ───────────────────────────
TOP_K_PER_RETRIEVER = 10   # how many results each retriever returns
TOP_K_FINAL = 5            # how many fused results go to the LLM
RRF_K = 60                 # RRF constant (standard default)
