"""
Phase 1 – PDF Extraction
Extracts text from each handbook PDF using pdfplumber.
Outputs a list of document dicts with text + metadata.
"""

import json
import pdfplumber
from pathlib import Path
from tqdm import tqdm

#from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, PDF_SOURCES

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, PDF_SOURCES


def extract_single_pdf(pdf_path: Path, source_key: str) -> dict:
    """
    Extract all text from a single PDF.

    Returns:
        dict with keys: source_key, program, level, filename, pages
        where pages is a list of {page_num, text}
    """
    meta = PDF_SOURCES[source_key]
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page_num": i + 1,
                    "text": text.strip()
                })

    return {
        "source_key": source_key,
        "program": meta["program"],
        "level": meta["level"],
        "filename": meta["file"],
        "total_pages": len(pages),
        "pages": pages
    }


def extract_all_pdfs() -> list[dict]:
    """
    Extract text from all PDFs listed in PDF_SOURCES.
    Skips any files not found in data/raw/.

    Returns:
        List of document dicts (one per PDF)
    """
    documents = []
    missing = []

    for key, meta in tqdm(PDF_SOURCES.items(), desc="Extracting PDFs"):
        pdf_path = DATA_RAW_DIR / meta["file"]

        if not pdf_path.exists():
            missing.append(meta["file"])
            continue

        doc = extract_single_pdf(pdf_path, key)
        documents.append(doc)
        print(f" {meta['file']} — {doc['total_pages']} pages extracted")

    if missing:
        print(f"\n⚠ Missing PDFs (skipped): {missing}")
        print(f"  Place them in: {DATA_RAW_DIR}")

    return documents


# ── Preview utility ─────────────────────────────────────
def preview_extraction(doc: dict, max_chars: int = 500):
    """Print a quick preview of an extracted document."""
    print(f"\n{'='*60}")
    print(f"Program : {doc['program']} ({doc['level']})")
    print(f"File    : {doc['filename']}")
    print(f"Pages   : {doc['total_pages']}")
    print(f"{'='*60}")

    if doc["pages"]:
        first_page = doc["pages"][0]["text"]
        print(f"\n--- Page 1 (first {max_chars} chars) ---")
        print(first_page[:max_chars])
        print("...")


if __name__ == "__main__":
    docs = extract_all_pdfs()
    print(f"\nExtracted {len(docs)} documents total.")

    # Save to disk for Phase 2
    output_path = DATA_PROCESSED_DIR / "extracted_docs.json"
    with open(output_path, "w") as f:
        json.dump(docs, f, indent=2)
    print(f"Saved to: {output_path}")

    # Preview the first one
    if docs:
        preview_extraction(docs[0])