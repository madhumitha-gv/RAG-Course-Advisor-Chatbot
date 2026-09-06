"""
Phase 2 – Smart Chunking
Section-based splitting first, size-based fallback for oversized sections.
Each chunk carries metadata for retrieval and citation.
"""

import re
import json
import hashlib
from pathlib import Path

from src.config import MAX_CHUNK_SIZE, CHUNK_OVERLAP, DATA_PROCESSED_DIR


# ── Page Header/Footer Cleanup ──────────────────────────

# Patterns for lines that are page headers, footers, or junk
JUNK_LINE_PATTERNS = [
    r"^\d+$",                                          # Standalone page numbers
    r"^Page\s+\d+",                                    # "Page 5"
    r"^Guidebook\s+\d+",                               # "Guidebook 32"
    r"^\d{8}\s+\d+",                                   # Date-stamped page nums like "20250717 5"
    r"^Rev\.$",                                        # Revision markers
    r"^Luddy School of Informatics",                   # Repeated headers
    r"^Indiana University Bloomington",                # Repeated headers
    r"^https?://",                                     # URLs on their own line
    r"^\d+/\d+/\d+,\s+\d+:\d+\s+(AM|PM)",            # Timestamps like "6/15/26, 10:32 PM"
    r"^Master of .+ : Academic Bulletin",              # Bulletin headers
]

# Repeated table headers that appear on every page
TABLE_HEADER_PATTERNS = [
    r"^CORE REQUIREMENT\s+COURSE\s+TERM\s+CREDITS\s+GRADE",
    r"^REQUIREMENT\s+COURSE\s+TERM\s+CREDITS",
    r"^COURSE\s+TERM\s+CREDITS\s+GRADE",
]


def _is_junk_line(line: str) -> bool:
    """Check if a line is a header, footer, or junk to remove."""
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in JUNK_LINE_PATTERNS + TABLE_HEADER_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    return False


def _clean_text(text: str) -> str:
    """Remove page headers, footers, and junk lines from extracted text."""
    lines = text.split("\n")
    cleaned = [line for line in lines if not _is_junk_line(line)]
    return "\n".join(cleaned)


def _markdown_heading(line: str) -> str | None:
    """Return a Markdown heading title, excluding page marker comments."""
    match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line.strip())
    return match.group(1).strip() if match else None


def _markdown_to_sections(markdown: str) -> list[dict]:
    """Split Markdown into sections while retaining page markers in the text."""
    sections = []
    current_heading = "General Information"
    current_lines = []

    for line in markdown.splitlines():
        heading = _markdown_heading(line)
        if heading and current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append({"heading": current_heading, "text": section_text})
            current_heading = heading
            current_lines = []
        else:
            current_lines.append(line)

    section_text = "\n".join(current_lines).strip()
    if section_text:
        sections.append({"heading": current_heading, "text": section_text})
    return sections


# ── Section Detection ───────────────────────────────────

# Tighter heading patterns — must look like real section titles
HEADING_PATTERNS = [
    r"^\d+\.\s+[A-Z][A-Za-z\s]{5,60}$",              # "1. Course Requirements" or "4. Ph.D. in Computer Science Curriculum"
    r"^\d+\.\d+\.?\s+[A-Z]",                          # "2.1. Course Requirements"
    r"^(?:Section|Part|Chapter)\s+\d+",                # Section/Part/Chapter N
    r"^[IVXLC]{2,}\.\s+[A-Z]",                         # Roman numeral headings (min 2 chars)",                           # Roman numeral headings
]

# ALL CAPS headings — but only reasonable-length ones that aren't table headers
def _is_valid_caps_heading(line: str) -> bool:
    """Check if an ALL CAPS line is a real heading vs table header/junk."""
    stripped = line.strip()
    # Must be 5-80 chars, ALL CAPS with spaces/punctuation allowed
    if not re.match(r"^[A-Z][A-Z\s&,/\-()]{4,79}$", stripped):
        return False
    # Reject known non-headings
    reject_words = ["COURSE", "TERM", "CREDITS", "GRADE", "REQUIREMENT COURSE",
                     "WELCOME", "REV", "BULLETIN"]
    for word in reject_words:
        if word in stripped:
            return False
    # Reject if it's more than 5 words (likely a sentence, not a heading)
    if len(stripped.split()) > 6:
        return False
    return True


def _is_heading(line: str) -> bool:
    """Check if a line is a section heading."""
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 100:
        return False

    # Check regex patterns
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, stripped):
            return True

    # Check valid ALL CAPS headings
    if _is_valid_caps_heading(stripped):
        return True

    return False


def _split_into_sections(full_text: str) -> list[dict]:
    """
    Split document text into sections based on heading detection.
    Returns list of {heading, text} dicts.
    """
    lines = full_text.split("\n")
    sections = []
    current_heading = "General Information"
    current_lines = []

    for line in lines:
        if _is_heading(line) and current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append({
                    "heading": current_heading,
                    "text": section_text
                })
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append({
                "heading": current_heading,
                "text": section_text
            })

    return sections


def _split_oversized(text: str, max_size: int, overlap: int) -> list[str]:
    """
    Fallback splitter for sections exceeding max_size.
    Splits on sentence boundaries with overlap.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        if current_len + sent_len > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Build overlap from end of current chunk
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current_chunk = overlap_sentences
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ── Minimum Quality Filter ──────────────────────────────

MIN_CHUNK_CHARS = 50  # Chunks shorter than this are junk


# ── Main Chunking Pipeline ──────────────────────────────

def chunk_document(doc: dict) -> list[dict]:
    """
    Chunk a single extracted document into retrieval-ready pieces.

    Args:
        doc: Output from extract_single_pdf()

    Returns:
        List of chunk dicts with keys:
            chunk_id, text, heading, program, level, source_file
    """
    # Step 1: Prefer the extracted Markdown representation; support old JSON artifacts.
    markdown = doc.get("markdown")
    if markdown:
        sections = _markdown_to_sections(markdown)
    else:
        full_text = "\n\n".join(page["text"] for page in doc["pages"])

        # Step 2: Clean out headers, footers, junk
        full_text = _clean_text(full_text)

        # Step 3: Split legacy plain-text artifacts by headings
        sections = _split_into_sections(full_text)

    # Step 4: Chunk sections, splitting oversized ones
    chunks = []
    for section in sections:
        if not section["text"].strip():
            continue

        if len(section["text"]) <= MAX_CHUNK_SIZE:
            text_pieces = [section["text"]]
        else:
            text_pieces = _split_oversized(
                section["text"], MAX_CHUNK_SIZE, CHUNK_OVERLAP
            )

        for i, text in enumerate(text_pieces):
            # Step 5: Filter out tiny junk chunks
            if len(text.strip()) < MIN_CHUNK_CHARS:
                continue

            chunk_id = hashlib.md5(
                f"{doc['source_key']}:{section['heading']}:{i}".encode()
            ).hexdigest()[:12]

            chunks.append({
                "chunk_id": chunk_id,
                "text": text.strip(),
                "heading": section["heading"],
                "program": doc["program"],
                "level": doc["level"],
                "source_file": doc["filename"],
            })

    return chunks


def chunk_all_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk all extracted documents.
    Saves the result to data/processed/chunks.json.
    """
    all_chunks = []

    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        print(f"  ✓ {doc['filename']} → {len(doc_chunks)} chunks")

    output_path = DATA_PROCESSED_DIR / "chunks.json"
    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {output_path}")

    return all_chunks


# ── Preview utility ─────────────────────────────────────

def preview_chunks(chunks: list[dict], n: int = 3):
    """Print a preview of the first n chunks."""
    for chunk in chunks[:n]:
        print(f"\n{'─'*50}")
        print(f"ID      : {chunk['chunk_id']}")
        print(f"Program : {chunk['program']} ({chunk['level']})")
        print(f"Heading : {chunk['heading']}")
        print(f"Length  : {len(chunk['text'])} chars")
        print(f"Text    : {chunk['text'][:200]}...")


if __name__ == "__main__":
    import json as _json
    sample_path = DATA_PROCESSED_DIR / "extracted_docs.json"
    if sample_path.exists():
        with open(sample_path) as f:
            docs = _json.load(f)
        chunks = chunk_all_documents(docs)
        preview_chunks(chunks)
    else:
        print("Run extract.py first to generate extracted docs.")