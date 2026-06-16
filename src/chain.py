"""
Phase 5 – RAG Chain
Connects the hybrid retriever to Mistral API for grounded responses
with source citations.
"""

import os
from mistralai import Mistral
from dotenv import load_dotenv

from src.retriever import HybridRetriever

load_dotenv()

# ── Config ──────────────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-small-latest"


# ── System Prompt ───────────────────────────────────────

SYSTEM_PROMPT = """You are the Luddy Course Advisor, an AI assistant for graduate students 
at the Luddy School of Informatics, Computing, and Engineering at Indiana University Bloomington.

Your role is to help students with:
- Course selection and elective recommendations
- Degree requirement clarification
- Prerequisite chains and course sequencing
- Program-specific policies and milestones

RULES:
1. ONLY answer based on the provided context from official Luddy handbooks.
2. If the context does not contain enough information to answer, say so clearly.
   Do NOT make up information about courses, requirements, or policies.
3. Always cite which handbook/program your information comes from.
4. When listing courses, include course codes when available.
5. If a question spans multiple programs, compare them using the context provided.
6. Be concise but thorough. Students need accurate advising.
"""


def _format_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['program']} ({chunk['level']}) — "
            f"Section: {chunk['heading']} — File: {chunk['source_file']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


class RAGChain:
    """
    Retrieval-Augmented Generation chain:
    query → retrieve → format context → Mistral → response with citations
    """

    def __init__(self):
        self.retriever = HybridRetriever()
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        print("✓ RAG chain ready")

    def query(self, user_question: str) -> dict:
        """
        Full RAG pipeline.

        Returns:
            dict with keys: answer, sources
        """
        # Step 1: Retrieve relevant chunks
        retrieved = self.retriever.retrieve(user_question)

        # Step 2: Format context
        context = _format_context(retrieved)

        # Step 3: Build the prompt
        user_message = (
            f"Context from Luddy handbooks:\n\n{context}\n\n"
            f"---\n\n"
            f"Student question: {user_question}\n\n"
            f"Answer the question using ONLY the context above. "
            f"Cite the source number [Source N] for each piece of information."
        )

        # Step 4: Call Mistral
        response = self.client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content

        # Step 5: Build source citations
        sources = [
            {
                "source_num": i + 1,
                "program": chunk["program"],
                "level": chunk["level"],
                "heading": chunk["heading"],
                "file": chunk["source_file"],
                "rrf_score": chunk["rrf_score"],
            }
            for i, chunk in enumerate(retrieved)
        ]

        return {
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":
    chain = RAGChain()

    question = "What are the core courses required for the Data Science MS?"
    print(f"\nQuestion: {question}\n")

    result = chain.query(question)

    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['source_num']}] {s['program']} — {s['heading']} ({s['file']})")