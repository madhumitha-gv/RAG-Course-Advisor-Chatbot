"""
Phase 6 – Streamlit Chat Interface
Run with: streamlit run app.py
"""

import streamlit as st
from src.chain import RAGChain


# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Luddy Course Advisor",
    page_icon="🎓",
    layout="centered",
)

st.title("Luddy Course Advisor")
st.caption(
    "AI-powered course advising for Luddy School graduate students — "
    "powered by RAG over official program handbooks."
)


# ── Initialize RAG Chain (cached) ───────────────────────
@st.cache_resource
def load_chain():
    return RAGChain()


chain = load_chain()


# ── Chat History ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[Source {s['source_num']}]** "
                        f"{s['program']} ({s['level']}) — "
                        f"*{s['heading']}* — `{s['file']}`"
                    )


# ── Chat Input ──────────────────────────────────────────
if prompt := st.chat_input("Ask about courses, requirements, or programs..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching handbooks..."):
            result = chain.query(prompt)

        st.markdown(result["answer"])

        # Show sources in expandable section
        if result["sources"]:
            with st.expander("📚 View Sources"):
                for s in result["sources"]:
                    st.markdown(
                        f"**[Source {s['source_num']}]** "
                        f"{s['program']} ({s['level']}) — "
                        f"*{s['heading']}* — `{s['file']}`"
                    )

    # Save assistant message with sources
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })


# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        "This chatbot uses **hybrid retrieval** (BM25 + FAISS) "
        "with **Reciprocal Rank Fusion** over 10 official Luddy "
        "graduate handbooks to provide grounded course advising."
    )

    st.markdown("---")
    st.markdown("**Programs covered:**")
    programs = [
        "Computer Science (PhD, MS)",
        "Data Science (Residential MS)",
        "Data Science (Online MS)",
        "HCID (MS)",
        "Informatics (MS, PhD)",
        "Information & Library Science (MS, PhD)",
        "Intelligent Systems Engineering",
        "Secure Computing (MS)",
    ]
    for p in programs:
        st.markdown(f"- {p}")

    st.markdown("---")
    st.caption("Always verify with your academic advisor.")
