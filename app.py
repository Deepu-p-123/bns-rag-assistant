"""
Streamlit chat UI for the BNS RAG assistant.

Usage:
    streamlit run app.py
"""

import streamlit as st
from rag_chain import answer_question

st.set_page_config(page_title="BNS Assistant", page_icon="⚖️", layout="centered")

st.title("⚖️ BNS Assistant")
st.caption(
    "Ask questions about the Bharatiya Nyaya Sanhita, 2023. "
    "Answers are grounded in the actual Gazette text and cite section numbers."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.caption("Sections referenced: " + ", ".join(msg["sources"]))

# Chat input
user_query = st.chat_input("Ask about a BNS section, offence, or punishment...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Looking through the BNS..."):
            try:
                result = answer_question(user_query)
                st.markdown(result["answer"])
                if result["sources"]:
                    st.caption("Sections referenced: " + ", ".join(result["sources"]))
            except Exception as e:
                result = {"answer": f"Something went wrong: {e}", "sources": []}
                st.error(result["answer"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant retrieves relevant sections from the Bharatiya Nyaya "
        "Sanhita, 2023 and answers using only that retrieved text. It always "
        "cites the section numbers it relied on."
    )
    st.warning(
        "This is a reference tool, not legal advice. Always verify against "
        "the official Gazette text or consult a qualified lawyer for real "
        "legal matters."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()