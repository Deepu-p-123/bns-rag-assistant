# ⚖️ BNS Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system designed to query, search, and interpret the **Bharatiya Nyaya Sanhita (BNS)**. This tool uses semantic search to find relevant legal sections and a local LLM to explain them in plain language.



## 🌟 Features

- **Hybrid Retrieval:** Combines direct section lookup with semantic vector search.
- **FAISS Vector Store:** Efficient similarity search for high-dimensional legal embeddings.
- **Sentence-Transformers:** Uses `all-MiniLM-L6-v2` for high-speed, local embedding generation.
- **Local LLM (Flan-T5):** Private and cost-effective text generation without external API calls.
- **Streamlit UI:** A clean, user-friendly interface for legal professionals and students.

---

## 🏗️ Tech Stack

- **Language:** Python 3.10+
- **Orchestration:** LangChain
- **Vector DB:** FAISS (CPU)
- **Embeddings:** HuggingFace / Sentence-Transformers
- **Model:** Google/Flan-T5-base
- **Frontend:** Streamlit

---

## 📂 Project Structure

```text
bns_rag_app/
│
├── data/
│   └── bns_sections.csv       # Dataset containing Section No, Title, and Description
├── app.py                     # Streamlit web application
├── build_vectorstore.py       # Script to ingest CSV and create FAISS index
├── utils.py                   # Helper functions for retrieval and formatting
├── requirements.txt           # Project dependencies
└── .gitignore                 # Files to exclude from Git