"""
Core RAG logic: retrieve relevant BNS sections, then generate a grounded
answer with the Gemini API.

Setup:
    export GEMINI_API_KEY="your-key-here"   # https://aistudio.google.com/apikey

Usage:
    from rag_chain import answer_question

    result = answer_question("What is the punishment for theft?")
    print(result["answer"])
    print(result["sources"])
"""

import sys


import os
import re

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig, GenerateContentConfig

load_dotenv()  # reads GEMINI_API_KEY from a .env file in the current folder

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
DB_PATH = "./bns_chroma_db"
COLLECTION_NAME = "bns_sections"
TOP_K = 5

SYSTEM_PROMPT = """You are a legal reference assistant for the Bharatiya Nyaya \
Sanhita (BNS), 2023. Answer ONLY using the provided context of BNS sections.

Rules:
- Always cite the exact Section number(s) you used.
- If the answer is not contained in the provided context, say clearly that \
you don't have that information rather than guessing.
- Do not invent section numbers, punishments, or legal terms that are not in \
the context.
- Keep answers concise and precise, as befits legal reference material."""


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def _get_collection():
    db = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return db.get_collection(COLLECTION_NAME)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the most relevant BNS section chunks for a query.

    If the query names a specific section number, that section is
    prioritized directly instead of relying purely on semantic search.
    """
    collection = _get_collection()
    client = get_client()

    explicit_section = re.search(r"section\s+(\d{1,3})", query, re.IGNORECASE)
    direct_hits = []
    if explicit_section:
        section_num = explicit_section.group(1)
        result = collection.get(ids=[f"section_{section_num}"])
        if result["ids"]:
            direct_hits.append({
                "section_number": result["metadatas"][0]["section_number"],
                "text": result["documents"][0],
            })

    embed_response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = embed_response.embeddings[0].values

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    semantic_hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        semantic_hits.append({
            "section_number": meta["section_number"],
            "text": doc,
        })

    seen = set()
    merged = []
    for hit in direct_hits + semantic_hits:
        if hit["section_number"] not in seen:
            merged.append(hit)
            seen.add(hit["section_number"])

    return merged[:top_k]


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    client = get_client()
    context_text = "\n\n".join(c["text"] for c in context_chunks)
    user_prompt = f"Context (BNS sections):\n{context_text}\n\nQuestion: {query}"

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


def answer_question(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "I couldn't find any relevant BNS sections for that question.",
            "sources": [],
        }
    answer = generate_answer(query, chunks)
    return {
        "answer": answer,
        "sources": [c["section_number"] for c in chunks],
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the punishment for theft?"
    result = answer_question(q)
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES: Sections", ", ".join(result["sources"]))
