"""
Embeds BNS section chunks with the Gemini API and stores them in a
persistent ChromaDB collection on disk.

Setup:
    Get a free API key from https://aistudio.google.com/apikey
    Put it in a .env file (same folder as this script):
        GEMINI_API_KEY=your-key-here

Usage:
    python build_index.py raw_extracted_text.txt

This script is resumable: if it's interrupted (crash, rate limit, closing
the terminal), just run it again -- it skips sections already in the index.
"""

import os
import sys
import time

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig

from clean_and_chunk import clean_raw_text, chunk_by_section

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
DB_PATH = "./bns_chroma_db"
COLLECTION_NAME = "bns_sections"
BATCH_SIZE = 15          # chunks per API call
SECONDS_BETWEEN_BATCHES = 15  # keeps us under the free tier's ~100 items/min cap


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, retrying with backoff on rate limit errors."""
    for attempt in range(5):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f"  Batch failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Embedding batch failed after 5 attempts.")


def build_index(raw_text_path: str):
    with open(raw_text_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_raw_text(raw)
    chunks = chunk_by_section(cleaned)
    print(f"Chunked into {len(chunks)} sections.")

    client = get_client()
    db = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = db.get_or_create_collection(COLLECTION_NAME)

    # Resume support: skip sections already embedded in a previous run
    existing_ids = set(collection.get()["ids"])
    remaining = [c for c in chunks if f"section_{c['section_number']}" not in existing_ids]
    print(f"{len(chunks) - len(remaining)} sections already indexed, "
          f"{len(remaining)} remaining.")

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(client, texts)

        collection.add(
            ids=[f"section_{c['section_number']}" for c in batch],
            documents=texts,
            metadatas=[{"section_number": c["section_number"]} for c in batch],
            embeddings=embeddings,
        )
        done = min(i + BATCH_SIZE, len(remaining))
        print(f"Embedded sections {batch[0]['section_number']}-{batch[-1]['section_number']} "
              f"({done}/{len(remaining)} remaining batch)")

        if done < len(remaining):
            time.sleep(SECONDS_BETWEEN_BATCHES)

    total = collection.count()
    print(f"\nIndex complete: {total} sections stored at '{DB_PATH}'.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "raw_extracted_text.txt"
    build_index(path)