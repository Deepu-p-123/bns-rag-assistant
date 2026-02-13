import pandas as pd
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "bns_sections.csv")

df = pd.read_csv(CSV_PATH)

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("bns_sections.csv")
df.columns = df.columns.str.strip()

texts = []

for _, row in df.iterrows():
    text = f"""
Chapter {row['Chapter']} - {row['Chapter_name']}
Section {row['Section']} - {row['Section _name']}

{row['Description']}
"""
    texts.append(text)


# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(texts, convert_to_numpy=True)

# -----------------------------
# FAISS INDEX (Cosine Similarity)
# -----------------------------

embeddings = embeddings.astype("float32")
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine
index.add(embeddings)


# -----------------------------
# LOCAL LLM (FLAN-T5)
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------

def retrieve(query, k=3):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)
    results = [texts[i] for i in indices[0]]
    return results


# -----------------------------
# GENERATE ANSWER
# -----------------------------

import re

def generate_answer(query):

    # -------- STEP 1: Direct Section Lookup --------
    match = re.search(r"section\s+(\d+)", query.lower())

    if match:
        section_number = int(match.group(1))

        result = df[df["Section"] == section_number]

        if not result.empty:
            row = result.iloc[0]

            context = f"""
Chapter {row['Chapter']} - {row['Chapter_name']}
Section {row['Section']} - {row['Section _name']}

{row['Description']}
"""
            return context, [context]

    # -------- STEP 2: Semantic Retrieval --------
    context_chunks = retrieve(query)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a legal assistant specialized in Bharatiya Nyaya Sanhita (BNS).

Answer strictly using the context.
Mention section number clearly.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=300)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, context_chunks
