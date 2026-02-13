import pandas as pd
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load Data
CSV_PATH = "data/bns_sections.csv"
df = pd.read_csv(CSV_PATH)

# 2. Prepare Documents
# Assuming your CSV has 'section_id', 'title', and 'description'
documents = []
for _, row in df.iterrows():
    text = f"Section {row['section_id']}: {row['title']}\n{row['description']}"
    doc = Document(
        page_content=text,
        metadata={"section": row['section_id'], "title": row['title']}
    )
    documents.append(doc)

# 3. Create Embeddings
print("Loading embedding model... (this may take a minute)")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Build and Save FAISS Index
print("Building vector store...")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")

print("Success! 'faiss_index' folder created.")