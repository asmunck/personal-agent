from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from chunk_texts import chunk_documents
from pdf_processor import process_pdfs

EMBEDDINGS_FILE = "embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def generate_embeddings(chunks, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def save_embeddings(chunks, embeddings, file_path=EMBEDDINGS_FILE):
    data = [{**chunk, "embedding": emb.tolist()} for chunk, emb in zip(chunks, embeddings)]
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_embeddings(file_path=EMBEDDINGS_FILE):
    with open(file_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    docs = process_pdfs()
    chunks = chunk_documents(docs)
    embeddings = generate_embeddings(chunks)
    save_embeddings(chunks, embeddings)
    print(f"Embeddings gerados e salvos em {EMBEDDINGS_FILE}.")
