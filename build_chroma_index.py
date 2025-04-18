import chromadb
from chromadb.config import Settings
import pickle
import os
from generate_embeddings import EMBEDDINGS_FILE

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cyber_chunks"
METADATA_FILE = os.path.join(CHROMA_DIR, "metadata.pkl")

# Carrega embeddings e metadados
with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)

# Inicializa o ChromaDB local
client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
collection = client.get_or_create_collection(COLLECTION_NAME)

# Adiciona os embeddings e metadados ao ChromaDB
ids = [f"chunk_{i}" for i in range(len(data))]
embeddings = [d["embedding"] for d in data]
metadatas = [{k: v for k, v in d.items() if k != "embedding"} for d in data]
documents = [d["text"] for d in data]

collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)

# Salva os metadados para referência rápida
os.makedirs(CHROMA_DIR, exist_ok=True)
with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadatas, f)

print(f"ChromaDB populado em {CHROMA_DIR} e metadados salvos em {METADATA_FILE}")
