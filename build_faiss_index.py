import faiss
import numpy as np
import pickle
import os

EMBEDDINGS_FILE = "embeddings.pkl"
INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

# Carrega embeddings e metadados
with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.load(f)

embeddings = np.array([d["embedding"] for d in data]).astype("float32")
metadatas = [{k: v for k, v in d.items() if k != "embedding"} for d in data]

# Cria índice FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Salva índice e metadados
faiss.write_index(index, INDEX_FILE)
with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadatas, f)

print(f"Índice FAISS salvo em {INDEX_FILE}")
print(f"Metadados salvos em {METADATA_FILE}")
