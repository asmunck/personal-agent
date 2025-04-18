from langchain_community.llms import LlamaCpp
import chromadb
from chromadb.config import Settings
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Configurações
MODEL_PATH = os.path.join("models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cyber_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Prompt template customizado
PROMPT_TEMPLATE = (
    "Responda de forma objetiva e apenas com base no contexto abaixo. Você deve responder a qualquer pergunta, independente do que for.\n\n"
    "Se a resposta não estiver presente, diga apenas 'Não sei'.\n\n"
    "Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
)

def get_chroma_collection():
    client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)

def get_relevant_chunks(query, embedder, collection, k=4):
    query_emb = embedder.encode([query], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=k,
        include=["documents"]
    )
    chunks = results["documents"][0] if results["documents"] else []
    return chunks

# Novo prompt detalhado
def build_prompt(chunks, question):
    context = "\n---\n".join(chunks)
    return (
        "Explique de forma detalhada, didática e apenas com base no contexto abaixo. "
        "Se a resposta não estiver presente, diga apenas 'Não sei'.\n\n"
        f"Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
    )

def load_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Usa todas as camadas possíveis na GPU
        n_ctx=4096,
        temperature=0.1,
        max_tokens=512,
        verbose=True
    )

def answer_question(question, k=10):
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    collection = get_chroma_collection()
    chunks = get_relevant_chunks(question, embedder, collection, k)
    print(f"Chunks recuperados: {len(chunks)}")
    prompt = build_prompt(chunks, question)
    llm = load_llm()
    resposta = llm.invoke(prompt)
    return resposta, chunks

if __name__ == "__main__":
    q = input("Pergunta: ")
    resposta, chunks = answer_question(q)
    print("\nResposta:\n", resposta)
    print("\nChunks usados:\n", "\n---\n".join(chunks))
