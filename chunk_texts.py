from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf_processor import process_pdfs

def chunk_documents(documents, chunk_size=2000, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        print(f"Arquivo: {doc['file_name']} - {len(splits)} chunks gerados.")
        for i, chunk in enumerate(splits):
            chunks.append({
                "file_name": doc["file_name"],
                "chunk_id": i,
                "text": chunk
            })
    print(f"Total de chunks gerados: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    docs = process_pdfs()
    chunks = chunk_documents(docs)
    print(f"Total de chunks gerados: {len(chunks)}")
    for c in chunks[:3]:
        print(f"Arquivo: {c['file_name']} | Chunk: {c['chunk_id']} | Tamanho: {len(c['text'])} caracteres")
