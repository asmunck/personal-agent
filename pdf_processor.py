import os
from pypdf import PdfReader

def list_pdfs(pdf_dir="pdfs"):
    return [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

def extract_text_from_pdf(pdf_path):
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                text += page.extract_text() or ""
            except Exception as e:
                print(f"Erro ao extrair texto da página {i+1} de {pdf_path}: {e}")
                continue
        if text.strip():
            return text
    except Exception as e:
        print(f"Erro ao processar com pypdf: {e}")
    # Tenta fallback com pymupdf (fitz)
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            try:
                text += page.get_text()
            except Exception as e:
                print(f"Erro ao extrair texto (pymupdf) da página {page.number+1} de {pdf_path}: {e}")
        return text
    except Exception as e:
        print(f"Erro ao processar com pymupdf: {e}")
    return ""

def process_pdfs(pdf_dir="pdfs"):
    pdf_files = list_pdfs(pdf_dir)
    documents = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        documents.append({
            "file_name": os.path.basename(pdf_file),
            "text": text
        })
    return documents

if __name__ == "__main__":
    docs = process_pdfs()
    for doc in docs:
        print(f"Arquivo: {doc['file_name']} - {len(doc['text'])} caracteres extraídos.")
