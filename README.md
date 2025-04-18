# Agente RAG Local para Cibersegurança

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) totalmente local para responder perguntas sobre diferentes assuntos, utilizando apenas o conteúdo de documentos PDF fornecidos pelo usuário.

## Funcionalidades

- Busca semântica em PDFs.
- Respostas factuais baseadas exclusivamente no conteúdo dos PDFs.
- Processamento local: embeddings, banco vetorial e LLM rodam em sua máquina.
- Interface web via Streamlit para perguntas e respostas.

## Como funciona

1. Extração de texto dos PDFs (pasta `pdfs/`).
2. Chunking: divisão do texto em partes menores com referência ao arquivo.
3. Geração de embeddings locais com Sentence Transformers.
4. Indexação dos embeddings em um banco vetorial local (ChromaDB).
5. LLM local (modelo GGUF, ex: Mistral-7B-Instruct) responde perguntas usando apenas o contexto recuperado dos PDFs.
6. Interface web para interação.

## Requisitos

- Python 3.10
- NVIDIA GPU recomendada (para acelerar o LLM)
- Modelos GGUF (ex: Mistral-7B-Instruct-v0.2.Q4_K_M.gguf) baixados manualmente no Hugging Face
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/personal-agent.git
   cd personal-agent
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Baixe e coloque seus PDFs na pasta `pdfs/`.

5. Baixe o modelo GGUF desejado e coloque na pasta `models/`.

## Uso

1. Gere os embeddings:
   ```bash
   python generate_embeddings.py
   ```

2. Construa o banco vetorial:
   ```bash
   python build_chroma_index.py
   ```

3. Rode a interface web:
   ```bash
   streamlit run app.py
   ```

4. Acesse o endereço indicado (geralmente http://localhost:8501) e faça perguntas sobre o conteúdo dos PDFs.


## Estrutura do Projeto

```
personal-agent/
├── app.py
├── build_chroma_index.py
├── chunk_texts.py
├── generate_embeddings.py
├── pdf_processor.py
├── rag_agent.py
├── requirements.txt
├── pdfs/           # Coloque seus PDFs aqui (não versionado)
├── models/         # Coloque o modelo GGUF aqui (não versionado)
├── chroma_db/      # Banco vetorial local (não versionado)
├── embeddings.pkl  # Embeddings gerados (não versionado)
└── ...
```

## Licença

Este projeto é apenas para fins educacionais e de pesquisa.
Consulte a licença dos modelos LLM utilizados.
