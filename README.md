# RAG_Local

A local Retrieval-Augmented Generation (RAG) script that reads your PDF documents and answers questions using Ollama plus Chroma.

## Main File

- `rag.py`: end-to-end ingestion and interactive question answering

## What It Does

- Loads PDF files from a local `RAG_documents/` folder
- Extracts text with `pdfplumber`
- Splits content into chunks
- Stores embeddings in a local Chroma vector store
- Retrieves relevant context and answers questions with Ollama

## Requirements

- Python 3
- Ollama installed and running
- Required Python packages installed

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python rag.py
```

## Notes

- `RAG_documents/`, `chroma_db/`, and `processed_files.json` are local runtime/data artifacts and are not included in this publishable repo.
- If you change the embedding model, rebuild the vector store so old and new embeddings are not mixed.
