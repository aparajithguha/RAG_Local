#!/usr/bin/env python3
"""
Unified RAG System
- Automatically detects and ingests new PDF documents
- Interactive query interface
- One script to run everything
"""

import os
import json
import re
import warnings
warnings.filterwarnings('ignore')

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
import pdfplumber
import ollama

# ============================================================================
# Configuration
# ============================================================================
RAG_DOCUMENTS_FOLDER = "/Users/aparajithguha/Workspace/apara_space/RAG_documents"
PROCESSED_FILES_LOG = "/Users/aparajithguha/Workspace/apara_space/processed_files.json"
CHROMA_PERSIST_DIRECTORY = "/Users/aparajithguha/Workspace/apara_space/chroma_db"
BATCH_SIZE = 200
RETRIEVAL_K = 8
RETRIEVAL_FETCH_K = 20
KEYWORD_FALLBACK_K = 5
USE_RERANKING = False
RERANK_TOP_K = 5
RERANK_PREVIEW_CHARS = 500
MAX_CONTEXT_DOCS = 6
MAX_CONTEXT_CHARS_PER_DOC = 1200
model = "gemma4:e2b"
embedding_model = "nomic-embed-text"

# Create folders
os.makedirs(RAG_DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)

# ============================================================================
# File Tracking Functions
# ============================================================================
def load_processed_files() -> set:
    """Load the set of already processed files."""
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_files', []))
    return set()

def save_processed_files(processed_files: set):
    """Save the set of processed files to a log file."""
    with open(PROCESSED_FILES_LOG, 'w') as f:
        json.dump({'processed_files': list(processed_files)}, f, indent=2)

def get_new_files(processed_files: set) -> list:
    """Get list of new files that haven't been processed yet."""
    all_files = [f for f in os.listdir(RAG_DOCUMENTS_FOLDER) 
                 if f.endswith('.pdf')]
    new_files = [f for f in all_files if f not in processed_files]
    return new_files

# ============================================================================
# Fast PDF Processing with pdfplumber
# ============================================================================
def load_pdf_with_pdfplumber(file_path: str, file_name: str) -> list:
    """Extract text from PDF using pdfplumber (much faster)."""
    documents = []
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"    Extracting text from {total_pages} pages...", flush=True)
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source_file': file_name,
                            'page': page_num,
                            'total_pages': total_pages
                        }
                    )
                    documents.append(doc)
        
        if documents:
            print(f"    ✓ Extracted {len(documents)} pages")
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
    
    return documents

def load_and_process_documents(file_paths: list) -> list:
    """Load and process documents from given file paths."""
    all_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    
    for file_name in file_paths:
        file_path = os.path.join(RAG_DOCUMENTS_FOLDER, file_name)
        try:
            print(f"  Loading: {file_name}")
            documents = load_pdf_with_pdfplumber(file_path, file_name)
            all_documents.extend(documents)
        except Exception as e:
            print(f"  ✗ Error loading {file_name}: {str(e)}")
    
    # Split and chunk all documents
    if all_documents:
        print(
            "  Creating chunks with chunk_size=1200 and chunk_overlap=300...",
            flush=True,
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"  ✓ Created {len(chunks)} chunks", flush=True)
        return chunks
    return []

def add_documents_with_progress(vector_db: Chroma, chunks: list, batch_size: int = BATCH_SIZE):
    """Add documents to Chroma in batches and print progress."""
    global keyword_cache
    total_chunks = len(chunks)
    if total_chunks == 0:
        return

    print(f"   Adding {total_chunks} chunks to database...")
    for start_index in range(0, total_chunks, batch_size):
        end_index = min(start_index + batch_size, total_chunks)
        batch = chunks[start_index:end_index]
        vector_db.add_documents(documents=batch)

        progress_pct = (end_index / total_chunks) * 100
        print(
            f"     Saved {end_index}/{total_chunks} chunks ({progress_pct:.1f}%)",
            flush=True,
        )

    print("   ✓ Finished saving chunks")
    keyword_cache = None

# ============================================================================
# Initialization
# ============================================================================
print("\n" + "="*80)
print("RAG SYSTEM - INITIALIZATION")
print("="*80 + "\n")

print("1. Preparing embedding model...", end=" ", flush=True)
ollama.pull(embedding_model)
print("✓")

print("2. Loading vector database...", end=" ", flush=True)
vector_db = Chroma(
    collection_name="RAG_collections",
    embedding_function=OllamaEmbeddings(model=embedding_model),
    persist_directory=CHROMA_PERSIST_DIRECTORY,
)
print("✓")

print("3. Checking for new documents...")
processed_files = load_processed_files()
new_files = get_new_files(processed_files)

if new_files:
    print(f"   Found {len(new_files)} new file(s)")
    chunks = load_and_process_documents(new_files)
    
    if chunks:
        add_documents_with_progress(vector_db, chunks)
        
        processed_files.update(new_files)
        save_processed_files(processed_files)
        print(f"   ✓ Updated tracking\n")
else:
    print("   ✓ No new files\n")

print("4. Initializing LLM...", end=" ", flush=True)
llm = ChatOllama(model=model)
print("✓\n")

keyword_cache = None

# ============================================================================
# RAG Functions
# ============================================================================
def rerank_documents(question: str, documents: list, top_k: int = RERANK_TOP_K) -> list:
    """Rerank documents based on relevance to the question."""
    if not documents:
        return []

    print(
        f"  Reranking {len(documents)} retrieved chunks and keeping top {top_k}...",
        flush=True,
    )
    
    scoring_prompt = ChatPromptTemplate.from_template(
        """Rate how relevant this document is to the question on a scale of 1-10.
Question: {question}
Document: {document}
Reply with ONLY the number (1-10):"""
    )
    
    scores = []
    for doc in documents:
        try:
            score_str = llm.invoke(
                scoring_prompt.format_messages(
                    question=question,
                    document=doc.page_content[:RERANK_PREVIEW_CHARS],
                )
            ).content.strip()
            score = int(''.join(filter(str.isdigit, score_str.split('\n')[0]))) if any(c.isdigit() for c in score_str) else 5
            scores.append((doc, score))
        except:
            scores.append((doc, 5))
    
    sorted_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    print("  ✓ Reranking complete", flush=True)
    return [doc for doc, score in sorted_docs[:top_k]]

def tokenize_text(text: str) -> list:
    """Tokenize text into lowercase words for lightweight keyword matching."""
    return re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())

def build_query_variants(question: str) -> list:
    """Generate a few normalized query variants for more robust retrieval."""
    variants = [question.strip()]
    normalized = " ".join(tokenize_text(question))
    if normalized and normalized not in variants:
        variants.append(normalized)

    compact = re.sub(r"[^a-zA-Z0-9\s-]", " ", question.lower())
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact and compact not in variants:
        variants.append(compact)

    no_hyphen = compact.replace("-", " ")
    if no_hyphen and no_hyphen not in variants:
        variants.append(no_hyphen)

    return variants[:4]

def load_keyword_cache() -> list:
    """Load stored chunks from Chroma for keyword fallback retrieval."""
    global keyword_cache

    if keyword_cache is not None:
        return keyword_cache

    print("  Loading keyword fallback cache from vector store...", flush=True)
    collection_data = vector_db.get(include=["documents", "metadatas"])
    documents = collection_data.get("documents", [])
    metadatas = collection_data.get("metadatas", [])

    keyword_cache = []
    for page_content, metadata in zip(documents, metadatas):
        token_set = set(tokenize_text(page_content))
        keyword_cache.append(
            {
                "document": Document(
                    page_content=page_content,
                    metadata=metadata or {},
                ),
                "token_set": token_set,
                "normalized_text": page_content.lower(),
            }
        )

    print(f"  ✓ Keyword cache ready with {len(keyword_cache)} chunks", flush=True)
    return keyword_cache

def keyword_fallback_search(question: str, top_k: int = KEYWORD_FALLBACK_K) -> list:
    """Find relevant chunks using simple keyword overlap as a fallback."""
    query_variants = build_query_variants(question)
    query_tokens = set()
    for variant in query_variants:
        query_tokens.update(tokenize_text(variant))
    if not query_tokens:
        return []

    scored_docs = []

    for entry in load_keyword_cache():
        overlap = len(query_tokens & entry["token_set"])
        phrase_bonus = 0
        for variant in query_variants:
            normalized_variant = variant.lower()
            if normalized_variant and normalized_variant in entry["normalized_text"]:
                phrase_bonus += 2
        score = overlap + phrase_bonus
        if score > 0:
            scored_docs.append((entry["document"], score))

    scored_docs.sort(key=lambda item: item[1], reverse=True)
    return [doc for doc, _score in scored_docs[:top_k]]

def combine_documents(primary_docs: list, fallback_docs: list) -> list:
    """Combine two document lists without duplicates while preserving order."""
    combined_docs = []
    seen = set()

    for doc in primary_docs + fallback_docs:
        doc_key = (
            doc.metadata.get("source_file", ""),
            doc.metadata.get("page", ""),
            doc.page_content[:200],
        )
        if doc_key in seen:
            continue
        seen.add(doc_key)
        combined_docs.append(doc)

    return combined_docs

def semantic_search(question: str) -> list:
    """Run semantic retrieval across a few query variants and merge results."""
    query_variants = build_query_variants(question)
    all_docs = []

    for variant in query_variants:
        print(f"  Semantic search variant: {variant}", flush=True)
        base_retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_FETCH_K},
        )
        docs = base_retriever.invoke(variant)
        all_docs.extend(docs)

    return combine_documents(all_docs, [])

def limit_context_documents(documents: list, max_docs: int = MAX_CONTEXT_DOCS) -> list:
    """Keep a bounded number of documents for final context assembly."""
    return documents[:max_docs]

def retrieve_and_rerank(question: str) -> str:
    """Retrieve and rerank documents for better relevance."""
    print("  Searching vector database for similar chunks...", flush=True)
    docs = semantic_search(question)
    print(f"  ✓ Retrieved {len(docs)} candidate chunks", flush=True)
    keyword_docs = keyword_fallback_search(question)
    print(f"  ✓ Keyword fallback found {len(keyword_docs)} candidate chunks", flush=True)
    combined_docs = combine_documents(docs, keyword_docs)
    print(f"  ✓ Combined candidate pool has {len(combined_docs)} chunks", flush=True)
    if USE_RERANKING:
        selected_docs = rerank_documents(question, combined_docs, top_k=RERANK_TOP_K)
    else:
        print("  Skipping reranking and using combined chunks directly...", flush=True)
        selected_docs = combined_docs
    selected_docs = limit_context_documents(selected_docs)
    print(f"  ✓ Using {len(selected_docs)} chunks in final context", flush=True)
    
    print("  Building context from top-ranked chunks...", flush=True)
    context = "\n\n---\n\n".join([
        (
            f"[Source: {doc.metadata.get('source_file', 'Unknown')}"
            f" | Page: {doc.metadata.get('page', 'Unknown')}]\n"
            f"{doc.page_content[:MAX_CONTEXT_CHARS_PER_DOC]}"
        )
        for doc in selected_docs
    ])
    print("  ✓ Context ready", flush=True)
    
    return context if context else "No relevant documents found."

def query_rag(question: str):
    """Query the RAG system."""
    rag_response_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context. Prefer answering from the context even if the wording differs from the question. Synthesize across multiple excerpts when needed. Only say the context does not contain the answer when the retrieved context is clearly unrelated. When possible, mention the source file names you relied on."),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])
    
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: retrieve_and_rerank(x["question"])),
            "question": lambda x: x["question"],
        }
        | rag_response_prompt
        | llm
        | StrOutputParser()
    )
    
    print("  Sending context to the model and generating answer...", flush=True)
    answer = rag_chain.invoke({"question": question})
    print("  ✓ Answer generated", flush=True)
    return answer

# ============================================================================
# Interactive Mode
# ============================================================================
def main():
    """Interactive query interface."""
    try:
        doc_count = vector_db._collection.count()
    except:
        doc_count = 0
    
    print("="*80)
    print("RAG QUERY SYSTEM - INTERACTIVE MODE")
    print("="*80)
    print(f"Documents loaded: {doc_count}")
    print(f"Document folder: {RAG_DOCUMENTS_FOLDER}\n")
    print(f"Chroma directory: {CHROMA_PERSIST_DIRECTORY}\n")
    
    if doc_count == 0:
        print("⚠️  No documents loaded yet!")
        print("   Add PDFs to the folder above and restart this script.\n")
    
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'info' to see stats")
    print("  - Type 'help' for more info")
    print("="*80 + "\n")
    
    while True:
        try:
            question = input("Ask a question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'info':
                try:
                    doc_count = vector_db._collection.count()
                except:
                    doc_count = 0
                print(f"\nDocuments: {doc_count}")
                print(f"Folder: {RAG_DOCUMENTS_FOLDER}\n")
                continue
            
            if question.lower() == 'help':
                print("\nTo add documents:")
                print(f"  1. Copy PDF files to: {RAG_DOCUMENTS_FOLDER}")
                print("  2. Restart this script (it auto-detects new files)\n")
                continue
            
            # Query the RAG system
            print("\nProcessing your question...")
            answer = query_rag(question)
            print("\n")
            print(answer)
            print()
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n✗ Error: {str(e)}\n")

if __name__ == "__main__":
    main()
