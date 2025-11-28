from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os
import json

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1100,
    chunk_overlap=350,
    separators=["\n\n", "\n", " ", ""]
)

CHUNK_DIR = "data/chunks"

def load_documents_from_dir(directory: str = 'data/pdfs') -> List[Document]:
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    all_docs = []
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {directory}.")
        return []

    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        all_docs.extend(docs)
            
    return all_docs


def get_chunks(documents: List[Document]) -> List[Document]:
    if not documents:
        return []
    return TEXT_SPLITTER.split_documents(documents)

def get_cleaned_chunks(directory: str = 'data/pdfs') -> List[Document]:
    documents = load_documents_from_dir(directory)
    return get_chunks(documents)

def save_chunks_to_folder(chunks: List[Document]):
    os.makedirs(CHUNK_DIR, exist_ok=True)

    # Clear old chunks
    for old in os.listdir(CHUNK_DIR):
        os.remove(os.path.join(CHUNK_DIR, old))

    for i, doc in enumerate(chunks):
        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i:05}.json")

        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }, f, ensure_ascii=False, indent=2)

    print(f"[CHUNKS] Saved {len(chunks)} chunks → {CHUNK_DIR}")



def create_chunks() -> List[Document]:
    chunks = get_cleaned_chunks()
    print(f"[CHUNK AGENT] Created {len(chunks)} chunks.")
    save_chunks_to_folder(chunks)
    return chunks


