from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1100,
    chunk_overlap=350,
    separators=["\n\n", "\n", " ", ""]
)

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


def create_chunks() -> List[Document]:
    chunks = get_cleaned_chunks()
    print(f"[CHUNK AGENT] Created {len(chunks)} chunks.")
    return chunks


