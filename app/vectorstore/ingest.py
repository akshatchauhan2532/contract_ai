import os
import shutil
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from app.models.embeddings import get_embeddings
from app.settings import settings

COLLECTION_NAME = "multi_agent_rag_collection"

def create_chroma_collection(chunks: List[Document]) -> Chroma:
    hf_embeddings = get_embeddings()
    print("INGEST PATH:", settings.CHROMA_DB_PATH)

    # Always rebuild DB from chunks
    if os.path.exists(settings.CHROMA_DB_PATH):
        shutil.rmtree(settings.CHROMA_DB_PATH)

    print(f"[CHROMA] Creating new ChromaDB at {settings.CHROMA_DB_PATH}...")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=hf_embeddings,
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )

    db.persist()
    print(f"[CHROMA] Successfully added {len(chunks)} chunks to ChromaDB.")
    return db


if __name__ == "__main__":
    from app.utils.pdf_utils import create_chunks
    chunks = create_chunks()
    create_chroma_collection(chunks)
