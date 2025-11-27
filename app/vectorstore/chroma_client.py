import os
from langchain_chroma import Chroma
from app.models.embeddings import get_embeddings
from app.settings import settings

def get_chroma():
    """Return Chroma vectorstore instance"""
    
    os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
    return Chroma(
        collection_name="multi_agent_rag_collection",
        persist_directory=settings.CHROMA_DB_PATH,
        embedding_function=get_embeddings()
    )
