from langchain_huggingface import HuggingFaceEmbeddings
from app.settings import settings

def get_embeddings():
    """Return embedding model instance"""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
