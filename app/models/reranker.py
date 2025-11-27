from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from app.settings import settings

def get_reranker():
    """Return CrossEncoder reranker instance"""
    model = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
    return CrossEncoderReranker(model=model, top_n=3)
