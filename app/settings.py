import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    EMBEDDING_MODEL: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CHROMA_DB_PATH: str 
    LLM_MODEL: str = "gemini-2.5-flash"  

    class Config:
        env_file = ".env"

settings = Settings()
