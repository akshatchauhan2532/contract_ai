from dotenv import load_dotenv
import os

load_dotenv()  # <--- MUST be called before os.getenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from app.settings import settings

def get_llm():
    """Return LLM model instance"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google GenAI API key not found. Set GOOGLE_API_KEY in your .env")
    
    return ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0,
        max_output_tokens=1024,
        api_key=api_key
    )
