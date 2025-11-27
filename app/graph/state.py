from typing import List, TypedDict
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    refined_question: str
    documents: List[Document]
    answer: str
    pii_handled: bool = False
