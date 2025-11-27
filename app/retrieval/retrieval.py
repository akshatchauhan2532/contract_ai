from langchain_core.tools import tool
from app.vectorstore.chroma_client import get_chroma
from app.models.reranker import get_reranker

@tool("retrieve_documents", return_direct=False)
def retrieve_tool(query: str):
    """Retrieve top documents from vectorstore and rerank them"""
    vs = get_chroma()
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    docs = retriever.invoke(query)

    reranker = get_reranker()
    reranked = reranker.compress_documents(query, docs)

    return [d.page_content for d in reranked]
