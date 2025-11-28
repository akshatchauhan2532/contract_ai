from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from copy import deepcopy

from app.models.llm import get_llm
from app.vectorstore.chroma_client import get_chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from .state import AgentState
import re 

from app.middleware.context_limit import ContextLimitMiddleware
from app.middleware.latency import LatencyMiddleware
from app.middleware.prompt_firewall import PromptFirewall
from app.middleware.step_counter import StepCounterMiddleware
from app.retrieval.bm25_store import bm25

llm = get_llm()
db = get_chroma()
reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


context_limit = ContextLimitMiddleware().wrap
latency = LatencyMiddleware().wrap
step_counter = StepCounterMiddleware().wrap
prompt_firewall = PromptFirewall().wrap

@latency
@prompt_firewall
def node_reason(state: AgentState) -> AgentState:
    prompt = ChatPromptTemplate.from_template("""
You are a reasoning agent.
Rewrite the user question in a clearer and more search-optimized form.

Rules:
- Keep original meaning.
- Be concise.
- DO NOT answer the question; only rewrite it.

User Question: {question}

Refined Question:
""")

    chain = prompt | llm | StrOutputParser()
    refined = chain.invoke({"question": state["question"]})
    print("refined question:", refined.strip())

    return {
        "question": state["question"],
        "refined_question": refined.strip(),
        "documents": [],
        "answer": ""
    }

@latency
@latency
def node_retrieve(state: AgentState) -> AgentState:
    query = state.get("refined_question") or state["question"]

    vector_results = db.similarity_search(query, k=12)
    bm25_results = bm25.search(query, k=12)

    def doc_key(d: Document) -> str:
        src = d.metadata.get("source", "unknown")
        page = str(d.metadata.get("page", ""))
        head = d.page_content[:80]
        return f"{src}|{page}|{head}"

    combined: dict[str, Document] = {}
    for d in vector_results + bm25_results:
        k = doc_key(d)
        combined[k] = d

    merged_docs = list(combined.values())

    if not merged_docs:
        return {
            **state,
            "documents": [],
            "answer": "No documents found.",
        }

    pairs = [(query, d.page_content) for d in merged_docs]

    try:
        scores = reranker.score(pairs)
    except Exception as e:
        print("[RERANKER] Error while scoring:", e)
        scores = [0.0] * len(merged_docs)

    ranked = sorted(zip(merged_docs, scores), key=lambda x: x[1], reverse=True)
    top_docs: List[Document] = [d for d, _ in ranked[:5]]

    
    print("\n--- DEBUG RETRIEVED DOCS ---")
    for d in top_docs:
        print(f"[DOC] Source={d.metadata.get('source')} Page={d.metadata.get('page')}")
        print(f"Content Preview: {d.page_content[:200]}")
        print("--------------------------------------------------")

    return {
        **state,                     
        "refined_question": query,
        "documents": deepcopy(top_docs),
        "answer": "",                 
    }


@latency
def node_pii_redact(state: AgentState) -> AgentState:
    """
    Redacts sensitive information from documents, the original question,
    and the refined question using enterprise-level PII patterns.
    """
    # Enterprise-level PII patterns
    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",         # SSN
        r"\b\d{16}\b",                     # Credit card number (simple)
        r"\b\d{3}-\d{3}-\d{4}\b",          # Phone number
        r"\w+@\w+\.\w+",                    # Email
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"     # Date patterns (MM/DD/YYYY or DD/MM/YYYY)
    ]

    # Function to apply all patterns
    def redact_text(text: str) -> str:
        for pattern in pii_patterns:
            text = re.sub(pattern, "[REDACTED]", text)
        return text

    # Redact documents
    docs = state.get("documents", [])
    for d in docs:
        d.page_content = redact_text(d.page_content)

    # Redact user question
    if "question" in state and state["question"]:
        state["question"] = redact_text(state["question"])

    # Redact refined question
    if "refined_question" in state and state["refined_question"]:
        state["refined_question"] = redact_text(state["refined_question"])

    state["pii_handled"] = True
    print("[PII] Sensitive info redacted from documents and questions")
    return state

@latency
def node_validate(state: AgentState) -> AgentState:
    """
    Enterprise-grade validation node for service-related questions.

    This node checks if the retrieved context is relevant to the user's question.
    Returns 'VALID' if context mentions service, delivery, obligations, or related concepts.
    Otherwise, it sets the answer to None.
    """

    docs = state.get("documents", [])
    if not docs:
        state["answer"] = None
        return state

    # Build context from all documents, remove empty pages
    context = "\n\n".join(d.page_content.strip() for d in docs if getattr(d, "page_content", "").strip())

    # Skip if context is empty
    if not context:
        state["answer"] = None
        return state

    # Enterprise prompt: deterministic single-word output
    prompt = ChatPromptTemplate.from_template("""
You are a validation agent for legal/service-related content.

Your task:
- Only respond with ONE word: VALID or INVALID
- Do NOT add any explanation or extra text.
- VALID if ANY part of the context mentions service, delivery, performance,
  obligations, contractual duties, logistics, vendors, responsibilities,
  inventory, technology, support, or any similar enterprise/service term.
- INVALID only if the context is completely unrelated to service or contractual obligations.

Context:
{context}

Question:
{question}

VALID or INVALID:
""")

    # Invoke LLM
    result = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": state["refined_question"]
    }).strip().upper()

    # Ensure only 'VALID' is accepted, everything else → INVALID
    if result != "VALID":
        state["answer"] = None

    return state



@latency
def node_generate(state: AgentState) -> AgentState:
    existing_answer = state.get("answer")
    if isinstance(existing_answer, str) and existing_answer.strip():
        return state

    if not state.get("documents"):
        return {
            **state,
            "answer": "No relevant documents were available to answer this question.",
        }

    context_text = "\n\n".join([d.page_content for d in state["documents"]])

    prompt = ChatPromptTemplate.from_template("""
You are a contract analysis agent.

Answer the question using ONLY the provided documents.

Rules:
- Use exact quotes when possible.
- Do NOT hallucinate.
- If answer is not available → say: "Information not found in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context_text,
        "question": state["refined_question"]
    })

    return {
        **state,
        "answer": answer.strip()
    }
