from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.models.llm import get_llm
from app.vectorstore.chroma_client import get_chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from .state import AgentState

import re 

from app.middleware.context_limit import ContextLimitMiddleware
from app.middleware.latency import LatencyMiddleware
from app.middleware.prompt_firewall import PromptFirewall
from app.middleware.step_counter import StepCounterMiddleware

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

    return {
        "question": state["question"],
        "refined_question": refined.strip(),
        "documents": [],
        "answer": ""
    }

@latency
def node_retrieve(state: AgentState) -> AgentState:
    query = state.get("refined_question", state["question"])
    results = db.similarity_search(query, k=15)

    if not results:
        return {
            "question": state["question"],
            "refined_question": query,
            "documents": [],
            "answer": "No documents found."
        }

    pairs = [(query, d.page_content) for d in results]
    scores = reranker.score(pairs)
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    top_docs: List[Document] = [d for d, s in ranked[:5]]

    return {
        "question": state["question"],
        "refined_question": query,
        "documents": top_docs,
        "answer": ""
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

    if not state["documents"]:
        return state

    context = "\n\n".join([d.page_content for d in state["documents"]])

    prompt = ChatPromptTemplate.from_template("""
You are a validation agent.

Check if the context contains ANY information that could possibly help answer the question.

Relaxed rules:
- If context contains ANY related keywords, legal clauses, rights, permissions, obligations, or affiliate/marketing related content → return "VALID".
- ONLY return "INVALID" if the context is totally unrelated.

Context:
{context}

Question:
{question}

Is the answer possible? (VALID / INVALID)
""")

    chain = prompt | llm | StrOutputParser()
    status = chain.invoke({
        "context": context,
        "question": state["refined_question"]
    }).strip().upper()

    if "INVALID" in status:
        return { **state, "answer": None }

    return state


@latency
def node_generate(state: AgentState) -> AgentState:
    if state.get("answer"):
        return state

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
