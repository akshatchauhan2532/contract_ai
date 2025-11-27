from langgraph.graph import StateGraph, END
from .agents import (
    node_reason,
    node_retrieve,
    node_validate,
    node_generate,
    node_pii_redact
)
from .state import AgentState

from app.middleware.context_limit import ContextLimitMiddleware
from app.middleware.latency import LatencyMiddleware
from app.middleware.prompt_firewall import PromptFirewall
from app.middleware.step_counter import StepCounterMiddleware

def build_workflow():
    middlewares = [
        LatencyMiddleware(),
        PromptFirewall()
    ]
    workflow = StateGraph(AgentState,middlewares=middlewares)

    # Add graph nodes
    workflow.add_node("reason", node_reason)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("pii_redact", node_pii_redact)
    workflow.add_node("validate", node_validate)
    workflow.add_node("generate", node_generate)

    # Entry point
    workflow.set_entry_point("reason")

    # Linear edges
    workflow.add_edge("reason", "retrieve")
    workflow.add_edge("retrieve", "pii_redact")
    workflow.add_edge("pii_redact", "validate")
    workflow.add_edge("validate", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
