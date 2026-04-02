"""
LangGraph Multi-Agent RCA Workflow
------------------------------------
Chains three agents in sequence:
  1. Log Analysis Agent  – summarise anomalies
  2. RCA Agent           – identify root cause + confidence
  3. Remediation Agent   – propose a fix

The graph state flows through all three nodes and returns a consolidated
``RCAResult`` dict from the final node.
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.log_analysis_agent import run_log_analysis_agent
from agents.remediation_agent import run_remediation_agent
from agents.root_cause_agent import run_rca_agent
from rag.retrieval import retrieve_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class RCAState(TypedDict, total=False):
    """Shared state passed between nodes in the LangGraph workflow."""

    raw_logs: str                       # user-provided input
    context_chunks: list[str]           # RAG-retrieved context
    anomaly_summary: str                # output of Log Analysis Agent
    rca_result: dict[str, Any]          # output of RCA Agent
    remediation_result: dict[str, Any]  # output of Remediation Agent


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _retrieve_node(state: RCAState) -> RCAState:
    """Retrieve relevant context from the vector store."""
    raw_logs = state["raw_logs"]
    chunks = retrieve_context(raw_logs)
    logger.debug("Retrieved %d context chunks.", len(chunks))
    return {**state, "context_chunks": chunks}


def _log_analysis_node(state: RCAState) -> RCAState:
    """Run the Log Analysis Agent."""
    summary = run_log_analysis_agent(
        raw_logs=state["raw_logs"],
        context_chunks=state.get("context_chunks", []),
    )
    logger.debug("Anomaly summary produced.")
    return {**state, "anomaly_summary": summary}


def _rca_node(state: RCAState) -> RCAState:
    """Run the Root Cause Analysis Agent."""
    rca = run_rca_agent(anomaly_summary=state["anomaly_summary"])
    logger.debug("Root cause identified: %s", rca.get("root_cause"))
    return {**state, "rca_result": rca}


def _remediation_node(state: RCAState) -> RCAState:
    """Run the Remediation Agent."""
    remediation = run_remediation_agent(rca_result=state["rca_result"])
    logger.debug("Remediation proposed: %s", remediation.get("recommended_fix"))
    return {**state, "remediation_result": remediation}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_rca_graph() -> StateGraph:
    """Build and compile the LangGraph RCA workflow.

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    graph = StateGraph(RCAState)

    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("log_analysis", _log_analysis_node)
    graph.add_node("rca", _rca_node)
    graph.add_node("remediation", _remediation_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "log_analysis")
    graph.add_edge("log_analysis", "rca")
    graph.add_edge("rca", "remediation")
    graph.add_edge("remediation", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

_app = None  # module-level compiled graph cache


def run_rca_pipeline(raw_logs: str) -> dict[str, Any]:
    """Execute the full multi-agent RCA pipeline.

    Args:
        raw_logs: User-provided log text describing the network event.

    Returns:
        A dictionary with keys:
        - ``root_cause``        (str)
        - ``confidence``        (float)
        - ``recommended_fix``   (str)
        - ``anomaly_summary``   (str)
        - ``evidence``          (list[str])
        - ``remediation_steps`` (list[str])
        - ``risk_level``        (str)
    """
    global _app
    if _app is None:
        _app = build_rca_graph()

    initial_state: RCAState = {"raw_logs": raw_logs}
    final_state: RCAState = _app.invoke(initial_state)

    rca = final_state.get("rca_result", {})
    rem = final_state.get("remediation_result", {})

    return {
        "root_cause": rca.get("root_cause", "unknown"),
        "confidence": round(float(rca.get("confidence", 0.0)), 4),
        "recommended_fix": rem.get("recommended_fix", "No fix proposed"),
        "anomaly_summary": final_state.get("anomaly_summary", ""),
        "evidence": rca.get("evidence", []),
        "remediation_steps": rem.get("steps", []),
        "risk_level": rem.get("risk_level", "unknown"),
        "estimated_resolution_time": rem.get("estimated_resolution_time", "unknown"),
    }
