"""
Log Analysis Agent
------------------
First step in the multi-agent RCA pipeline.

Role:  Receive raw log text plus retrieved context, then produce a concise
       structured summary of detected anomalies and evidence.
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a senior network operations analyst.
Your job is to analyse network telemetry logs and identify anomalies,
patterns, and relevant evidence that may indicate a failure or degradation.

Given raw log data and retrieved context, produce a concise structured summary:
- List the key anomalies observed (bullet points).
- Note affected devices, interfaces, and protocols.
- Highlight any temporal patterns or correlated events.
- Keep your response factual and evidence-based.
"""


def run_log_analysis_agent(
    raw_logs: str,
    context_chunks: list[str],
    llm: ChatOpenAI | None = None,
) -> str:
    """Analyse raw logs using retrieved context.

    Args:
        raw_logs:       Free-text or structured log input from the user.
        context_chunks: Relevant log snippets retrieved from the vector store.
        llm:            Optional pre-built ChatOpenAI instance (injectable for testing).

    Returns:
        A string containing the anomaly summary produced by the agent.
    """
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
        )

    context_text = "\n\n".join(context_chunks) if context_chunks else "No additional context."

    user_content = (
        f"## Raw Input Logs\n{raw_logs}\n\n"
        f"## Retrieved Context (from knowledge base)\n{context_text}\n\n"
        "Please analyse these logs and summarise the key anomalies."
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    response = llm.invoke(messages)
    result = response.content.strip()
    logger.debug("Log Analysis Agent output: %s", result[:200])
    return result
