"""
Remediation Agent
-----------------
Third (final) step in the multi-agent RCA pipeline.

Role:  Given the root cause analysis, propose a concrete, actionable
       remediation step that an on-call engineer can execute immediately.
"""

from __future__ import annotations

import json
import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a senior network engineer who specialises in incident remediation.

Given a root cause analysis (root cause + evidence), propose the single most effective
remediation action that will resolve the incident quickly with minimal risk.

Respond ONLY with a valid JSON object using this exact schema:
{
  "recommended_fix": "<concise, actionable fix in one or two sentences>",
  "steps": ["<step 1>", "<step 2>", ...],
  "risk_level": "<low | medium | high>",
  "estimated_resolution_time": "<e.g. 5 minutes | 30 minutes>"
}

Rules:
- recommended_fix must be actionable (a real command, restart procedure, or configuration change).
- steps must list 2-5 numbered actions in execution order.
- risk_level must be low, medium, or high.
- Do NOT include markdown fences, only raw JSON.
"""


def run_remediation_agent(
    rca_result: dict,
    llm: ChatOpenAI | None = None,
) -> dict:
    """Propose a remediation plan for the identified root cause.

    Args:
        rca_result: Output from the RCA Agent containing ``root_cause``,
                    ``confidence``, and ``evidence``.
        llm:        Optional pre-built ChatOpenAI instance.

    Returns:
        Dictionary with keys: ``recommended_fix``, ``steps``,
        ``risk_level``, ``estimated_resolution_time``.
    """
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
        )

    rca_text = (
        f"Root Cause: {rca_result.get('root_cause', 'unknown')}\n"
        f"Confidence: {rca_result.get('confidence', 0.0):.2f}\n"
        f"Evidence:\n" + "\n".join(f"  - {e}" for e in rca_result.get("evidence", []))
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Root Cause Analysis\n{rca_text}\n\n"
                "Propose the remediation action."
            )
        ),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if returned despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Remediation Agent returned non-JSON; falling back to defaults.")
        result = {
            "recommended_fix": "Manual investigation required",
            "steps": ["Review logs manually", "Escalate to network team"],
            "risk_level": "medium",
            "estimated_resolution_time": "unknown",
        }

    logger.debug("Remediation Agent output: %s", result)
    return result
