"""
Root Cause Analysis Agent
--------------------------
Second step in the multi-agent RCA pipeline.

Role:  Take the anomaly summary produced by the Log Analysis Agent and infer
       the most probable root cause(s) with a confidence score.
"""

from __future__ import annotations

import json
import logging
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert network reliability engineer specialised in root cause analysis.

Given a structured anomaly summary from a log analyst, determine the single most probable root cause
of the observed network failure or degradation.

Respond ONLY with a valid JSON object using this exact schema:
{
  "root_cause": "<concise description of the root cause>",
  "confidence": <float between 0.0 and 1.0>,
  "evidence": ["<evidence point 1>", "<evidence point 2>", ...]
}

Rules:
- confidence must reflect how certain you are given the evidence (0.5 = uncertain, 0.9 = very confident).
- root_cause must be a single concise sentence (under 15 words).
- evidence must list 2-5 specific facts from the anomaly summary that support the conclusion.
- Do NOT include markdown fences, only raw JSON.
"""


def run_rca_agent(
    anomaly_summary: str,
    llm: ChatOpenAI | None = None,
) -> dict:
    """Infer root cause from the anomaly summary.

    Args:
        anomaly_summary: Output from the Log Analysis Agent.
        llm:             Optional pre-built ChatOpenAI instance.

    Returns:
        Dictionary with keys: ``root_cause``, ``confidence``, ``evidence``.
    """
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.0,
        )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"## Anomaly Summary\n{anomaly_summary}\n\n"
                "Based on this summary, determine the root cause."
            )
        ),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if the model returns them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("RCA Agent returned non-JSON output; falling back to defaults.")
        result = {
            "root_cause": "Unable to determine root cause",
            "confidence": 0.0,
            "evidence": [raw[:200]],
        }

    # Clamp confidence
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    logger.debug("RCA Agent output: %s", result)
    return result
