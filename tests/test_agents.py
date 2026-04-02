"""Tests for agent modules using mocked LLMs."""

from unittest.mock import MagicMock, patch

import pytest

from agents.log_analysis_agent import run_log_analysis_agent
from agents.root_cause_agent import run_rca_agent
from agents.remediation_agent import run_remediation_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm(content: str) -> MagicMock:
    """Return a ChatOpenAI mock that always returns *content*."""
    llm = MagicMock()
    response = MagicMock()
    response.content = content
    llm.invoke.return_value = response
    return llm


# ---------------------------------------------------------------------------
# Log Analysis Agent
# ---------------------------------------------------------------------------

def test_log_analysis_agent_returns_string():
    llm = _mock_llm("• Packet drop on R17\n• OSPF flapping")
    result = run_log_analysis_agent(
        raw_logs="router R17 packet loss",
        context_chunks=["[2024-01-15] Device: router-R17 | Event: packet_drop"],
        llm=llm,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_log_analysis_agent_uses_llm():
    llm = _mock_llm("anomaly summary")
    run_log_analysis_agent("some logs", [], llm=llm)
    llm.invoke.assert_called_once()


def test_log_analysis_agent_empty_context():
    """Should still work with no context chunks."""
    llm = _mock_llm("No anomalies detected.")
    result = run_log_analysis_agent("normal traffic", [], llm=llm)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# RCA Agent
# ---------------------------------------------------------------------------

RCA_JSON = '{"root_cause": "routing loop congestion", "confidence": 0.87, "evidence": ["OSPF flapping"]}'


def test_rca_agent_valid_json():
    llm = _mock_llm(RCA_JSON)
    result = run_rca_agent("anomaly summary", llm=llm)
    assert result["root_cause"] == "routing loop congestion"
    assert result["confidence"] == pytest.approx(0.87)
    assert isinstance(result["evidence"], list)


def test_rca_agent_clamps_confidence():
    """Confidence above 1.0 should be clamped to 1.0."""
    llm = _mock_llm('{"root_cause": "congestion", "confidence": 1.5, "evidence": []}')
    result = run_rca_agent("summary", llm=llm)
    assert result["confidence"] <= 1.0


def test_rca_agent_strips_markdown_fences():
    """Agent should handle markdown-fenced JSON from the LLM."""
    fenced = "```json\n" + RCA_JSON + "\n```"
    llm = _mock_llm(fenced)
    result = run_rca_agent("summary", llm=llm)
    assert result["root_cause"] == "routing loop congestion"


def test_rca_agent_fallback_on_invalid_json():
    """Non-JSON response should return a fallback dict without raising."""
    llm = _mock_llm("I cannot determine the root cause.")
    result = run_rca_agent("summary", llm=llm)
    assert "root_cause" in result
    assert result["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Remediation Agent
# ---------------------------------------------------------------------------

REM_JSON = (
    '{"recommended_fix": "restart OSPF process on R17", '
    '"steps": ["ssh to R17", "clear ip ospf process"], '
    '"risk_level": "low", "estimated_resolution_time": "5 minutes"}'
)

RCA_RESULT = {
    "root_cause": "routing loop congestion",
    "confidence": 0.87,
    "evidence": ["OSPF flapping between R17 and R22"],
}


def test_remediation_agent_valid_json():
    llm = _mock_llm(REM_JSON)
    result = run_remediation_agent(RCA_RESULT, llm=llm)
    assert "restart OSPF" in result["recommended_fix"]
    assert isinstance(result["steps"], list)
    assert len(result["steps"]) == 2
    assert result["risk_level"] == "low"


def test_remediation_agent_fallback_on_invalid_json():
    llm = _mock_llm("Please restart manually.")
    result = run_remediation_agent(RCA_RESULT, llm=llm)
    assert "recommended_fix" in result
    assert isinstance(result["steps"], list)


def test_remediation_agent_strips_markdown_fences():
    fenced = "```\n" + REM_JSON + "\n```"
    llm = _mock_llm(fenced)
    result = run_remediation_agent(RCA_RESULT, llm=llm)
    assert "restart OSPF" in result["recommended_fix"]
