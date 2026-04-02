"""
FastAPI Server
--------------
Exposes the /analyze_network endpoint and a Prometheus /metrics endpoint.

Environment variables
---------------------
OPENAI_API_KEY   – required for LLM calls
HOST             – bind host (default: 0.0.0.0)
PORT             – bind port  (default: 8000)
LOG_LEVEL        – Python logging level (default: INFO)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

# Configure root logger before importing local modules that use it.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

from agents.rca_workflow import run_rca_pipeline  # noqa: E402 – after load_dotenv
from monitoring.metrics import (  # noqa: E402
    AGENT_INVOCATIONS,
    CONFIDENCE_HISTOGRAM,
    PIPELINE_ERRORS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    get_metrics_output,
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Input payload for /analyze_network."""

    logs: str = Field(
        ...,
        min_length=3,
        description="Raw network log text or description of the observed issue.",
        examples=["router R17 experiencing packet loss"],
    )


class AnalyzeResponse(BaseModel):
    """Output payload from /analyze_network."""

    root_cause: str = Field(..., description="Inferred root cause of the network issue.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0–1).")
    recommended_fix: str = Field(..., description="Actionable remediation step.")
    anomaly_summary: str = Field(..., description="Summary of detected anomalies.")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence.")
    remediation_steps: list[str] = Field(
        default_factory=list, description="Ordered remediation steps."
    )
    risk_level: str = Field(..., description="Risk level of the remediation: low|medium|high.")
    estimated_resolution_time: str = Field(..., description="Estimated time to resolve.")


# ---------------------------------------------------------------------------
# Lifespan – warm up the RAG index on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the RAG retriever on application startup."""
    logger.info("Starting up – initialising RAG retriever …")
    try:
        from rag.retrieval import _get_manager  # noqa: WPS433 – intentional warm-up

        _get_manager()
        logger.info("RAG retriever ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG warm-up failed (will retry on first request): %s", exc)
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic Network RCA",
    description=(
        "AI-powered network root cause analysis service. "
        "Ingests telemetry logs, retrieves context via RAG, and runs a "
        "multi-agent LangGraph workflow to diagnose failures and recommend fixes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health_check() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    payload, content_type = get_metrics_output()
    return Response(content=payload, media_type=content_type)


@app.post("/analyze_network", response_model=AnalyzeResponse, tags=["rca"])
async def analyze_network(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the multi-agent RCA pipeline on the supplied log text.

    Returns the identified root cause, confidence score, and recommended fix.
    """
    start = time.perf_counter()

    # Track agent invocations for metrics
    AGENT_INVOCATIONS.labels(agent="log_analysis").inc()
    AGENT_INVOCATIONS.labels(agent="rca").inc()
    AGENT_INVOCATIONS.labels(agent="remediation").inc()

    try:
        result = run_rca_pipeline(request.logs)
    except Exception as exc:
        PIPELINE_ERRORS.inc()
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("RCA pipeline failed for input: %r", request.logs[:100])
        raise HTTPException(
            status_code=500,
            detail=f"RCA pipeline error: {exc}",
        ) from exc

    elapsed = time.perf_counter() - start
    REQUEST_LATENCY.observe(elapsed)
    REQUEST_COUNT.labels(status="success").inc()
    CONFIDENCE_HISTOGRAM.observe(result["confidence"])

    logger.info(
        "RCA complete | root_cause=%r confidence=%.2f latency=%.2fs",
        result["root_cause"],
        result["confidence"],
        elapsed,
    )

    return AnalyzeResponse(**result)


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
