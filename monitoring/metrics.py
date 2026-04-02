"""
Prometheus-style Metrics
------------------------
Exposes request counters and latency histograms for the /analyze_network
endpoint via the prometheus-client library.

A /metrics HTTP endpoint is mounted on the FastAPI app so Prometheus can
scrape it.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "rca_request_total",
    "Total number of /analyze_network requests received.",
    ["status"],  # label values: "success" | "error"
)

REQUEST_LATENCY = Histogram(
    "rca_request_latency_seconds",
    "End-to-end latency of /analyze_network requests in seconds.",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

PIPELINE_ERRORS = Counter(
    "rca_pipeline_errors_total",
    "Total number of RCA pipeline failures (non-HTTP errors).",
)

AGENT_INVOCATIONS = Counter(
    "rca_agent_invocations_total",
    "Total LLM agent invocations.",
    ["agent"],  # label values: "log_analysis" | "rca" | "remediation"
)

CONFIDENCE_HISTOGRAM = Histogram(
    "rca_confidence_score",
    "Distribution of confidence scores returned by the RCA agent.",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_metrics_output() -> tuple[bytes, str]:
    """Return the current Prometheus metrics payload and content-type header.

    Returns:
        A tuple of (bytes payload, content-type string).
    """
    return generate_latest(), CONTENT_TYPE_LATEST
