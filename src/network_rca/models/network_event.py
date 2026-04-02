"""Core data models for network events, anomalies, and RCA reports."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    LATENCY = "latency"
    PACKET_LOSS = "packet_loss"
    BANDWIDTH_UTILIZATION = "bandwidth_utilization"
    ERROR_RATE = "error_rate"
    JITTER = "jitter"
    THROUGHPUT = "throughput"
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class NetworkMetric(BaseModel):
    """A single network metric observation."""

    timestamp: datetime
    device_id: str
    interface: str
    metric_type: MetricType
    value: float
    unit: str
    tags: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": False}


class Anomaly(BaseModel):
    """An anomaly detected from one or more metrics."""

    anomaly_id: str
    detected_at: datetime
    device_id: str
    interface: str
    metric_type: MetricType
    observed_value: float
    baseline_value: float
    deviation_score: float  # z-score or similar
    severity: Severity
    description: str
    raw_metrics: list[NetworkMetric] = Field(default_factory=list)

    @property
    def deviation_pct(self) -> float:
        if self.baseline_value == 0:
            return 0.0
        return abs(self.observed_value - self.baseline_value) / abs(self.baseline_value) * 100


class RootCause(BaseModel):
    """A hypothesised root cause for one or more anomalies."""

    cause_id: str
    category: str  # e.g. "hardware_failure", "congestion", "misconfiguration"
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    affected_devices: list[str] = Field(default_factory=list)
    affected_interfaces: list[str] = Field(default_factory=list)
    supporting_anomalies: list[str] = Field(default_factory=list)  # anomaly_ids
    recommended_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RCAReport(BaseModel):
    """Full RCA report produced by the pipeline."""

    report_id: str
    generated_at: datetime
    analysis_window_start: datetime
    analysis_window_end: datetime
    total_metrics_analyzed: int
    anomalies_detected: list[Anomaly] = Field(default_factory=list)
    root_causes: list[RootCause] = Field(default_factory=list)
    summary: str = ""
    severity: Severity = Severity.INFO

    @property
    def top_cause(self) -> RootCause | None:
        if not self.root_causes:
            return None
        return max(self.root_causes, key=lambda c: c.confidence)
