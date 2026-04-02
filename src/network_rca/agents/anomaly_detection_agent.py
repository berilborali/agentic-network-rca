"""Anomaly Detection Agent — identifies statistical deviations in metrics."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from network_rca.agents.base_agent import BaseAgent
from network_rca.models.network_event import (
    Anomaly,
    MetricType,
    NetworkMetric,
    Severity,
)
from network_rca.utils.metrics import (
    METRIC_THRESHOLDS,
    group_metrics_by_key,
    sliding_window_stats,
    z_score,
)


# Absolute thresholds above which we always flag (overrides z-score gate)
_ABSOLUTE_UPPER: dict[MetricType, float] = {
    MetricType.LATENCY: 100.0,           # ms
    MetricType.PACKET_LOSS: 1.0,          # %
    MetricType.BANDWIDTH_UTILIZATION: 70.0,  # %
    MetricType.ERROR_RATE: 0.5,           # %
    MetricType.JITTER: 20.0,             # ms
    MetricType.CPU_UTILIZATION: 70.0,     # %
    MetricType.MEMORY_UTILIZATION: 75.0,  # %
}

_ABSOLUTE_LOWER: dict[MetricType, float] = {
    MetricType.THROUGHPUT: 100.0,  # Mbps — below this is suspicious
}


def _severity_from_z(z: float, metric_type: MetricType, value: float) -> Severity:
    thresholds = METRIC_THRESHOLDS.get(metric_type, {})
    critical_val = thresholds.get("critical", float("inf"))
    warning_val = thresholds.get("warning", float("inf"))

    # For throughput, low is bad
    if metric_type == MetricType.THROUGHPUT:
        if value < 50:
            return Severity.CRITICAL
        if value < 100:
            return Severity.HIGH
        return Severity.MEDIUM

    if value >= critical_val:
        return Severity.CRITICAL
    if value >= warning_val:
        return Severity.HIGH
    if abs(z) >= 4:
        return Severity.HIGH
    if abs(z) >= 3:
        return Severity.MEDIUM
    return Severity.LOW


class AnomalyDetectionAgent(BaseAgent):
    """Detects anomalies in ingested metrics using sliding-window z-scores.

    Configuration keys
    ------------------
    z_threshold : float
        Minimum absolute z-score to flag a data point (default: 2.5).
    window_size : int
        Sliding window length for baseline estimation (default: 10).
    min_samples : int
        Minimum samples required before scoring (default: 5).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("anomaly_detection", config)

    # ------------------------------------------------------------------
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        metrics: list[NetworkMetric] = context.get("metrics", [])
        if not metrics:
            self.logger.warning("No metrics to analyse")
            context["anomalies"] = []
            return context

        z_threshold: float = self.config.get("z_threshold", 2.5)
        window_size: int = self.config.get("window_size", 10)
        min_samples: int = self.config.get("min_samples", 5)

        groups = group_metrics_by_key(metrics)
        anomalies: list[Anomaly] = []

        for (device_id, interface, metric_type), group_metrics in groups.items():
            if len(group_metrics) < min_samples:
                continue

            values = [m.value for m in group_metrics]
            stats = sliding_window_stats(values, window=window_size)

            for i, (metric, (mean, stdev)) in enumerate(
                zip(group_metrics, stats)
            ):
                z = z_score(metric.value, mean, stdev)
                above_abs_upper = (
                    metric_type in _ABSOLUTE_UPPER
                    and metric.value >= _ABSOLUTE_UPPER[metric_type]
                )
                below_abs_lower = (
                    metric_type in _ABSOLUTE_LOWER
                    and metric.value < _ABSOLUTE_LOWER[metric_type]
                )
                is_statistical_anomaly = abs(z) >= z_threshold
                if not (above_abs_upper or below_abs_lower or is_statistical_anomaly):
                    continue

                severity = _severity_from_z(z, metric_type, metric.value)
                if severity == Severity.LOW and not is_statistical_anomaly:
                    continue  # skip low-severity absolute breaches to reduce noise

                anomaly = Anomaly(
                    anomaly_id=str(uuid.uuid4()),
                    detected_at=datetime.now(tz=timezone.utc),
                    device_id=device_id,
                    interface=interface,
                    metric_type=metric_type,
                    observed_value=metric.value,
                    baseline_value=mean,
                    deviation_score=z,
                    severity=severity,
                    description=(
                        f"{metric_type.value} on {device_id}/{interface} "
                        f"reached {metric.value:.2f} "
                        f"(baseline {mean:.2f}, z={z:.2f})"
                    ),
                    raw_metrics=[metric],
                )
                anomalies.append(anomaly)

        # De-duplicate: keep worst anomaly per (device, interface, metric_type)
        anomalies = self._deduplicate(anomalies)
        self.logger.info("Detected %d anomalies", len(anomalies))
        context["anomalies"] = anomalies
        return context

    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate(anomalies: list[Anomaly]) -> list[Anomaly]:
        """Keep the highest-severity anomaly per (device, interface, metric)."""
        best: dict[tuple[str, str, MetricType], Anomaly] = {}
        sev_order = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
            Severity.INFO: 0,
        }
        for a in anomalies:
            key = (a.device_id, a.interface, a.metric_type)
            existing = best.get(key)
            if existing is None or sev_order[a.severity] > sev_order[existing.severity]:
                best[key] = a
        return list(best.values())
