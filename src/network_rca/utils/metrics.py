"""Utility helpers for statistical analysis of network metrics."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Sequence

from network_rca.models.network_event import MetricType, NetworkMetric


# Normal operating thresholds per metric type (unit-agnostic, relative)
METRIC_THRESHOLDS: dict[MetricType, dict[str, float]] = {
    MetricType.LATENCY: {"warning": 100.0, "critical": 300.0, "unit": "ms"},
    MetricType.PACKET_LOSS: {"warning": 1.0, "critical": 5.0, "unit": "%"},
    MetricType.BANDWIDTH_UTILIZATION: {"warning": 70.0, "critical": 90.0, "unit": "%"},
    MetricType.ERROR_RATE: {"warning": 0.5, "critical": 2.0, "unit": "%"},
    MetricType.JITTER: {"warning": 20.0, "critical": 50.0, "unit": "ms"},
    MetricType.THROUGHPUT: {"warning": 0.0, "critical": 0.0, "unit": "Mbps"},  # low-is-bad
    MetricType.CPU_UTILIZATION: {"warning": 70.0, "critical": 90.0, "unit": "%"},
    MetricType.MEMORY_UTILIZATION: {"warning": 75.0, "critical": 90.0, "unit": "%"},
}


def compute_baseline(values: Sequence[float]) -> tuple[float, float]:
    """Return (mean, stdev) for a sequence of values."""
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def z_score(value: float, mean: float, stdev: float) -> float:
    """Compute z-score; returns 0 if stdev is 0."""
    if stdev == 0:
        return 0.0
    return (value - mean) / stdev


def group_metrics_by_key(
    metrics: list[NetworkMetric],
) -> dict[tuple[str, str, MetricType], list[NetworkMetric]]:
    """Group metrics by (device_id, interface, metric_type)."""
    groups: dict[tuple[str, str, MetricType], list[NetworkMetric]] = defaultdict(list)
    for m in metrics:
        groups[(m.device_id, m.interface, m.metric_type)].append(m)
    for key in groups:
        groups[key].sort(key=lambda x: x.timestamp)
    return dict(groups)


def sliding_window_stats(
    values: list[float], window: int = 10
) -> list[tuple[float, float]]:
    """Compute per-point (mean, stdev) using a trailing window of *window* samples."""
    result: list[tuple[float, float]] = []
    for i, v in enumerate(values):
        start = max(0, i - window)
        window_vals = values[start:i] if i > 0 else [v]
        mean, stdev = compute_baseline(window_vals)
        result.append((mean, stdev))
    return result


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def safe_div(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    if denominator == 0:
        return fallback
    return numerator / denominator


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient between two equal-length sequences."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)
