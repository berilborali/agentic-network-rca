"""Data Ingestion Agent — loads or simulates network metrics."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from network_rca.agents.base_agent import BaseAgent
from network_rca.models.network_event import MetricType, NetworkMetric


# ---------------------------------------------------------------------------
# Scenario definitions used by the built-in simulator
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, dict[str, Any]] = {
    "congestion": {
        "description": "High bandwidth utilization causing latency and packet loss spikes",
        "metrics": {
            MetricType.BANDWIDTH_UTILIZATION: (95.0, 3.0),
            MetricType.LATENCY: (450.0, 50.0),
            MetricType.PACKET_LOSS: (8.0, 2.0),
        },
    },
    "hardware_failure": {
        "description": "Faulty NIC generating excessive CRC errors",
        "metrics": {
            MetricType.ERROR_RATE: (15.0, 4.0),
            MetricType.PACKET_LOSS: (12.0, 3.0),
            MetricType.THROUGHPUT: (10.0, 5.0),
        },
    },
    "misconfiguration": {
        "description": "MTU mismatch causing fragmentation and elevated jitter",
        "metrics": {
            MetricType.JITTER: (80.0, 15.0),
            MetricType.LATENCY: (200.0, 30.0),
            MetricType.ERROR_RATE: (3.0, 1.0),
        },
    },
    "resource_exhaustion": {
        "description": "Device CPU and memory saturation degrading forwarding performance",
        "metrics": {
            MetricType.CPU_UTILIZATION: (92.0, 4.0),
            MetricType.MEMORY_UTILIZATION: (88.0, 3.0),
            MetricType.LATENCY: (350.0, 60.0),
        },
    },
    "normal": {
        "description": "Healthy baseline traffic — no anomalies",
        "metrics": {
            MetricType.LATENCY: (15.0, 3.0),
            MetricType.PACKET_LOSS: (0.05, 0.02),
            MetricType.BANDWIDTH_UTILIZATION: (30.0, 5.0),
            MetricType.ERROR_RATE: (0.01, 0.005),
            MetricType.JITTER: (3.0, 1.0),
            MetricType.THROUGHPUT: (900.0, 50.0),
            MetricType.CPU_UTILIZATION: (25.0, 5.0),
            MetricType.MEMORY_UTILIZATION: (40.0, 5.0),
        },
    },
}

_METRIC_UNITS: dict[MetricType, str] = {
    MetricType.LATENCY: "ms",
    MetricType.PACKET_LOSS: "%",
    MetricType.BANDWIDTH_UTILIZATION: "%",
    MetricType.ERROR_RATE: "%",
    MetricType.JITTER: "ms",
    MetricType.THROUGHPUT: "Mbps",
    MetricType.CPU_UTILIZATION: "%",
    MetricType.MEMORY_UTILIZATION: "%",
}


class DataIngestionAgent(BaseAgent):
    """Loads network metrics from a JSON file or generates synthetic data.

    Configuration keys
    ------------------
    data_file : str | None
        Path to a JSON file containing serialised ``NetworkMetric`` objects.
        When omitted the agent generates synthetic data.
    scenario : str
        One of ``congestion``, ``hardware_failure``, ``misconfiguration``,
        ``resource_exhaustion``, ``normal``.  Defaults to ``congestion``.
    num_devices : int
        Number of simulated devices (default: 3).
    num_samples : int
        Number of time-series samples per device/metric (default: 60).
    sample_interval_seconds : int
        Seconds between synthetic samples (default: 60).
    seed : int | None
        Optional random seed for reproducibility.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("data_ingestion", config)

    # ------------------------------------------------------------------
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        data_file: str | None = self.config.get("data_file")
        if data_file:
            metrics = self._load_from_file(Path(data_file))
        else:
            metrics = self._simulate()

        self.logger.info("Ingested %d metrics", len(metrics))
        context["metrics"] = metrics
        context["scenario"] = self.config.get("scenario", "congestion")
        return context

    # ------------------------------------------------------------------
    def _load_from_file(self, path: Path) -> list[NetworkMetric]:
        with path.open() as fh:
            raw: list[dict[str, Any]] = json.load(fh)
        return [NetworkMetric.model_validate(r) for r in raw]

    # ------------------------------------------------------------------
    def _simulate(self) -> list[NetworkMetric]:
        scenario_name: str = self.config.get("scenario", "congestion")
        if scenario_name not in _SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario_name!r}. "
                f"Choose from: {list(_SCENARIOS)}"
            )
        scenario = _SCENARIOS[scenario_name]
        num_devices: int = self.config.get("num_devices", 3)
        num_samples: int = self.config.get("num_samples", 60)
        interval: int = self.config.get("sample_interval_seconds", 60)
        seed: int | None = self.config.get("seed")

        rng = random.Random(seed)
        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(seconds=interval * num_samples)

        # Always include healthy baseline metrics
        all_metrics_cfg: dict[MetricType, tuple[float, float]] = dict(
            _SCENARIOS["normal"]["metrics"]
        )
        # Overlay scenario anomalies
        all_metrics_cfg.update(scenario["metrics"])

        metrics: list[NetworkMetric] = []
        devices = [f"router-{i:02d}" for i in range(1, num_devices + 1)]
        interfaces = ["GigE0/0", "GigE0/1"]

        for device in devices:
            for iface in interfaces:
                for metric_type, (mean, std) in all_metrics_cfg.items():
                    for i in range(num_samples):
                        ts = start + timedelta(seconds=interval * i)
                        # Add a gradual ramp for anomaly metrics in the second half
                        ramp = 1.0
                        if metric_type in scenario["metrics"] and i >= num_samples // 2:
                            progress = (i - num_samples // 2) / max(1, num_samples // 2)
                            ramp = 1.0 + progress * 0.5  # up to 50% amplification
                        raw_val = rng.gauss(mean * ramp, std)
                        value = max(0.0, raw_val)
                        metrics.append(
                            NetworkMetric(
                                timestamp=ts,
                                device_id=device,
                                interface=iface,
                                metric_type=metric_type,
                                value=value,
                                unit=_METRIC_UNITS[metric_type],
                                tags={"scenario": scenario_name},
                            )
                        )
        return metrics
