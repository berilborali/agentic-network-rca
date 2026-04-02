"""Orchestrator — chains agents into a complete RCA pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from network_rca.agents import (
    AnomalyDetectionAgent,
    DataIngestionAgent,
    RCAAgent,
    ReportAgent,
)
from network_rca.models.network_event import RCAReport


log = logging.getLogger(__name__)


class RCAOrchestrator:
    """Coordinates the full agentic RCA pipeline.

    Pipeline stages
    ---------------
    1. **DataIngestionAgent** — loads or simulates network metrics.
    2. **AnomalyDetectionAgent** — flags statistical/threshold anomalies.
    3. **RCAAgent** — maps anomaly patterns to root-cause hypotheses.
    4. **ReportAgent** — assembles and (optionally) persists the report.

    Parameters
    ----------
    config:
        Top-level configuration dict.  Keys recognised:

        ``ingestion`` (dict)  → forwarded to :class:`DataIngestionAgent`
        ``anomaly_detection`` (dict) → forwarded to :class:`AnomalyDetectionAgent`
        ``rca`` (dict) → forwarded to :class:`RCAAgent`
        ``report`` (dict) → forwarded to :class:`ReportAgent`
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._agents = [
            DataIngestionAgent(config=cfg.get("ingestion", {})),
            AnomalyDetectionAgent(config=cfg.get("anomaly_detection", {})),
            RCAAgent(config=cfg.get("rca", {})),
            ReportAgent(config=cfg.get("report", {})),
        ]

    # ------------------------------------------------------------------
    def run(self) -> RCAReport:
        """Execute all pipeline stages and return the final :class:`RCAReport`."""
        context: dict[str, Any] = {}
        for agent in self._agents:
            log.info("Running agent: %s", agent.name)
            t0 = time.perf_counter()
            context = agent.run(context)
            elapsed = time.perf_counter() - t0
            log.debug("Agent %s finished in %.3fs", agent.name, elapsed)

        report: RCAReport = context["report"]
        return report
