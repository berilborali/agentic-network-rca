"""Tests for the RCAOrchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from network_rca.orchestrator import RCAOrchestrator
from network_rca.models.network_event import RCAReport, Severity


class TestRCAOrchestrator:
    def _run(self, scenario: str = "congestion", seed: int = 42, **overrides) -> RCAReport:
        config = {
            "ingestion": {
                "scenario": scenario,
                "num_devices": 2,
                "num_samples": 40,
                "seed": seed,
            },
            "anomaly_detection": {},
            "rca": {},
            "report": {},
        }
        config["ingestion"].update(overrides)
        return RCAOrchestrator(config=config).run()

    def test_returns_rca_report(self):
        report = self._run()
        assert isinstance(report, RCAReport)

    def test_report_has_metrics(self):
        report = self._run()
        assert report.total_metrics_analyzed > 0

    def test_congestion_scenario(self):
        report = self._run(scenario="congestion")
        # Should detect anomalies in congestion scenario
        assert len(report.anomalies_detected) > 0

    def test_hardware_failure_scenario(self):
        report = self._run(scenario="hardware_failure")
        categories = {rc.category for rc in report.root_causes}
        assert "hardware_failure" in categories

    def test_misconfiguration_scenario(self):
        report = self._run(scenario="misconfiguration")
        categories = {rc.category for rc in report.root_causes}
        assert "misconfiguration" in categories

    def test_resource_exhaustion_scenario(self):
        report = self._run(scenario="resource_exhaustion")
        categories = {rc.category for rc in report.root_causes}
        assert "resource_exhaustion" in categories

    def test_normal_scenario_lower_severity(self):
        report = self._run(scenario="normal")
        # Normal scenario should yield no CRITICAL findings
        assert report.severity != Severity.CRITICAL

    def test_report_summary_non_empty(self):
        report = self._run()
        assert report.summary.strip() != ""

    def test_root_causes_confidence_sorted(self):
        report = self._run()
        confs = [rc.confidence for rc in report.root_causes]
        assert confs == sorted(confs, reverse=True)

    def test_report_saved_as_json(self, tmp_path):
        import json
        out = tmp_path / "report.json"
        config = {
            "ingestion": {"scenario": "congestion", "num_devices": 1, "num_samples": 30, "seed": 0},
            "anomaly_detection": {},
            "rca": {},
            "report": {"output_path": str(out), "output_format": "json"},
        }
        report = RCAOrchestrator(config=config).run()
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["report_id"] == report.report_id

    def test_default_config(self):
        """Orchestrator should work with no config (uses all defaults)."""
        report = RCAOrchestrator().run()
        assert isinstance(report, RCAReport)
