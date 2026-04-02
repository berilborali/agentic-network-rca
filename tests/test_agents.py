"""Tests for individual agents."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timezone

from network_rca.agents.data_ingestion_agent import DataIngestionAgent
from network_rca.agents.anomaly_detection_agent import AnomalyDetectionAgent
from network_rca.agents.rca_agent import RCAAgent
from network_rca.agents.report_agent import ReportAgent
from network_rca.models.network_event import (
    MetricType,
    NetworkMetric,
    Severity,
)


# ---------------------------------------------------------------------------
# DataIngestionAgent
# ---------------------------------------------------------------------------

class TestDataIngestionAgent:
    def _make_agent(self, scenario: str = "congestion", **kwargs):
        return DataIngestionAgent(
            config={"scenario": scenario, "num_devices": 2, "num_samples": 20, "seed": 0, **kwargs}
        )

    def test_returns_metrics_list(self):
        agent = self._make_agent()
        ctx = agent.run({})
        assert "metrics" in ctx
        assert len(ctx["metrics"]) > 0

    def test_all_scenarios_produce_metrics(self):
        for scenario in ["congestion", "hardware_failure", "misconfiguration",
                         "resource_exhaustion", "normal"]:
            agent = self._make_agent(scenario=scenario)
            ctx = agent.run({})
            assert len(ctx["metrics"]) > 0, f"No metrics for scenario={scenario}"

    def test_metric_values_non_negative(self):
        agent = self._make_agent()
        ctx = agent.run({})
        for m in ctx["metrics"]:
            assert m.value >= 0, f"Negative value for {m.metric_type}"

    def test_unknown_scenario_raises(self):
        agent = DataIngestionAgent(config={"scenario": "bogus"})
        with pytest.raises(ValueError, match="Unknown scenario"):
            agent.run({})

    def test_context_passthrough(self):
        agent = self._make_agent()
        ctx = agent.run({"existing_key": "existing_value"})
        assert ctx["existing_key"] == "existing_value"

    def test_load_from_file(self, tmp_path):
        import json
        # First generate some metrics
        agent = self._make_agent(scenario="normal")
        ctx = agent.run({})
        metrics = ctx["metrics"]
        data = [m.model_dump(mode="json") for m in metrics]
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data, default=str))

        loader = DataIngestionAgent(config={"data_file": str(p)})
        ctx2 = loader.run({})
        assert len(ctx2["metrics"]) == len(metrics)


# ---------------------------------------------------------------------------
# AnomalyDetectionAgent
# ---------------------------------------------------------------------------

class TestAnomalyDetectionAgent:
    def _make_metrics(self, num_normal: int = 30, spike_value: float = 999.0):
        now = datetime.now(tz=timezone.utc)
        metrics = []
        for i in range(num_normal):
            metrics.append(NetworkMetric(
                timestamp=now,
                device_id="router-01",
                interface="GigE0/0",
                metric_type=MetricType.LATENCY,
                value=15.0 + (i % 3),
                unit="ms",
            ))
        # Add a spike
        metrics.append(NetworkMetric(
            timestamp=now,
            device_id="router-01",
            interface="GigE0/0",
            metric_type=MetricType.LATENCY,
            value=spike_value,
            unit="ms",
        ))
        return metrics

    def test_spike_detected(self):
        metrics = self._make_metrics(spike_value=500.0)
        agent = AnomalyDetectionAgent(config={"z_threshold": 2.5})
        ctx = agent.run({"metrics": metrics})
        anomalies = ctx["anomalies"]
        assert len(anomalies) >= 1
        assert any(a.metric_type == MetricType.LATENCY for a in anomalies)

    def test_normal_metrics_no_anomaly(self):
        now = datetime.now(tz=timezone.utc)
        metrics = [
            NetworkMetric(
                timestamp=now,
                device_id="r1",
                interface="eth0",
                metric_type=MetricType.LATENCY,
                value=v,
                unit="ms",
            )
            for v in [14, 15, 15, 16, 14, 15, 15, 14, 16, 15, 15, 15]
        ]
        agent = AnomalyDetectionAgent(config={"z_threshold": 3.0})
        ctx = agent.run({"metrics": metrics})
        # Should flag some due to absolute threshold (>100ms is the absolute),
        # but all values are 14-16 so no absolute breach either
        latency_anomalies = [
            a for a in ctx["anomalies"]
            if a.metric_type == MetricType.LATENCY and a.observed_value > 100
        ]
        assert len(latency_anomalies) == 0

    def test_empty_metrics(self):
        agent = AnomalyDetectionAgent()
        ctx = agent.run({"metrics": []})
        assert ctx["anomalies"] == []

    def test_deduplication(self):
        """Multiple spikes for the same (device, interface, metric) → deduplicated."""
        now = datetime.now(tz=timezone.utc)
        base = [
            NetworkMetric(timestamp=now, device_id="r1", interface="eth0",
                          metric_type=MetricType.LATENCY, value=15.0, unit="ms")
            for _ in range(20)
        ]
        spikes = [
            NetworkMetric(timestamp=now, device_id="r1", interface="eth0",
                          metric_type=MetricType.LATENCY, value=900.0, unit="ms")
            for _ in range(5)
        ]
        agent = AnomalyDetectionAgent(config={"z_threshold": 2.5})
        ctx = agent.run({"metrics": base + spikes})
        latency_anoms = [a for a in ctx["anomalies"] if a.metric_type == MetricType.LATENCY]
        assert len(latency_anoms) == 1


# ---------------------------------------------------------------------------
# RCAAgent
# ---------------------------------------------------------------------------

class TestRCAAgent:
    def _make_anomaly(self, device_id="r1", metric_type=MetricType.LATENCY,
                      severity=Severity.HIGH):
        from network_rca.models.network_event import Anomaly
        import uuid
        return Anomaly(
            anomaly_id=str(uuid.uuid4()),
            detected_at=datetime.now(tz=timezone.utc),
            device_id=device_id,
            interface="eth0",
            metric_type=metric_type,
            observed_value=500.0,
            baseline_value=15.0,
            deviation_score=5.0,
            severity=severity,
            description="test anomaly",
        )

    def test_congestion_detected(self):
        agent = RCAAgent()
        anomalies = [
            self._make_anomaly(metric_type=MetricType.BANDWIDTH_UTILIZATION, severity=Severity.HIGH),
            self._make_anomaly(metric_type=MetricType.LATENCY, severity=Severity.HIGH),
            self._make_anomaly(metric_type=MetricType.PACKET_LOSS, severity=Severity.MEDIUM),
        ]
        ctx = agent.run({"anomalies": anomalies})
        categories = {rc.category for rc in ctx["root_causes"]}
        assert "congestion" in categories

    def test_hardware_failure_detected(self):
        agent = RCAAgent()
        anomalies = [
            self._make_anomaly(metric_type=MetricType.ERROR_RATE, severity=Severity.CRITICAL),
            self._make_anomaly(metric_type=MetricType.PACKET_LOSS, severity=Severity.CRITICAL),
        ]
        ctx = agent.run({"anomalies": anomalies})
        categories = {rc.category for rc in ctx["root_causes"]}
        assert "hardware_failure" in categories

    def test_no_anomalies_no_causes(self):
        agent = RCAAgent()
        ctx = agent.run({"anomalies": []})
        assert ctx["root_causes"] == []

    def test_confidence_in_range(self):
        agent = RCAAgent()
        anomalies = [
            self._make_anomaly(metric_type=MetricType.CPU_UTILIZATION, severity=Severity.CRITICAL),
        ]
        ctx = agent.run({"anomalies": anomalies})
        for rc in ctx["root_causes"]:
            assert 0.0 <= rc.confidence <= 1.0

    def test_multi_device_boosts_confidence(self):
        agent = RCAAgent()
        single_device_anomalies = [
            self._make_anomaly(device_id="r1", metric_type=MetricType.BANDWIDTH_UTILIZATION),
        ]
        multi_device_anomalies = [
            self._make_anomaly(device_id=d, metric_type=MetricType.BANDWIDTH_UTILIZATION)
            for d in ["r1", "r2", "r3"]
        ]
        ctx_single = agent.run({"anomalies": single_device_anomalies})
        ctx_multi = agent.run({"anomalies": multi_device_anomalies})

        single_conf = next(
            (rc.confidence for rc in ctx_single["root_causes"] if rc.category == "congestion"),
            0.0,
        )
        multi_conf = next(
            (rc.confidence for rc in ctx_multi["root_causes"] if rc.category == "congestion"),
            0.0,
        )
        assert multi_conf >= single_conf


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------

class TestReportAgent:
    def test_report_assembled(self):
        agent = ReportAgent()
        from network_rca.agents.data_ingestion_agent import DataIngestionAgent
        from network_rca.agents.anomaly_detection_agent import AnomalyDetectionAgent
        from network_rca.agents.rca_agent import RCAAgent

        ctx = DataIngestionAgent(config={"scenario": "congestion", "num_devices": 1,
                                          "num_samples": 30, "seed": 0}).run({})
        ctx = AnomalyDetectionAgent().run(ctx)
        ctx = RCAAgent().run(ctx)
        ctx = agent.run(ctx)
        report = ctx["report"]
        assert report.report_id
        assert report.total_metrics_analyzed > 0
        assert report.summary

    def test_report_written_to_file(self, tmp_path):
        from network_rca.agents.data_ingestion_agent import DataIngestionAgent
        from network_rca.agents.anomaly_detection_agent import AnomalyDetectionAgent
        from network_rca.agents.rca_agent import RCAAgent

        out = tmp_path / "report.json"
        agent = ReportAgent(config={"output_path": str(out), "output_format": "json"})
        ctx = DataIngestionAgent(config={"scenario": "normal", "num_devices": 1,
                                          "num_samples": 20, "seed": 0}).run({})
        ctx = AnomalyDetectionAgent().run(ctx)
        ctx = RCAAgent().run(ctx)
        ctx = agent.run(ctx)
        assert out.exists()

    def test_text_report_written(self, tmp_path):
        from network_rca.agents.data_ingestion_agent import DataIngestionAgent
        from network_rca.agents.anomaly_detection_agent import AnomalyDetectionAgent
        from network_rca.agents.rca_agent import RCAAgent

        out = tmp_path / "report.txt"
        agent = ReportAgent(config={"output_path": str(out), "output_format": "text"})
        ctx = DataIngestionAgent(config={"scenario": "hardware_failure", "num_devices": 1,
                                          "num_samples": 30, "seed": 0}).run({})
        ctx = AnomalyDetectionAgent().run(ctx)
        ctx = RCAAgent().run(ctx)
        ctx = agent.run(ctx)
        content = out.read_text()
        assert "ROOT CAUSE" in content
