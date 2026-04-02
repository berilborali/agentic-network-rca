"""RCA Agent — maps anomaly patterns to root-cause hypotheses."""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from typing import Any

from network_rca.agents.base_agent import BaseAgent
from network_rca.models.network_event import Anomaly, MetricType, RootCause, Severity


# ---------------------------------------------------------------------------
# Rule definitions: each rule maps a set of metric types to a root cause
# ---------------------------------------------------------------------------

_SEV_WEIGHT = {
    Severity.CRITICAL: 1.0,
    Severity.HIGH: 0.75,
    Severity.MEDIUM: 0.5,
    Severity.LOW: 0.25,
    Severity.INFO: 0.1,
}


class _Rule:
    """A simple pattern-matching rule."""

    def __init__(
        self,
        category: str,
        title: str,
        description: str,
        required: set[MetricType],
        optional: set[MetricType],
        base_confidence: float,
        actions: list[str],
    ) -> None:
        self.category = category
        self.title = title
        self.description = description
        self.required = required
        self.optional = optional
        self.base_confidence = base_confidence
        self.actions = actions

    def score(self, anomaly_types: set[MetricType]) -> float:
        if not self.required.issubset(anomaly_types):
            return 0.0
        matched_optional = len(self.optional & anomaly_types)
        optional_boost = matched_optional / max(1, len(self.optional)) * 0.3
        return min(1.0, self.base_confidence + optional_boost)


_RULES: list[_Rule] = [
    _Rule(
        category="congestion",
        title="Network Congestion",
        description=(
            "High bandwidth utilisation is causing queue build-up, increasing "
            "end-to-end latency and driving packet drops."
        ),
        required={MetricType.BANDWIDTH_UTILIZATION},
        optional={MetricType.LATENCY, MetricType.PACKET_LOSS, MetricType.JITTER},
        base_confidence=0.60,
        actions=[
            "Identify and throttle top-talker flows (NetFlow/sFlow analysis).",
            "Enable QoS policies to prioritise critical traffic.",
            "Consider capacity upgrade or traffic engineering (ECMP/MPLS TE).",
            "Activate traffic policing on uplink interfaces.",
        ],
    ),
    _Rule(
        category="hardware_failure",
        title="Hardware or Physical Layer Failure",
        description=(
            "Elevated error rate and packet loss indicate a failing NIC, "
            "damaged cable, or dirty optical fibre."
        ),
        required={MetricType.ERROR_RATE, MetricType.PACKET_LOSS},
        optional={MetricType.THROUGHPUT},
        base_confidence=0.65,
        actions=[
            "Inspect physical layer: check optical power levels, SFP modules, cables.",
            "Review interface error counters (CRC, input errors, runts, giants).",
            "Replace suspect transceiver or cable segment.",
            "Check for duplex mismatches with `show interface`.",
        ],
    ),
    _Rule(
        category="misconfiguration",
        title="Network Misconfiguration",
        description=(
            "Anomalous jitter and latency without proportional bandwidth "
            "increase suggest MTU mismatch, routing loop, or ACL issue."
        ),
        required={MetricType.JITTER},
        optional={MetricType.LATENCY, MetricType.ERROR_RATE},
        base_confidence=0.55,
        actions=[
            "Audit MTU settings across all path segments (check for fragmentation).",
            "Inspect routing tables for loops or black-holes (`traceroute`, `show ip route`).",
            "Review recent configuration changes (git diff of device configs).",
            "Validate ACL and QoS policy consistency.",
        ],
    ),
    _Rule(
        category="resource_exhaustion",
        title="Device Resource Exhaustion",
        description=(
            "CPU and/or memory saturation on the forwarding device is causing "
            "control-plane instability and delayed packet processing."
        ),
        required={MetricType.CPU_UTILIZATION},
        optional={MetricType.MEMORY_UTILIZATION, MetricType.LATENCY},
        base_confidence=0.65,
        actions=[
            "Identify CPU-hogging processes (`show processes cpu sorted`).",
            "Check for routing protocol instability (BGP/OSPF flaps).",
            "Disable unnecessary services and features.",
            "Upgrade device memory/CPU if resource headroom is chronically low.",
            "Consider redistributing workload across multiple devices.",
        ],
    ),
    _Rule(
        category="throughput_degradation",
        title="Unexplained Throughput Degradation",
        description=(
            "Throughput has dropped significantly below baseline without a "
            "corresponding rise in bandwidth utilisation, suggesting software "
            "bug, rate-limiting, or asymmetric routing."
        ),
        required={MetricType.THROUGHPUT},
        optional={MetricType.LATENCY, MetricType.PACKET_LOSS},
        base_confidence=0.50,
        actions=[
            "Verify rate-limiter and policer configurations on ingress/egress.",
            "Check for asymmetric routing paths that may drop or delay return traffic.",
            "Review recent software updates for known throughput regression bugs.",
            "Run iperf3/iPerf tests to isolate the affected segment.",
        ],
    ),
]


class RCAAgent(BaseAgent):
    """Performs root-cause analysis by matching anomaly patterns to rules.

    The agent also boosts confidence when multiple devices exhibit the
    same anomaly pattern (pointing to a shared upstream cause).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("rca", config)

    # ------------------------------------------------------------------
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        anomalies: list[Anomaly] = context.get("anomalies", [])
        if not anomalies:
            context["root_causes"] = []
            return context

        # Group anomalies by device
        by_device: dict[str, list[Anomaly]] = defaultdict(list)
        for a in anomalies:
            by_device[a.device_id].append(a)

        global_types = {a.metric_type for a in anomalies}
        device_causes: list[RootCause] = []

        for device_id, device_anomalies in by_device.items():
            dev_types = {a.metric_type for a in device_anomalies}
            affected_interfaces = list({a.interface for a in device_anomalies})
            severity_boost = self._severity_boost(device_anomalies)

            for rule in _RULES:
                score = rule.score(dev_types)
                if score < 0.3:
                    continue
                adjusted = min(1.0, score + severity_boost)
                cause = RootCause(
                    cause_id=str(uuid.uuid4()),
                    category=rule.category,
                    title=rule.title,
                    description=rule.description,
                    confidence=adjusted,
                    affected_devices=[device_id],
                    affected_interfaces=affected_interfaces,
                    supporting_anomalies=[
                        a.anomaly_id
                        for a in device_anomalies
                        if a.metric_type in (rule.required | rule.optional)
                    ],
                    recommended_actions=rule.actions,
                )
                device_causes.append(cause)

        # Merge same-category causes across devices
        root_causes = self._merge_causes(device_causes, len(by_device))
        root_causes.sort(key=lambda c: c.confidence, reverse=True)

        self.logger.info(
            "Identified %d root-cause hypothesis(es)", len(root_causes)
        )
        context["root_causes"] = root_causes
        return context

    # ------------------------------------------------------------------
    @staticmethod
    def _severity_boost(anomalies: list[Anomaly]) -> float:
        """Extra confidence proportional to the worst anomaly severity."""
        if not anomalies:
            return 0.0
        max_weight = max(_SEV_WEIGHT.get(a.severity, 0) for a in anomalies)
        return max_weight * 0.1

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_causes(causes: list[RootCause], total_devices: int) -> list[RootCause]:
        """Merge per-device causes of the same category; boost confidence."""
        by_category: dict[str, list[RootCause]] = defaultdict(list)
        for c in causes:
            by_category[c.category].append(c)

        merged: list[RootCause] = []
        for category, group in by_category.items():
            devices = sorted({d for c in group for d in c.affected_devices})
            interfaces = sorted({i for c in group for i in c.affected_interfaces})
            anomaly_ids = sorted({a for c in group for a in c.supporting_anomalies})
            # Boost confidence when multiple devices are affected (shared-cause signal)
            spread_factor = len(devices) / max(1, total_devices)
            avg_conf = sum(c.confidence for c in group) / len(group)
            boosted = min(1.0, avg_conf + spread_factor * 0.1)
            representative = group[0]
            merged.append(
                RootCause(
                    cause_id=representative.cause_id,
                    category=category,
                    title=representative.title,
                    description=representative.description,
                    confidence=boosted,
                    affected_devices=devices,
                    affected_interfaces=interfaces,
                    supporting_anomalies=anomaly_ids,
                    recommended_actions=representative.recommended_actions,
                    metadata={"device_count": len(devices), "rule_hits": len(group)},
                )
            )
        return merged
