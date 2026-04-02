"""Report Agent — formats and persists the RCA report."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from network_rca.agents.base_agent import BaseAgent
from network_rca.models.network_event import (
    Anomaly,
    NetworkMetric,
    RCAReport,
    RootCause,
    Severity,
)


_SEVERITY_ORDER = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
}


def _overall_severity(anomalies: list[Anomaly]) -> Severity:
    if not anomalies:
        return Severity.INFO
    return max(anomalies, key=lambda a: _SEVERITY_ORDER[a.severity]).severity


def _build_summary(
    anomalies: list[Anomaly], root_causes: list[RootCause]
) -> str:
    if not anomalies:
        return "No anomalies were detected during the analysis window."
    sev = _overall_severity(anomalies)
    cause_titles = (
        ", ".join(f'"{c.title}"' for c in root_causes[:3]) if root_causes else "unknown"
    )
    return (
        f"Analysis detected {len(anomalies)} anomaly(ies) with overall severity "
        f"{sev.value.upper()}. "
        f"Top root-cause hypothesis(es): {cause_titles}. "
        f"See recommended_actions in each root cause for remediation steps."
    )


class ReportAgent(BaseAgent):
    """Assembles and optionally serialises the final RCA report.

    Configuration keys
    ------------------
    output_path : str | None
        If set, write the report as JSON to this path.
    output_format : "json" | "text"
        Format for the output file (default: "json").
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__("report", config)

    # ------------------------------------------------------------------
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        metrics: list[NetworkMetric] = context.get("metrics", [])
        anomalies: list[Anomaly] = context.get("anomalies", [])
        root_causes: list[RootCause] = context.get("root_causes", [])

        timestamps = [m.timestamp for m in metrics] if metrics else [datetime.now(tz=timezone.utc)]
        window_start = min(timestamps)
        window_end = max(timestamps)

        report = RCAReport(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.now(tz=timezone.utc),
            analysis_window_start=window_start,
            analysis_window_end=window_end,
            total_metrics_analyzed=len(metrics),
            anomalies_detected=anomalies,
            root_causes=root_causes,
            summary=_build_summary(anomalies, root_causes),
            severity=_overall_severity(anomalies),
        )

        output_path: str | None = self.config.get("output_path")
        output_format: str = self.config.get("output_format", "json")
        if output_path:
            self._write(report, Path(output_path), output_format)

        self.logger.info("Report %s generated (severity=%s)", report.report_id, report.severity.value)
        context["report"] = report
        return context

    # ------------------------------------------------------------------
    def _write(self, report: RCAReport, path: Path, fmt: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            with path.open("w") as fh:
                json.dump(report.model_dump(mode="json"), fh, indent=2, default=str)
            self.logger.info("Report written to %s", path)
        elif fmt == "text":
            with path.open("w") as fh:
                fh.write(self._to_text(report))
            self.logger.info("Text report written to %s", path)
        else:
            raise ValueError(f"Unknown output_format {fmt!r}")

    # ------------------------------------------------------------------
    @staticmethod
    def _to_text(report: RCAReport) -> str:
        lines = [
            "=" * 70,
            "  AGENTIC NETWORK ROOT CAUSE ANALYSIS REPORT",
            "=" * 70,
            f"  Report ID  : {report.report_id}",
            f"  Generated  : {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"  Window     : {report.analysis_window_start} → {report.analysis_window_end}",
            f"  Metrics    : {report.total_metrics_analyzed}",
            f"  Severity   : {report.severity.value.upper()}",
            "",
            "SUMMARY",
            "-" * 70,
            report.summary,
            "",
        ]

        if report.anomalies_detected:
            lines += ["ANOMALIES DETECTED", "-" * 70]
            for a in sorted(
                report.anomalies_detected,
                key=lambda x: _SEVERITY_ORDER[x.severity],
                reverse=True,
            ):
                lines.append(
                    f"  [{a.severity.value.upper():8s}] {a.description}"
                )
            lines.append("")

        if report.root_causes:
            lines += ["ROOT CAUSES & RECOMMENDED ACTIONS", "-" * 70]
            for i, rc in enumerate(
                sorted(report.root_causes, key=lambda c: c.confidence, reverse=True), 1
            ):
                lines += [
                    f"  {i}. {rc.title}  (confidence: {rc.confidence:.0%})",
                    f"     Category : {rc.category}",
                    f"     Devices  : {', '.join(rc.affected_devices)}",
                    f"     {rc.description}",
                    "     Actions:",
                ]
                for action in rc.recommended_actions:
                    lines.append(f"       • {action}")
                lines.append("")

        lines += ["=" * 70]
        return "\n".join(lines)
