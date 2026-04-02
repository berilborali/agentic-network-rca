"""Command-line interface for the agentic-network-rca tool."""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from network_rca.orchestrator import RCAOrchestrator
from network_rca.models.network_event import RCAReport, Severity


console = Console()

_SEVERITY_COLOUR = {
    Severity.CRITICAL: "red",
    Severity.HIGH: "orange3",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "green",
}


def _print_report(report: RCAReport) -> None:
    sev_colour = _SEVERITY_COLOUR.get(report.severity, "white")

    console.rule(f"[bold]Agentic Network RCA Report[/bold]  "
                 f"[{sev_colour}]{report.severity.value.upper()}[/{sev_colour}]")

    console.print(f"\n[bold]Summary:[/bold] {report.summary}\n")
    console.print(
        f"  Metrics analysed : [bold]{report.total_metrics_analyzed}[/bold]"
    )
    console.print(f"  Anomalies found  : [bold]{len(report.anomalies_detected)}[/bold]")
    console.print(f"  Root causes      : [bold]{len(report.root_causes)}[/bold]\n")

    if report.anomalies_detected:
        table = Table(title="Anomalies", box=box.SIMPLE_HEAVY, show_lines=False)
        table.add_column("Severity", style="bold", width=10)
        table.add_column("Device", width=14)
        table.add_column("Interface", width=12)
        table.add_column("Metric", width=26)
        table.add_column("Observed", justify="right", width=12)
        table.add_column("Baseline", justify="right", width=12)
        table.add_column("Z-Score", justify="right", width=9)

        for a in sorted(
            report.anomalies_detected,
            key=lambda x: {
                Severity.CRITICAL: 4,
                Severity.HIGH: 3,
                Severity.MEDIUM: 2,
                Severity.LOW: 1,
                Severity.INFO: 0,
            }[x.severity],
            reverse=True,
        ):
            colour = _SEVERITY_COLOUR.get(a.severity, "white")
            table.add_row(
                f"[{colour}]{a.severity.value.upper()}[/{colour}]",
                a.device_id,
                a.interface,
                a.metric_type.value,
                f"{a.observed_value:.2f}",
                f"{a.baseline_value:.2f}",
                f"{a.deviation_score:.2f}",
            )
        console.print(table)

    if report.root_causes:
        for i, rc in enumerate(
            sorted(report.root_causes, key=lambda c: c.confidence, reverse=True), 1
        ):
            colour = "green" if rc.confidence >= 0.7 else "yellow"
            title = (
                f"[bold]{i}. {rc.title}[/bold]  "
                f"[{colour}]{rc.confidence:.0%} confidence[/{colour}]"
            )
            body_lines = [
                f"[dim]Category:[/dim] {rc.category}",
                f"[dim]Devices:[/dim]  {', '.join(rc.affected_devices)}",
                "",
                rc.description,
                "",
                "[bold]Recommended Actions:[/bold]",
            ]
            for action in rc.recommended_actions:
                body_lines.append(f"  • {action}")
            console.print(Panel("\n".join(body_lines), title=title, border_style=colour))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="network-rca",
        description="Agentic Network Root Cause Analysis",
    )
    parser.add_argument(
        "--scenario",
        choices=["congestion", "hardware_failure", "misconfiguration",
                 "resource_exhaustion", "normal"],
        default="congestion",
        help="Simulation scenario (default: congestion)",
    )
    parser.add_argument(
        "--data-file",
        metavar="PATH",
        help="Path to a JSON file of NetworkMetric objects (overrides --scenario)",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write the report to this file path",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="json",
        help="Output file format (default: json)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=3,
        metavar="N",
        help="Number of simulated devices (default: 3)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=60,
        metavar="N",
        help="Time-series samples per metric per device (default: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulation",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    config: dict = {
        "ingestion": {
            "scenario": args.scenario,
            "num_devices": args.devices,
            "num_samples": args.samples,
            "seed": args.seed,
        },
        "anomaly_detection": {},
        "rca": {},
        "report": {},
    }
    if args.data_file:
        config["ingestion"]["data_file"] = args.data_file
    if args.output:
        config["report"]["output_path"] = args.output
        config["report"]["output_format"] = args.output_format

    try:
        orchestrator = RCAOrchestrator(config=config)
        report = orchestrator.run()
        _print_report(report)
        # Exit code reflects severity
        if report.severity == Severity.CRITICAL:
            return 2
        if report.severity in (Severity.HIGH, Severity.MEDIUM):
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]ERROR:[/red] {exc}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
