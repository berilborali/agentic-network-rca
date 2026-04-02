#!/usr/bin/env python3
"""Demonstrate the full agentic-network-rca pipeline.

Run from the repository root:
    python examples/run_rca.py
    python examples/run_rca.py --scenario hardware_failure
    python examples/run_rca.py --scenario all   # iterate over all scenarios
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console

from network_rca.orchestrator import RCAOrchestrator
from network_rca.models.network_event import Severity

console = Console()

SCENARIOS = [
    "congestion",
    "hardware_failure",
    "misconfiguration",
    "resource_exhaustion",
    "normal",
]

_SEV_COLOUR = {
    Severity.CRITICAL: "red",
    Severity.HIGH: "orange3",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "green",
}


def run_scenario(scenario: str, seed: int = 42) -> None:
    console.rule(f"[bold]Scenario: {scenario}[/bold]")
    config = {
        "ingestion": {
            "scenario": scenario,
            "num_devices": 3,
            "num_samples": 60,
            "seed": seed,
        },
        "anomaly_detection": {"z_threshold": 2.5},
        "rca": {},
        "report": {},
    }
    orchestrator = RCAOrchestrator(config=config)
    report = orchestrator.run()

    sev = report.severity
    colour = _SEV_COLOUR.get(sev, "white")
    console.print(f"  Severity : [{colour}]{sev.value.upper()}[/{colour}]")
    console.print(f"  Metrics  : {report.total_metrics_analyzed}")
    console.print(f"  Anomalies: {len(report.anomalies_detected)}")
    console.print(f"  Summary  : {report.summary}\n")

    if report.root_causes:
        console.print("  [bold]Top root cause:[/bold]")
        top = report.top_cause
        if top:
            console.print(f"    [{colour}]{top.title}[/{colour}]  ({top.confidence:.0%} confidence)")
            console.print(f"    {top.description}")
            console.print(f"    Actions: {top.recommended_actions[0]}")
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agentic-network-rca demo")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS + ["all"],
        default="congestion",
        help="Scenario to simulate (use 'all' to iterate through every scenario)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        run_scenario(scenario, seed=args.seed)


if __name__ == "__main__":
    main()
