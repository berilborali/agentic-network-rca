#!/usr/bin/env python3
"""Generate a sample JSON data file for agentic-network-rca.

Usage:
    python examples/generate_sample_data.py --scenario hardware_failure \
        --output data/sample_hardware_failure.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from network_rca.agents.data_ingestion_agent import DataIngestionAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample network metric data")
    parser.add_argument(
        "--scenario",
        choices=["congestion", "hardware_failure", "misconfiguration",
                 "resource_exhaustion", "normal"],
        default="congestion",
    )
    parser.add_argument("--devices", type=int, default=3)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/sample_data.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    agent = DataIngestionAgent(
        config={
            "scenario": args.scenario,
            "num_devices": args.devices,
            "num_samples": args.samples,
            "seed": args.seed,
        }
    )
    context = agent.run({})
    metrics = context["metrics"]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(
            [m.model_dump(mode="json") for m in metrics],
            fh,
            indent=2,
            default=str,
        )
    print(f"Wrote {len(metrics)} metrics to {out_path}")


if __name__ == "__main__":
    main()
