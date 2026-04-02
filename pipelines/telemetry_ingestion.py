"""
Telemetry Ingestion Pipeline
----------------------------
Loads raw network log records from a JSON file (or a list of dicts),
normalises them into a flat text representation suitable for embedding,
and returns them ready for ingestion into the vector store.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path(__file__).parent.parent / "data" / "network_logs.json"


def load_logs(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load network log records from a JSON file.

    Args:
        path: Absolute or relative path to the JSON log file.
              Defaults to ``data/network_logs.json`` in the project root.

    Returns:
        List of raw log dictionaries.
    """
    log_path = Path(path) if path else _DEFAULT_LOG_PATH
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        raise ValueError("Log file must contain a JSON array of objects.")

    logger.info("Loaded %d log records from %s", len(records), log_path)
    return records


def normalise_log(record: dict[str, Any]) -> str:
    """Convert a single log record dict into a human-readable text chunk.

    This text chunk is what gets embedded into the vector store.

    Args:
        record: A raw log record dictionary.

    Returns:
        A flat string representation of the log entry.
    """
    parts = [
        f"[{record.get('timestamp', 'unknown')}]",
        f"Device: {record.get('device', 'unknown')}",
        f"Event: {record.get('event_type', 'unknown')}",
        f"Severity: {record.get('severity', 'unknown')}",
        f"Message: {record.get('message', '')}",
    ]

    metrics = record.get("metrics", {})
    if metrics:
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
        parts.append(f"Metrics: {metric_str}")

    tags = record.get("tags", [])
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")

    return " | ".join(parts)


def ingest_logs(
    path: str | Path | None = None,
    records: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Load and normalise network logs into embeddable text chunks.

    Provide *either* ``path`` (file path) or ``records`` (pre-loaded list).

    Args:
        path:    Path to the JSON log file (optional).
        records: Pre-loaded list of log dictionaries (optional).

    Returns:
        List of normalised text strings ready for embedding.
    """
    if records is None:
        records = load_logs(path)

    chunks = [normalise_log(r) for r in records]
    logger.info("Produced %d text chunks from telemetry logs.", len(chunks))
    return chunks
