"""Tests for the telemetry ingestion pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from pipelines.telemetry_ingestion import ingest_logs, load_logs, normalise_log


SAMPLE_RECORD = {
    "id": "test_001",
    "timestamp": "2024-01-15T08:23:11Z",
    "device": "router-R17",
    "event_type": "packet_drop",
    "severity": "high",
    "message": "Router R17 packet drop rate increased to 18%",
    "metrics": {"packet_drop_rate": 0.18, "cpu_usage": 0.72},
    "tags": ["packet_loss", "R17"],
}


def test_load_logs_default_file():
    """Default log file should contain multiple records."""
    records = load_logs()
    assert isinstance(records, list)
    assert len(records) > 0


def test_load_logs_custom_path():
    """Custom file path should load correctly."""
    data = [SAMPLE_RECORD]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name

    records = load_logs(tmp_path)
    assert len(records) == 1
    assert records[0]["device"] == "router-R17"


def test_load_logs_file_not_found():
    """Non-existent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_logs("/nonexistent/path/logs.json")


def test_load_logs_invalid_format():
    """File with dict instead of list should raise ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"not": "a list"}, f)
        tmp_path = f.name

    with pytest.raises(ValueError):
        load_logs(tmp_path)


def test_normalise_log_contains_device():
    """Normalised log text should include device name."""
    text = normalise_log(SAMPLE_RECORD)
    assert "router-R17" in text


def test_normalise_log_contains_message():
    """Normalised log text should include the message."""
    text = normalise_log(SAMPLE_RECORD)
    assert "packet drop rate increased" in text


def test_normalise_log_contains_metrics():
    """Normalised log text should include metric values."""
    text = normalise_log(SAMPLE_RECORD)
    assert "cpu_usage" in text


def test_normalise_log_contains_tags():
    """Normalised log text should include tags."""
    text = normalise_log(SAMPLE_RECORD)
    assert "packet_loss" in text


def test_ingest_logs_from_records():
    """ingest_logs should accept pre-loaded records and return text chunks."""
    records = [SAMPLE_RECORD]
    chunks = ingest_logs(records=records)
    assert len(chunks) == 1
    assert isinstance(chunks[0], str)
    assert len(chunks[0]) > 10


def test_ingest_logs_default():
    """ingest_logs with no args should ingest the default log file."""
    chunks = ingest_logs()
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
