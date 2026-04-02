# Agentic Network RCA

> **AI-powered network root cause analysis** – ingests telemetry logs, retrieves contextual evidence with RAG, and runs a multi-agent LangGraph workflow to diagnose failures and recommend fixes.

[![CI](https://github.com/berilborali/agentic-network-rca/actions/workflows/ci.yml/badge.svg)](https://github.com/berilborali/agentic-network-rca/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-orange.svg)](https://langchain-ai.github.io/langgraph/)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Setup & Installation](#4-setup--installation)
5. [Running Locally](#5-running-locally)
6. [Docker Deployment](#6-docker-deployment)
7. [API Reference](#7-api-reference)
8. [Monitoring](#8-monitoring)
9. [Running Tests](#9-running-tests)
10. [Configuration](#10-configuration)
11. [Future Improvements](#11-future-improvements)

---

## 1. Project Overview

**agentic-network-rca** is an enterprise-style AI system that automates the root cause analysis of network incidents.

### Core workflow

```
User Input (free-text log)
        │
        ▼
  RAG Retrieval  ──► FAISS / Pinecone Vector Store
        │
        ▼
  Log Analysis Agent  ──► Summarises anomalies & evidence
        │
        ▼
  RCA Agent           ──► Infers root cause + confidence score
        │
        ▼
  Remediation Agent   ──► Proposes actionable fix + risk level
        │
        ▼
  JSON Response (FastAPI)
```

### Example

**Request:**
```json
POST /analyze_network
{
  "logs": "router R17 experiencing packet loss"
}
```

**Response:**
```json
{
  "root_cause": "routing loop congestion",
  "confidence": 0.87,
  "recommended_fix": "restart OSPF process on R17",
  "anomaly_summary": "Packet drop rate 18% on R17; OSPF adjacency flapping between R17 and R22 …",
  "evidence": ["OSPF flap count: 14", "CPU usage: 92%"],
  "remediation_steps": ["SSH to R17", "Execute: clear ip ospf process"],
  "risk_level": "low",
  "estimated_resolution_time": "5 minutes"
}
```

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│  POST /analyze_network    GET /health    GET /metrics           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LangGraph Multi-Agent Workflow                     │
│                                                                 │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │  Retrieve   │──►│  Log Analysis    │──►│   RCA Agent     │  │
│  │  (RAG)      │   │  Agent           │   │  (root cause +  │  │
│  └─────────────┘   └──────────────────┘   │   confidence)   │  │
│        │                                  └────────┬────────┘  │
│        │                                           │            │
│  ┌─────▼──────┐                          ┌─────────▼────────┐  │
│  │   FAISS    │                          │  Remediation     │  │
│  │  Vector    │                          │  Agent (fix +    │  │
│  │  Store     │                          │  risk level)     │  │
│  └────────────┘                          └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Telemetry Ingestion (data/network_logs.json)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           Prometheus Metrics  (prometheus_client)               │
│  rca_request_total | rca_request_latency_seconds | ...          │
└─────────────────────────────────────────────────────────────────┘
```

**Tech Stack:** Python 3.11 · LangChain · LangGraph · OpenAI API · FAISS · FastAPI · Prometheus · Docker

---

## 3. Repository Structure

```
agentic-network-rca/
├── agents/
│   ├── log_analysis_agent.py    # Summarises anomalies from logs
│   ├── root_cause_agent.py      # Infers root cause + confidence
│   ├── remediation_agent.py     # Proposes actionable fix
│   └── rca_workflow.py          # LangGraph workflow (chains all agents)
├── data/
│   └── network_logs.json        # Sample network telemetry data (15 records)
├── rag/
│   ├── vector_store.py          # FAISS / Pinecone abstraction
│   └── retrieval.py             # High-level retrieve_context() helper
├── pipelines/
│   └── telemetry_ingestion.py   # Loads & normalises log records
├── api/
│   └── server.py                # FastAPI app + request/response models
├── monitoring/
│   └── metrics.py               # Prometheus counters & histograms
├── docker/
│   ├── Dockerfile               # Production container image
│   ├── docker-compose.yml       # API + Prometheus stack
│   └── prometheus.yml           # Prometheus scrape config
├── notebooks/
│   └── demo.ipynb               # Usage examples
├── tests/
│   ├── test_agents.py           # Unit tests for all three agents
│   ├── test_api.py              # Integration tests for FastAPI endpoints
│   └── test_telemetry_ingestion.py
├── .env.example                 # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Setup & Installation

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- `pip` (or a virtual environment tool)

### Install

```bash
# 1. Clone the repository
git clone https://github.com/berilborali/agentic-network-rca.git
cd agentic-network-rca

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## 5. Running Locally

```bash
# Start the FastAPI server
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

The server will:
1. Load sample network logs from `data/network_logs.json`
2. Build a FAISS vector index in memory (warm-up)
3. Expose the API at `http://localhost:8000`
4. Serve interactive docs at `http://localhost:8000/docs`

### Quick test

```bash
curl -X POST http://localhost:8000/analyze_network \
  -H "Content-Type: application/json" \
  -d '{"logs": "router R17 experiencing packet loss and high CPU"}'
```

---

## 6. Docker Deployment

```bash
# Build and start the full stack (API + Prometheus)
cd docker
docker compose up --build

# Or build the image alone
docker build -f docker/Dockerfile -t agentic-network-rca:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=<your-key> agentic-network-rca:latest
```

Services exposed:
| Service    | URL                        |
|------------|----------------------------|
| RCA API    | http://localhost:8000      |
| API docs   | http://localhost:8000/docs |
| Prometheus | http://localhost:9090      |

---

## 7. API Reference

### `POST /analyze_network`

Run the multi-agent RCA pipeline on log text.

**Request body:**
```json
{
  "logs": "string (min 3 chars) – raw log text or incident description"
}
```

**Response:**
```json
{
  "root_cause":                "string",
  "confidence":                0.87,
  "recommended_fix":           "string",
  "anomaly_summary":           "string",
  "evidence":                  ["string", "..."],
  "remediation_steps":         ["string", "..."],
  "risk_level":                "low | medium | high",
  "estimated_resolution_time": "string"
}
```

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

### `GET /metrics`

Prometheus metrics in text format.

---

## 8. Monitoring

The `/metrics` endpoint exposes Prometheus-compatible metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `rca_request_total{status}` | Counter | Total requests by status (`success`/`error`) |
| `rca_request_latency_seconds` | Histogram | End-to-end request latency |
| `rca_pipeline_errors_total` | Counter | Pipeline (non-HTTP) failures |
| `rca_agent_invocations_total{agent}` | Counter | LLM invocations per agent |
| `rca_confidence_score` | Histogram | Distribution of RCA confidence scores |

Scrape with Prometheus or query via:
```bash
curl http://localhost:8000/metrics
```

---

## 9. Running Tests

```bash
# All tests (no OpenAI key required – LLMs are mocked)
OPENAI_API_KEY=test pytest tests/ -v

# A specific test file
pytest tests/test_agents.py -v
```

---

## 10. Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | – | **Required.** OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model to use |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTOR_STORE_BACKEND` | `faiss` | `faiss` or `pinecone` |
| `FAISS_INDEX_PATH` | `faiss_index` | Directory for persisted FAISS index |
| `RAG_TOP_K` | `5` | Number of context chunks to retrieve |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |
| `LOG_LEVEL` | `INFO` | Python logging level |

### Swapping to a local LLM

Replace `ChatOpenAI` with any LangChain-compatible chat model:

```python
# e.g. Ollama
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="llama3")
```

Pass the custom `llm` to each agent's `run_*` function, or update the `OPENAI_MODEL` env var when using the workflow.

---

## 11. Future Improvements

- **Streaming responses** – stream agent reasoning steps via Server-Sent Events
- **Persistent FAISS index** – save/load the vector index across restarts
- **Real-time log ingestion** – Kafka / Fluentd integration for live telemetry
- **Feedback loop** – human-in-the-loop rating to fine-tune confidence calibration
- **Multi-tenant support** – per-customer vector namespaces (Pinecone)
- **Graph-based topology awareness** – ingest network topology and use graph RAG
- **LLM swap to local model** – Ollama / vLLM integration for air-gapped deployments
- **Grafana dashboard** – pre-built dashboard consuming Prometheus metrics
- **Alertmanager webhook** – receive alerts directly from Prometheus Alertmanager
- **Auth & rate limiting** – API key auth and request throttling for production use
