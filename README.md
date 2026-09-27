# financialassist

An agentic financial assistant that combines retrieval, structured financial analysis, tool use, evaluation, and a production-style API.

**Development is in progress.** Phase 1 provides only the Python API scaffold:
a health endpoint, environment-based configuration, and automated tests.
Agentic workflows, tool calling, retrieval, grounded answers with citations,
transaction analysis, and evaluations are planned and are not implemented yet.

## Local setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the
following commands from the repository root. The project uses Python 3.12,
selected by `.python-version` and constrained in `pyproject.toml`.

```bash
uv sync --locked
cp .env.example .env
```

`uv sync --locked` creates `.venv` and installs the locked runtime and development
dependencies. uv can download Python 3.12 if it is not already available.
Commit `uv.lock` alongside dependency changes to keep environments reproducible.

## Run the API

```bash
uv run uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`. Interactive API documentation is available
at `/docs`. The `--reload` option is intended for local development.

```bash
curl http://127.0.0.1:8000/health
```

Expected response: HTTP 200 with JSON:

```json
{"status": "healthy"}
```

This endpoint reports that the API process is running. It does not check any
external services.

## Configuration

Settings are validated by Pydantic and loaded at application startup.
Environment variables take precedence over the optional `.env` file in the
working directory; otherwise, the defaults below apply.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FINANCIALASSIST_APP_NAME` | `financialassist` | Title shown in API documentation. |
| `FINANCIALASSIST_DEBUG` | `false` | Enables FastAPI debug mode when `true`; keep disabled outside local development. |

`.env` is ignored by Git. `.env.example` documents the supported settings.
No API keys or external services are required for Phase 1.

## Run the tests

```bash
uv run pytest
```

The tests verify the health response, configuration defaults, `.env` loading,
and environment variable precedence.

## Planned architecture

The planned flow is an API request handled by an agent that can select structured
financial tools and retrieve financial documents. Future answers will use
supporting citations, and evaluations will measure retrieval quality, tool
selection, numerical accuracy, and groundedness.

| Location | Responsibility and current status |
| --- | --- |
| `app/main.py` | Creates the FastAPI application and registers the health route. |
| `app/config.py` | Loads and validates application settings. |
| `app/api/` | API routes; currently only `GET /health`. |
| `app/agents/` | Placeholder for future LLM-based agentic workflows. |
| `app/tools/` | Placeholder for future structured financial analysis and tool calling. |
| `app/retrieval/` | Placeholder for future retrieval and citation support. |
| `app/evals/` | Placeholder for future retrieval, tool selection, numerical accuracy, and groundedness evaluations. |
| `app/models/` | Pydantic schemas; currently only the health response. |
| `tests/` | API and configuration tests. |
| `data/documents/` | Empty directory reserved for financial documents. |
| `data/sample_transactions/` | Empty directory reserved for sample transactions. |
| `scripts/` | Empty directory reserved for ingestion and evaluation commands. |
| `.github/workflows/` | Empty directory reserved for future GitHub Actions CI. |

Empty directories contain `.gitkeep` files so Git preserves the structure.
OpenAI integration, agents, RAG, transaction analysis, evaluation pipelines, CI,
Docker support, and a frontend are outside the Phase 1 scaffold. No databases,
vector stores, LangChain, or LangGraph have been added.
