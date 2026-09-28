# financialassist

An agentic financial assistant that combines retrieval, structured financial analysis, tool use, evaluation, and a production-style API.

**Development is in progress.** Phase 3 adds semantic retrieval over synthetic
financial policy documents alongside the FastAPI scaffold and deterministic
transaction analysis. It returns evidence and citation-ready metadata only;
LLM response generation, agent orchestration, and grounded answers are not
implemented yet.

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
| `OPENAI_API_KEY` | — | Required only when building or querying embeddings with OpenAI. |
| `FINANCIALASSIST_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model for the local retrieval index. |

`.env` is ignored by Git. `.env.example` documents the supported settings.
No API key is required to run the API or test suite. An API key is needed only
for the retrieval scripts, which use the official OpenAI Python SDK.

## Run the tests

```bash
uv run pytest
```

The tests verify the health response, configuration defaults, `.env` loading,
environment variable precedence, transaction CSV validation, and deterministic
spending calculations.

## Transaction analysis

Structured transaction analysis is implemented in `app/tools/transactions.py`.
It uses Python's standard-library CSV support and `Decimal` arithmetic, so money
calculations are deterministic and do not depend on an LLM. There are no database
or external service dependencies.

Every transaction has this schema:

| Field | Type | Description |
| --- | --- | --- |
| `transaction_id` | string | Unique transaction identifier. |
| `date` | ISO date | Posting date in `YYYY-MM-DD` format. |
| `merchant` | string | Merchant or counterparty name. |
| `category` | string | Spending category. |
| `amount` | Decimal | Positive monetary amount. |
| `transaction_type` | `debit` or `credit` | Direction of the transaction. |

Amounts are always stored as positive values. A `debit` represents money spent;
a `credit` represents money received or refunded. Spending summaries and period
comparisons count debit transactions only. Date ranges are inclusive. If the
first comparison period has zero spending, `percentage_change` is `null` because
a percentage change from zero is undefined.

The synthetic dataset at `data/sample_transactions/sample_transactions.csv`
contains 31 transactions across July and August 2026. It includes repeated
merchants, seven spending categories, and a credit refund; it does not contain
personal financial data.

```python
from datetime import date
from pathlib import Path

from app.tools.transactions import load_transactions_csv, summarize_spending

transactions = load_transactions_csv(
    Path("data/sample_transactions/sample_transactions.csv")
)
summary = summarize_spending(transactions, date(2026, 8, 1), date(2026, 8, 31))

print(summary.total_spent)
print([(item.name, item.amount) for item in summary.by_category])
```

`load_transactions_csv()` validates the required CSV columns, ISO dates, Decimal
amounts, and transaction types. Invalid input raises `TransactionCsvError` with
the line and problem rather than silently dropping data.

## Document retrieval

Phase 3 implements transparent semantic retrieval over four concise, fully
synthetic Northstar Financial policy documents in `data/documents/`. Markdown
documents are loaded into typed records, split deterministically by section,
embedded through an injected provider, and searched with local cosine similarity.
The local index is JSON-serializable and generated indexes in `data/index/` are
ignored by Git.

```text
User query
    ↓
Embedding
    ↓
Vector similarity search
    ↓
Top-k document chunks
    ↓
Citation-ready retrieval results
```

The production provider uses OpenAI's `text-embedding-3-small` model. The
retrieval code does not construct an OpenAI client until an embedding is actually
requested, and tests inject deterministic fake vectors instead of calling an API.

Each result preserves document, section, source filename, text, score, and a
citation identifier generated from metadata. For example, the `Late Payments`
section in the credit-card document has this citation:

```text
[northstar-credit-card#late-payments]
```

Build an index when `OPENAI_API_KEY` is configured:

```bash
uv run python scripts/build_retrieval_index.py
```

Then retrieve source evidence without generating an answer:

```bash
uv run python scripts/query_retrieval.py "What happens if I make a late payment?"
```

The scripts print ranked chunks with their sources, scores, citations, and text.
They do not use an LLM to generate an answer.

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
| `app/tools/` | Deterministic transaction CSV loading, spending summaries, and period comparisons. |
| `app/retrieval/` | Markdown loading, chunking, embeddings abstraction, local vector search, and index persistence. |
| `app/evals/` | Placeholder for future retrieval, tool selection, numerical accuracy, and groundedness evaluations. |
| `app/models/` | Pydantic schemas for health, transaction analysis, and retrieval. |
| `tests/` | API and configuration tests. |
| `data/documents/` | Synthetic Northstar Financial policy documents for retrieval. |
| `data/sample_transactions/` | Synthetic CSV used to demonstrate transaction analysis. |
| `scripts/` | Retrieval-index build and query commands; reserved for future ingestion and evaluation commands. |
| `.github/workflows/` | Empty directory reserved for future GitHub Actions CI. |

Empty directories contain `.gitkeep` files so Git preserves the structure.
Implemented: FastAPI scaffold, deterministic transaction analysis, document
loading, Markdown-aware chunking, an embeddings abstraction, semantic retrieval,
and citation metadata.

Not yet implemented: LLM response generation, agent orchestration, agent tool
selection, grounded natural-language answers, the evaluation framework, a final
`/ask` API, CI, Docker support, or a frontend. No database, vector database,
LangChain, or LangGraph has been added.
