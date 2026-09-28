# financialassist

An agentic financial assistant that combines retrieval, structured financial analysis, tool use, evaluation, and a production-style API.

**Development is in progress.** Phase 4 adds a single finance agent that selects
deterministic transaction tools, policy retrieval, or both, and returns grounded
natural-language responses with validated citation metadata. The final `/ask`
API and evaluation framework are not implemented yet.

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
| `FINANCIALASSIST_AGENT_MODEL` | `gpt-5.6-terra` | OpenAI model used by the finance agent. |

`.env` is ignored by Git. `.env.example` documents the supported settings.
No API key is required to run the API or test suite. An API key is needed for
the retrieval scripts and the finance-agent demo, which use OpenAI SDKs.

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

## Finance agent

Phase 4 adds one controlled finance agent using the OpenAI Agents SDK. It has
three local function tools: `summarize_transactions`,
`compare_transaction_periods`, and `search_financial_documents`. The agent uses
transaction tools for personal spending facts, document retrieval for Northstar
policy evidence, and both tools when a question requires both kinds of evidence.

```text
User question
      ↓
Finance Agent
   /        \
  ↓          ↓
Transaction  Retrieval
Tools        Tool
  ↓          ↓
Exact math   Evidence + citations
   \        /
    \      /
 Grounded response
```

Transaction calculations remain deterministic Python with `Decimal`; the model
does not calculate financial totals itself. The runtime context holds local
transactions, the retrieval index, and the embedding provider. It is passed to
function tools through the Agents SDK and is not inserted into the model prompt.
At the start of each run, only its observability state is reset; these local
dependencies remain available for reuse.

Each tool records its stable name only after it completes successfully. The
retrieval tool records only citations it actually returned. Before the final
response is returned, citation identifiers are filtered against the evidence
from that run, deduplicated in retrieval order, and any unrecognized
citation-like identifier in the answer is removed.

With `OPENAI_API_KEY` and a built retrieval index, run a local demonstration:

```bash
uv run python scripts/ask_financialassist.py "How much did I spend in August?"
```

The script fails clearly when the API key or persisted retrieval index is absent.
It prints the answer, observed tools, and validated citations. It does not expose
a FastAPI endpoint.

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
| `app/agents/` | One finance agent, typed runtime context, controlled function tools, and runner service. |
| `app/tools/` | Deterministic transaction CSV loading, spending summaries, and period comparisons. |
| `app/retrieval/` | Markdown loading, chunking, embeddings abstraction, local vector search, and index persistence. |
| `app/evals/` | Placeholder for future retrieval, tool selection, numerical accuracy, and groundedness evaluations. |
| `app/models/` | Pydantic schemas for health, transaction analysis, retrieval, and agent responses. |
| `tests/` | API and configuration tests. |
| `data/documents/` | Synthetic Northstar Financial policy documents for retrieval. |
| `data/sample_transactions/` | Synthetic CSV used to demonstrate transaction analysis. |
| `scripts/` | Retrieval-index build/query and local finance-agent demonstration commands. |
| `.github/workflows/` | Empty directory reserved for future GitHub Actions CI. |

Empty directories contain `.gitkeep` files so Git preserves the structure.
Implemented: FastAPI scaffold, deterministic transaction analysis, semantic
financial document retrieval, a finance agent, tool selection, transaction
tools, a retrieval tool, grounded natural-language responses, citation
validation, and tool-usage tracking.

Not yet implemented: a final `/ask` FastAPI endpoint, the evaluation framework,
GitHub Actions CI, Docker support, or a frontend. No database, vector database,
LangChain, or LangGraph has been added.
