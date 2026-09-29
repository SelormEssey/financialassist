# financialassist

An agentic financial assistant that combines retrieval, structured financial analysis, tool use, evaluation, and a production-style API.

**Development is in progress.** Phase 5 adds an inspectable evaluation framework
for deterministic arithmetic, retrieval, live agent tool selection, citations,
and structural policy support. The final `/ask` API is not implemented yet.

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

## Evaluation framework

Phase 5 adds a small evaluation framework with a hand-authored, ordered golden
dataset at `evals/golden_dataset.jsonl`. Its 25 cases are based solely on the
synthetic transaction CSV, Northstar policy documents, and implemented tools:
seven transaction cases, seven retrieval cases, seven combined cases, and four
unsupported or insufficient-evidence cases. Monetary expectations are stored as
Decimal-safe strings, rather than being calculated from the implementation at
evaluation time.

Run the safe offline evaluation with no API key:

```bash
uv run python scripts/run_evals.py
```

This evaluates real `summarize_spending()` and `compare_spending_periods()`
against the golden values, comparing `Decimal` values exactly. Arithmetic is
measured independently of an LLM answer so the result demonstrates correctness
of the financial component itself. Every report includes `dataset_total`,
`executed_cases`, `passed_cases`, `failed_cases`, and `skipped_cases`. In
deterministic mode, non-transaction cases are skipped rather than counted as
failures.

Live modes are opt-in and require both `OPENAI_API_KEY` and a persisted index at
`data/index/retrieval_index.json`:

```bash
uv run python scripts/run_evals.py --live-retrieval
uv run python scripts/run_evals.py --live-agent
```

Live retrieval reports Recall@k: relevant golden citation IDs returned in the
first *k* results divided by all relevant golden citation IDs. Recall@1,
Recall@3, and Recall@5 are calculated per eligible case and reported as the
macro-average across those cases. Fake embeddings are used only in unit tests to
validate the metric implementation and are not presented as retrieval quality.

Live agent evaluation uses final validated response citations. Citation precision
is relevant expected citations returned divided by total final citations returned;
citation recall is relevant expected citations returned divided by expected
citations. `unexpected_citation_count` counts final citations outside the golden
expected set. `unsupported_citation_count` is different: it counts final
citations absent from citations actually retrieved in that same agent run. The
live evaluator obtains this provenance from the fresh `FinanceAgentContext` used
for that case. Where provenance is unavailable, this metric is reported as not
evaluated rather than zero.

The policy-grounding proxy is structural, not an LLM truthfulness judge: for a
policy-dependent case, retrieval must run and the final validated response must
include an expected citation. Unsupported cases make explicit structural
expectations such as no policy citation; they do not fabricate facts or claim a
semantic assessment of the answer. Each live agent case receives a fresh
`FinanceAgentContext` to prevent observability state from leaking across cases.
Metrics that a mode does not execute are represented as `NOT EVALUATED` with no
percentage; deterministic mode does not claim retrieval Recall@k, live tool
selection, live citation provenance, or policy grounding results.

Reports are calculated from each run and can be saved as JSON:

```bash
uv run python scripts/run_evals.py --json-output evals/results/latest.json
```

Generated JSON reports under `evals/results/` are ignored by Git; the directory's
`.gitkeep` remains tracked. This repository intentionally does not publish
benchmark numbers until a real live evaluation is run.

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
| `app/evals/` | Typed golden-dataset loading, pure metrics, deterministic transaction evaluation, and opt-in live retrieval/agent evaluation. |
| `app/models/` | Pydantic schemas for health, transaction analysis, retrieval, and agent responses. |
| `tests/` | API and configuration tests. |
| `data/documents/` | Synthetic Northstar Financial policy documents for retrieval. |
| `data/sample_transactions/` | Synthetic CSV used to demonstrate transaction analysis. |
| `scripts/` | Retrieval-index build/query, local finance-agent demonstration, and evaluation commands. |
| `.github/workflows/` | Empty directory reserved for future GitHub Actions CI. |

Empty directories contain `.gitkeep` files so Git preserves the structure.
Implemented: FastAPI scaffold, deterministic transaction analysis, semantic
financial document retrieval, a finance agent, tool selection, transaction
tools, a retrieval tool, grounded citations, and an evaluation framework.

Not yet implemented: a final `/ask` FastAPI endpoint, GitHub Actions CI, Docker
support, or a frontend. No database, vector database,
LangChain, or LangGraph has been added.
