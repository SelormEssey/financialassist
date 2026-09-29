"""Evaluation orchestration for safe offline and explicitly requested live modes."""

from datetime import UTC, datetime
from pathlib import Path

from app.agents.context import FinanceAgentContext
from app.config import Settings
from app.evals.agent import evaluate_live_agent_cases
from app.evals.dataset import load_golden_dataset
from app.evals.models import EvalCase, EvalCaseResult, EvalMetricSummary, EvalRunReport
from app.evals.retrieval import evaluate_retrieval_case
from app.evals.transactions import evaluate_transaction_case
from app.retrieval.embeddings import OpenAIEmbeddingProvider
from app.retrieval.index import load_index
from app.tools.transactions import load_transactions_csv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPOSITORY_ROOT / "evals" / "golden_dataset.jsonl"
DEFAULT_TRANSACTIONS_PATH = REPOSITORY_ROOT / "data" / "sample_transactions" / "sample_transactions.csv"
DEFAULT_INDEX_PATH = REPOSITORY_ROOT / "data" / "index" / "retrieval_index.json"

_LIVE_ONLY_METRICS = (
    "retrieval_recall_at_1",
    "retrieval_recall_at_3",
    "retrieval_recall_at_5",
    "exact_tool_selection_accuracy",
    "citation_precision",
    "citation_recall",
    "unexpected_citation_count",
    "unsupported_citation_count",
    "policy_grounding_rate",
)


def run_deterministic_evaluations(dataset_path: Path = DEFAULT_DATASET_PATH) -> EvalRunReport:
    """Evaluate transaction tools only; every other golden case is explicitly skipped."""
    cases = load_golden_dataset(dataset_path)
    transactions = load_transactions_csv(DEFAULT_TRANSACTIONS_PATH)
    results = [
        evaluate_transaction_case(case, transactions)
        if case.transaction_evaluation is not None
        else _skipped(case, "Deterministic mode evaluates transaction arithmetic only")
        for case in cases
    ]
    executed = [result for result in results if result.status != "skipped"]
    checks = sum(int(result.metrics["transaction_numeric_total"]) for result in executed)
    correct = sum(int(result.metrics["transaction_numeric_checks"]) for result in executed)
    summaries = [
        EvalMetricSummary(
            name="transaction_numeric_accuracy",
            availability="evaluated",
            numerator=correct,
            denominator=checks,
            percentage=(correct / checks * 100) if checks else None,
            aggregation="pooled exact Decimal checks",
        ),
        *[_unavailable(name) for name in _LIVE_ONLY_METRICS],
    ]
    return _report("deterministic", cases, results, summaries)


def run_live_retrieval_evaluations(
    dataset_path: Path = DEFAULT_DATASET_PATH, index_path: Path = DEFAULT_INDEX_PATH
) -> EvalRunReport:
    """Evaluate persisted OpenAI-backed retrieval; non-retrieval cases are skipped."""
    settings = _live_settings(index_path)
    provider = OpenAIEmbeddingProvider(settings.embedding_model, settings.openai_api_key)
    index = load_index(index_path)
    cases = load_golden_dataset(dataset_path)
    results = [
        evaluate_retrieval_case(case, index, provider)
        if case.expected_citations
        else _skipped(case, "Live retrieval mode evaluates cases with expected citations")
        for case in cases
    ]
    executed = [result for result in results if result.status != "skipped"]
    summaries = [
        _macro_metric(executed, f"retrieval_recall_at_{k}", f"retrieval_recall_at_{k}")
        for k in (1, 3, 5)
    ] + [
        _unavailable("exact_tool_selection_accuracy"),
        _unavailable("citation_precision"),
        _unavailable("citation_recall"),
        _unavailable("unexpected_citation_count"),
        _unavailable("unsupported_citation_count"),
        _unavailable("policy_grounding_rate"),
    ]
    return _report("live-retrieval", cases, results, summaries)


def run_live_agent_evaluations(
    dataset_path: Path = DEFAULT_DATASET_PATH, index_path: Path = DEFAULT_INDEX_PATH
) -> EvalRunReport:
    """Evaluate all cases with a new context per case and same-context provenance."""
    settings = _live_settings(index_path)
    provider = OpenAIEmbeddingProvider(settings.embedding_model, settings.openai_api_key)
    index = load_index(index_path)
    transactions = load_transactions_csv(DEFAULT_TRANSACTIONS_PATH)
    cases = load_golden_dataset(dataset_path)
    results = evaluate_live_agent_cases(
        cases,
        context_factory=lambda: FinanceAgentContext(
            transactions=transactions, retrieval_index=index, embedding_provider=provider
        ),
        model=settings.agent_model,
    )
    citation_results = [result for case, result in zip(cases, results, strict=True) if case.expected_citations]
    summaries = [
        _boolean_metric(results, "exact_tool_selection", "exact_tool_selection_accuracy"),
        _macro_metric(citation_results, "citation_precision", "citation_precision"),
        _macro_metric(citation_results, "citation_recall", "citation_recall"),
        _count_metric(results, "unexpected_citation_count"),
        _count_metric(results, "unsupported_citation_count"),
        _boolean_metric(citation_results, "policy_grounded", "policy_grounding_rate"),
    ] + [_unavailable(f"retrieval_recall_at_{k}") for k in (1, 3, 5)]
    return _report("live-agent", cases, results, summaries)


def _live_settings(index_path: Path) -> Settings:
    settings = Settings()
    if not settings.openai_api_key:
        raise RuntimeError("Live evaluation requires OPENAI_API_KEY. Run deterministic evaluation without live flags.")
    if not index_path.is_file():
        raise RuntimeError(
            f"Live evaluation requires a persisted retrieval index at {index_path}. "
            "Run scripts/build_retrieval_index.py first."
        )
    return settings


def _skipped(case: EvalCase, reason: str) -> EvalCaseResult:
    return EvalCaseResult(case_id=case.case_id, status="skipped", skip_reason=reason)


def _unavailable(name: str) -> EvalMetricSummary:
    return EvalMetricSummary(name=name, availability="unavailable")


def _report(
    mode: str,
    cases: list[EvalCase],
    results: list[EvalCaseResult],
    summaries: list[EvalMetricSummary],
) -> EvalRunReport:
    return EvalRunReport(
        timestamp=datetime.now(UTC),
        mode=mode,
        dataset_total=len(cases),
        executed_cases=sum(result.status != "skipped" for result in results),
        passed_cases=sum(result.status == "passed" for result in results),
        failed_cases=sum(result.status == "failed" for result in results),
        skipped_cases=sum(result.status == "skipped" for result in results),
        metric_summaries=summaries,
        results=results,
    )


def _macro_metric(results: list[EvalCaseResult], key: str, name: str) -> EvalMetricSummary:
    values = [float(result.metrics[key]) for result in results if result.metrics.get(key) is not None]
    return EvalMetricSummary(
        name=name,
        availability="evaluated" if values else "unavailable",
        denominator=len(values) if values else None,
        percentage=(sum(values) / len(values) * 100) if values else None,
        aggregation="macro-average across eligible cases" if values else None,
    )


def _boolean_metric(results: list[EvalCaseResult], key: str, name: str) -> EvalMetricSummary:
    values = [bool(result.metrics[key]) for result in results if key in result.metrics]
    return EvalMetricSummary(
        name=name,
        availability="evaluated" if values else "unavailable",
        numerator=sum(values) if values else None,
        denominator=len(values) if values else None,
        percentage=(sum(values) / len(values) * 100) if values else None,
        aggregation="eligible cases" if values else None,
    )


def _count_metric(results: list[EvalCaseResult], key: str) -> EvalMetricSummary:
    values = [int(result.metrics[key]) for result in results if result.metrics.get(key) is not None]
    return EvalMetricSummary(
        name=key,
        availability="evaluated" if values else "unavailable",
        numerator=sum(values) if values else None,
        denominator=len(values) if values else None,
        aggregation="sum across cases with retrieval provenance" if values else None,
    )
