"""Offline tests for evaluation integrity and reporting."""

import json
from datetime import date
from decimal import Decimal

import pytest

from app.agents.context import FinanceAgentContext
from app.evals import runner as evaluation_runner
from app.evals.agent import evaluate_agent_response, evaluate_live_agent_cases
from app.evals.dataset import GoldenDatasetError, load_golden_dataset
from app.evals.metrics import citation_precision, citation_recall, recall_at_k
from app.evals.models import EvalCase, EvalMetricSummary
from app.evals.retrieval import evaluate_retrieval_case
from app.evals.runner import _count_metric, run_deterministic_evaluations
from app.evals.transactions import evaluate_transaction_case
from app.models.agent import FinanceAgentResponse
from app.models.retrieval import DocumentChunk
from app.models.transactions import Transaction
from app.retrieval.index import build_index


class FakeEmbeddingProvider:
    model_name = "fake-eval-model"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def _transaction_raw(case_id: str = "case") -> dict:
    return {
        "case_id": case_id,
        "category": "transaction",
        "question": "How much did I spend?",
        "expected_tools": ["summarize_transactions"],
        "expected_numeric": {"total_spent": "1.00"},
        "transaction_evaluation": {
            "operation": "summary",
            "first_start": "2026-08-01",
            "first_end": "2026-08-31",
        },
    }


def _case(**overrides) -> EvalCase:
    raw = _transaction_raw()
    raw.update(overrides)
    return EvalCase.model_validate(raw)


def _write_dataset(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return str(path)


def test_golden_dataset_loads_in_file_order(tmp_path) -> None:
    unsupported = {
        "case_id": "second", "category": "unsupported", "question": "Can you give tax advice?",
        "expect_insufficient_data": True,
    }
    assert [case.case_id for case in load_golden_dataset(_write_dataset(tmp_path, [_transaction_raw("first"), unsupported]))] == ["first", "second"]


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({**_transaction_raw(), "expected_toolz": []}, "Invalid evaluation case"),
        ({"case_id": "r", "category": "retrieval", "question": "Policy?"}, "Invalid evaluation case"),
        ({"case_id": "c", "category": "combined", "question": "Both?", "expected_tools": ["search_financial_documents"], "expected_citations": ["[a]" ]}, "Invalid evaluation case"),
        ({"case_id": "u", "category": "unsupported", "question": "Unknown?"}, "Invalid evaluation case"),
        ({**_transaction_raw(), "transaction_evaluation": {"operation": "comparison", "first_start": "2026-08-01", "first_end": "2026-08-31"}}, "Invalid evaluation case"),
        ({**_transaction_raw(), "expected_tools": ["compare_transaction_periods"]}, "Invalid evaluation case"),
        ({"case_id": "r", "category": "retrieval", "question": "Policy?", "expected_citations": ["[a]"], "expected_tools": []}, "Invalid evaluation case"),
        ({"case_id": "c", "category": "combined", "question": "Both?", "expected_tools": ["search_financial_documents", "summarize_transactions"], "expected_citations": ["[a]"]}, "Invalid evaluation case"),
    ],
)
def test_golden_dataset_rejects_unknown_and_incomplete_category_cases(tmp_path, row, match: str) -> None:
    with pytest.raises(GoldenDatasetError, match=match):
        load_golden_dataset(_write_dataset(tmp_path, [row]))


def test_golden_dataset_rejects_malformed_json_and_duplicate_ids(tmp_path) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(GoldenDatasetError, match="Malformed JSON"):
        load_golden_dataset(broken)
    with pytest.raises(GoldenDatasetError, match="Duplicate case_id"):
        load_golden_dataset(_write_dataset(tmp_path, [_transaction_raw(), _transaction_raw()]))


def test_transaction_evaluation_compares_decimal_values_exactly() -> None:
    transaction = Transaction(transaction_id="one", date=date(2026, 8, 1), merchant="Market", category="Food", amount=Decimal("1.005"), transaction_type="debit")
    case = _case(expected_numeric={"total_spent": "1.005", "average_transaction": "1.01"})
    result = evaluate_transaction_case(case, [transaction])
    assert result.status == "passed"


def test_recall_at_k_and_fake_retrieval_harness_are_offline() -> None:
    expected = {"[doc#a]", "[doc#b]"}
    assert recall_at_k(expected, ["[doc#a]", "[x]", "[doc#b]"], 1) == 0.5
    assert recall_at_k(expected, ["[doc#a]", "[x]", "[doc#b]"], 3) == 1.0
    assert recall_at_k(set(), ["[doc#a]"], 1) is None
    chunks = [DocumentChunk(chunk_id="a", document_id="doc", title="Doc", section="A", source="a.md", text="policy a")]
    provider = FakeEmbeddingProvider({"policy a": [1.0], "question": [1.0]})
    case = EvalCase.model_validate({"case_id": "r", "category": "retrieval", "question": "question", "expected_citations": ["[doc#a]"], "expected_tools": ["search_financial_documents"]})
    result = evaluate_retrieval_case(case, build_index(chunks, provider, provider.model_name), provider)
    assert result.metrics["retrieval_recall_at_1"] == 1.0


@pytest.mark.parametrize(
    ("returned", "provenance", "precision", "recall", "unexpected", "unsupported"),
    [
        (["[a]", "[b]"], ["[a]", "[b]"], 0.5, 1.0, 1.0, 0.0),
        (["[a]", "[fake]"], ["[a]"], 0.5, 1.0, 1.0, 1.0),
        (["[b]"], ["[a]", "[b]"], 0.0, 0.0, 1.0, 0.0),
    ],
)
def test_citation_metrics_distinguish_relevance_and_retrieval_provenance(returned, provenance, precision, recall, unexpected, unsupported) -> None:
    case = EvalCase.model_validate({"case_id": "r", "category": "retrieval", "question": "Policy?", "expected_citations": ["[a]"], "expected_tools": ["search_financial_documents"]})
    response = FinanceAgentResponse(answer="Answer", citations=returned, tools_used=["search_financial_documents"])
    result = evaluate_agent_response(case, response, provenance)
    assert result.metrics["citation_precision"] == precision
    assert result.metrics["citation_recall"] == recall
    assert result.metrics["unexpected_citation_count"] == unexpected
    assert result.metrics["unsupported_citation_count"] == unsupported


def test_unsupported_citation_metric_is_unavailable_without_provenance() -> None:
    case = EvalCase.model_validate({"case_id": "r", "category": "retrieval", "question": "Policy?", "expected_citations": ["[a]"], "expected_tools": ["search_financial_documents"]})
    result = evaluate_agent_response(case, FinanceAgentResponse(answer="Answer", citations=["[a]"], tools_used=["search_financial_documents"]))
    assert result.metrics["unsupported_citation_count"] is None


def test_run_level_citation_count_metrics_preserve_relevance_and_provenance() -> None:
    case = EvalCase.model_validate({"case_id": "r", "category": "retrieval", "question": "Policy?", "expected_citations": ["[a]"], "expected_tools": ["search_financial_documents"]})
    first = evaluate_agent_response(
        case,
        FinanceAgentResponse(answer="Answer", citations=["[a]", "[b]"], tools_used=["search_financial_documents"]),
        ["[a]", "[b]"],
    )
    second = evaluate_agent_response(
        case,
        FinanceAgentResponse(answer="Answer", citations=["[a]", "[fake]"], tools_used=["search_financial_documents"]),
        ["[a]"],
    )
    unexpected = _count_metric([first, second], "unexpected_citation_count")
    unsupported = _count_metric([first, second], "unsupported_citation_count")
    assert (unexpected.availability, unexpected.numerator, unexpected.percentage) == ("evaluated", 2, None)
    assert (unsupported.availability, unsupported.numerator, unsupported.percentage) == ("evaluated", 1, None)


def test_count_metric_renders_without_percentage_formatting() -> None:
    from scripts.run_evals import format_metric_summary

    summary = EvalMetricSummary(
        name="unsupported_citation_count",
        availability="evaluated",
        numerator=0,
        denominator=25,
        percentage=None,
    )
    assert format_metric_summary(summary) == "unsupported_citation_count: 0 across 25 eligible cases"


@pytest.mark.parametrize(
    "observed_tools",
    [
        ["summarize_transactions"],
        ["search_financial_documents", "summarize_transactions"],
        ["summarize_transactions", "search_financial_documents", "extra_tool"],
    ],
)
def test_exact_tool_selection_rejects_missing_reversed_and_extra_tools(observed_tools) -> None:
    case = EvalCase.model_validate({
        "case_id": "combined", "category": "combined", "question": "Both?",
        "expected_tools": ["summarize_transactions", "search_financial_documents"],
        "expected_citations": ["[a]"],
    })
    result = evaluate_agent_response(
        case,
        FinanceAgentResponse(answer="Answer", citations=["[a]"], tools_used=observed_tools),
        ["[a]"],
    )
    assert result.metrics["exact_tool_selection"] is False
    assert result.status == "failed"


def test_deterministic_report_skips_non_transaction_cases_and_marks_live_metrics_unavailable() -> None:
    report = run_deterministic_evaluations()
    assert (report.dataset_total, report.executed_cases, report.passed_cases, report.failed_cases, report.skipped_cases) == (25, 7, 7, 0, 18)
    assert sum(result.status == "skipped" for result in report.results) == 18
    recall = next(metric for metric in report.metric_summaries if metric.name == "retrieval_recall_at_1")
    assert recall.availability == "unavailable"
    assert recall.percentage is None
    payload = json.loads(report.model_dump_json())
    assert payload["dataset_total"] == 25
    assert payload["skipped_cases"] == 18
    assert next(metric for metric in payload["metric_summaries"] if metric["name"] == "retrieval_recall_at_1")["percentage"] is None


def test_deterministic_runner_never_initializes_live_embedding_provider(monkeypatch) -> None:
    monkeypatch.setattr(evaluation_runner, "OpenAIEmbeddingProvider", lambda *args: (_ for _ in ()).throw(AssertionError()))
    assert evaluation_runner.run_deterministic_evaluations().executed_cases == 7


def test_live_agent_evaluator_uses_fresh_context_and_same_context_provenance() -> None:
    created_contexts: list[FinanceAgentContext] = []
    provider = FakeEmbeddingProvider({"text": [1.0]})
    chunk = DocumentChunk(chunk_id="chunk", document_id="doc", title="Doc", section="Section", source="doc.md", text="text")
    index = build_index([chunk], provider, provider.model_name)

    def make_context() -> FinanceAgentContext:
        context = FinanceAgentContext(transactions=[], retrieval_index=index, embedding_provider=provider)
        context.record_citations(["[a]"])
        created_contexts.append(context)
        return context

    class FakeRunner:
        def run(self, agent, question, context):
            context.record_citations(["[a]"])
            return FinanceAgentResponse(answer="Answer", citations=["[a]"], tools_used=[])

    cases = [EvalCase.model_validate({"case_id": name, "category": "unsupported", "question": "Unknown?", "expect_insufficient_data": True}) for name in ("one", "two")]
    results = evaluate_live_agent_cases(cases, make_context, "test-model", runner=FakeRunner())
    assert len({id(context) for context in created_contexts}) == 2
    assert all(result.retrieved_citations == ["[a]"] for result in results)
