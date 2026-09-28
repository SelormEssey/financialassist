"""Offline tests for finance-agent context, tools, and response validation."""

from datetime import date
from decimal import Decimal

import pytest

from app.agents.context import FinanceAgentContext
from app.agents.finance_agent import create_finance_agent, run_finance_agent
from app.agents.tools import (
    compare_transaction_periods,
    search_financial_documents,
    summarize_transactions,
)
from app.models.agent import FinanceAgentResponse
from app.models.retrieval import DocumentChunk
from app.models.transactions import Transaction
from app.retrieval.index import build_index


class FakeEmbeddingProvider:
    """Deterministic vectors for real local retrieval in agent tests."""

    model_name = "fake-embedding-model"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


class FakeRunner:
    """Run controlled local tool calls without invoking an LLM or network."""

    def __init__(self, callback) -> None:
        self.callback = callback

    def run(self, agent, question: str, context: FinanceAgentContext):
        return self.callback(context)


@pytest.fixture
def context() -> FinanceAgentContext:
    transactions = [
        Transaction(
            transaction_id="july",
            date=date(2026, 7, 1),
            merchant="Cafe",
            category="Dining",
            amount=Decimal("20.00"),
            transaction_type="debit",
        ),
        Transaction(
            transaction_id="august",
            date=date(2026, 8, 1),
            merchant="Market",
            category="Groceries",
            amount=Decimal("50.00"),
            transaction_type="debit",
        ),
    ]
    chunk = DocumentChunk(
        chunk_id="credit-late-payments",
        document_id="northstar-credit-card",
        title="Northstar Credit Card",
        section="Late Payments",
        source="northstar_credit_card.md",
        text="late payment fee and penalty APR",
    )
    provider = FakeEmbeddingProvider(
        {
            "late payment fee and penalty APR": [1.0, 0.0],
            "late payment": [1.0, 0.0],
        }
    )
    return FinanceAgentContext(
        transactions=transactions,
        retrieval_index=build_index([chunk], provider, provider.model_name),
        embedding_provider=provider,
    )


def test_transaction_tools_use_context_transactions(context: FinanceAgentContext) -> None:
    summary = summarize_transactions(context, "2026-08-01", "2026-08-31")
    comparison = compare_transaction_periods(
        context, "2026-07-01", "2026-07-31", "2026-08-01", "2026-08-31"
    )

    assert summary.total_spent == Decimal("50.00")
    assert comparison.absolute_change == Decimal("30.00")
    assert context.tools_used == [
        "summarize_transactions",
        "compare_transaction_periods",
    ]


def test_retrieval_tool_uses_context_index_and_tracks_citations(
    context: FinanceAgentContext,
) -> None:
    results = search_financial_documents(context, "late payment")

    assert results[0].citation == "[northstar-credit-card#late-payments]"
    assert context.tools_used == ["search_financial_documents"]
    assert context.available_citations == ["[northstar-credit-card#late-payments]"]


def test_agent_runner_rejects_blank_question(context: FinanceAgentContext) -> None:
    with pytest.raises(ValueError, match="question must not be blank"):
        run_finance_agent("  ", context, "test-agent-model")


def test_agent_response_uses_observed_tools_and_validates_citations(
    context: FinanceAgentContext,
) -> None:
    def run_with_retrieval(agent_context: FinanceAgentContext) -> FinanceAgentResponse:
        search_financial_documents(agent_context, "late payment")
        return FinanceAgentResponse(
            answer=(
                "Northstar charges a late fee [northstar-credit-card#late-payments] "
                "[fabricated-policy#invented]."
            ),
            citations=[
                "[fabricated-policy#invented]",
                "[northstar-credit-card#late-payments]",
            ],
            tools_used=["made_up_tool"],
        )

    response = run_finance_agent(
        "What is the late payment policy?",
        context,
        "test-agent-model",
        runner=FakeRunner(run_with_retrieval),
    )

    assert response.citations == ["[northstar-credit-card#late-payments]"]
    assert "[fabricated-policy#invented]" not in response.answer
    assert response.tools_used == ["search_financial_documents"]
    assert response.source_count == 1
    assert response.has_retrieval is True
    assert response.has_transaction_analysis is False


def test_reused_context_resets_transaction_tool_history(
    context: FinanceAgentContext,
) -> None:
    first_response = run_finance_agent(
        "How much did I spend in August?",
        context,
        "test-agent-model",
        runner=FakeRunner(
            lambda agent_context: (
                summarize_transactions(agent_context, "2026-08-01", "2026-08-31"),
                FinanceAgentResponse(answer="You spent $50.00."),
            )[1]
        ),
    )
    second_response = run_finance_agent(
        "What is the late payment policy?",
        context,
        "test-agent-model",
        runner=FakeRunner(
            lambda agent_context: (
                search_financial_documents(agent_context, "late payment"),
                FinanceAgentResponse(
                    answer="Northstar charges a late fee.",
                    citations=["[northstar-credit-card#late-payments]"],
                ),
            )[1]
        ),
    )

    assert first_response.tools_used == ["summarize_transactions"]
    assert second_response.tools_used == ["search_financial_documents"]


def test_reused_context_does_not_leak_citations_or_old_citation_text(
    context: FinanceAgentContext,
) -> None:
    old_citation = "[northstar-credit-card#late-payments]"
    run_finance_agent(
        "What is the late payment policy?",
        context,
        "test-agent-model",
        runner=FakeRunner(
            lambda agent_context: (
                search_financial_documents(agent_context, "late payment"),
                FinanceAgentResponse(answer="Policy evidence.", citations=[old_citation]),
            )[1]
        ),
    )
    second_response = run_finance_agent(
        "How much did I spend in August?",
        context,
        "test-agent-model",
        runner=FakeRunner(
            lambda agent_context: (
                summarize_transactions(agent_context, "2026-08-01", "2026-08-31"),
                FinanceAgentResponse(
                    answer=f"You spent $50.00. {old_citation}",
                    citations=[old_citation],
                ),
            )[1]
        ),
    )

    assert second_response.citations == []
    assert old_citation not in second_response.answer
    assert second_response.tools_used == ["summarize_transactions"]


def test_model_cannot_spoof_tools_used(context: FinanceAgentContext) -> None:
    response = run_finance_agent(
        "How much did I spend in August?",
        context,
        "test-agent-model",
        runner=FakeRunner(
            lambda agent_context: (
                summarize_transactions(agent_context, "2026-08-01", "2026-08-31"),
                FinanceAgentResponse(
                    answer="You spent $50.00.",
                    tools_used=["fake_tool", "search_financial_documents"],
                ),
            )[1]
        ),
    )

    assert response.tools_used == ["summarize_transactions"]


def test_valid_citations_are_deduplicated_in_retrieval_order(
    context: FinanceAgentContext,
) -> None:
    citation = "[northstar-credit-card#late-payments]"
    response = run_finance_agent(
        "What is the late payment policy?",
        context,
        "test-agent-model",
        runner=FakeRunner(
            lambda agent_context: (
                search_financial_documents(agent_context, "late payment"),
                FinanceAgentResponse(
                    answer=f"Northstar charges a late fee {citation}",
                    citations=[citation, citation, citation],
                ),
            )[1]
        ),
    )

    assert response.citations == [citation]
    assert response.source_count == 1


def test_failed_tool_is_not_recorded_as_executed(context: FinanceAgentContext) -> None:
    with pytest.raises(ValueError, match="start_date must use YYYY-MM-DD format"):
        summarize_transactions(context, "not-a-date", "2026-08-31")

    assert context.tools_used == []


@pytest.mark.parametrize(
    ("question", "expected_tools"),
    [
        ("How much did I spend in August?", ["summarize_transactions"]),
        ("What is Northstar's late-payment policy?", ["search_financial_documents"]),
        (
            "Compare July and August spending and tell me Northstar's late-payment policy.",
            ["compare_transaction_periods", "search_financial_documents"],
        ),
    ],
)
def test_deterministic_orchestration_seam_tracks_expected_tools(
    context: FinanceAgentContext, question: str, expected_tools: list[str]
) -> None:
    def run_selected_tools(agent_context: FinanceAgentContext) -> FinanceAgentResponse:
        if "Compare" in question:
            compare_transaction_periods(
                agent_context,
                "2026-07-01",
                "2026-07-31",
                "2026-08-01",
                "2026-08-31",
            )
        elif "spend" in question:
            summarize_transactions(agent_context, "2026-08-01", "2026-08-31")
        if "Northstar" in question:
            search_financial_documents(agent_context, "late payment")
        return FinanceAgentResponse(answer="Controlled test response")

    response = run_finance_agent(
        question, context, "test-agent-model", runner=FakeRunner(run_selected_tools)
    )

    assert response.tools_used == expected_tools


def test_agent_parses_json_runner_output(context: FinanceAgentContext) -> None:
    response = run_finance_agent(
        "No tool needed",
        context,
        "test-agent-model",
        runner=FakeRunner(lambda _: '{"answer":"No data available."}'),
    )

    assert response.answer == "No data available."
    assert response.tools_used == []


def test_agent_exposes_exactly_three_function_tools() -> None:
    agent = create_finance_agent("test-agent-model")

    assert [tool.name for tool in agent.tools] == [
        "summarize_transactions",
        "compare_transaction_periods",
        "search_financial_documents",
    ]
