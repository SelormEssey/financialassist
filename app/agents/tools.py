"""Agents SDK function tools over deterministic transaction and retrieval services."""

from datetime import date

from agents import RunContextWrapper, function_tool

from app.agents.context import FinanceAgentContext
from app.models.retrieval import RetrievalResult
from app.models.transactions import PeriodComparison, SpendingSummary
from app.retrieval.index import retrieve
from app.tools.transactions import compare_spending_periods, summarize_spending


def summarize_transactions(
    context: FinanceAgentContext, start_date: str, end_date: str
) -> SpendingSummary:
    """Summarize debit spending for an inclusive ISO date range."""
    summary = summarize_spending(
        context.transactions,
        _parse_date(start_date, "start_date"),
        _parse_date(end_date, "end_date"),
    )
    context.record_tool("summarize_transactions")
    return summary


def compare_transaction_periods(
    context: FinanceAgentContext,
    first_start: str,
    first_end: str,
    second_start: str,
    second_end: str,
) -> PeriodComparison:
    """Compare debit spending across two inclusive ISO date ranges."""
    comparison = compare_spending_periods(
        context.transactions,
        _parse_date(first_start, "first_start"),
        _parse_date(first_end, "first_end"),
        _parse_date(second_start, "second_start"),
        _parse_date(second_end, "second_end"),
    )
    context.record_tool("compare_transaction_periods")
    return comparison


def search_financial_documents(
    context: FinanceAgentContext, query: str, top_k: int = 5
) -> list[RetrievalResult]:
    """Retrieve Northstar policy evidence with metadata-derived citations."""
    results = retrieve(query, context.retrieval_index, context.embedding_provider, top_k)
    context.record_citations([result.citation for result in results])
    context.record_tool("search_financial_documents")
    return results


@function_tool(name_override="summarize_transactions")
def summarize_transactions_tool(
    context: RunContextWrapper[FinanceAgentContext], start_date: str, end_date: str
) -> SpendingSummary:
    """Use for spending totals, categories, merchants, averages, and date ranges."""
    return summarize_transactions(context.context, start_date, end_date)


@function_tool(name_override="compare_transaction_periods")
def compare_transaction_periods_tool(
    context: RunContextWrapper[FinanceAgentContext],
    first_start: str,
    first_end: str,
    second_start: str,
    second_end: str,
) -> PeriodComparison:
    """Use for changes in spending between two date ranges."""
    return compare_transaction_periods(
        context.context, first_start, first_end, second_start, second_end
    )


@function_tool(name_override="search_financial_documents")
def search_financial_documents_tool(
    context: RunContextWrapper[FinanceAgentContext], query: str, top_k: int = 5
) -> list[RetrievalResult]:
    """Use for Northstar policy, fee, fraud, rewards, card, and account questions."""
    return search_financial_documents(context.context, query, top_k)


def finance_agent_tools() -> list[object]:
    """Return the three controlled function tools exposed to the finance agent."""
    return [
        summarize_transactions_tool,
        compare_transaction_periods_tool,
        search_financial_documents_tool,
    ]


def _parse_date(value: str, parameter_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{parameter_name} must use YYYY-MM-DD format") from error
