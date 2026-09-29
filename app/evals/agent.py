"""Live finance-agent response scoring using runtime observations only."""

from collections.abc import Callable, Sequence

from app.agents.context import FinanceAgentContext
from app.agents.finance_agent import FinanceAgentRunner, run_finance_agent
from app.evals.metrics import (
    citation_precision,
    citation_recall,
    exact_tool_selection,
    policy_grounded,
    unexpected_citation_count,
    unsupported_citation_count,
)
from app.evals.models import EvalCase, EvalCaseResult
from app.models.agent import FinanceAgentResponse


def evaluate_agent_response(
    case: EvalCase,
    response: FinanceAgentResponse,
    retrieved_citations: Sequence[str] | None = None,
) -> EvalCaseResult:
    """Score final validated citations and, when present, same-run retrieval provenance."""
    expected_citations = set(case.expected_citations)
    tool_match = exact_tool_selection(case.expected_tools, response.tools_used)
    precision = citation_precision(expected_citations, response.citations)
    recall = citation_recall(expected_citations, response.citations)
    unexpected = unexpected_citation_count(expected_citations, response.citations)
    unsupported = unsupported_citation_count(response.citations, retrieved_citations)
    metrics: dict[str, bool | float | None] = {
        "exact_tool_selection": tool_match,
        "citation_precision": precision,
        "citation_recall": recall,
        "unexpected_citation_count": float(unexpected),
        "unsupported_citation_count": float(unsupported) if unsupported is not None else None,
    }
    if expected_citations:
        metrics["policy_grounded"] = policy_grounded(
            expected_citations, response.tools_used, response.citations
        )
    if case.expect_no_supported_policy_answer or case.expect_insufficient_data:
        metrics["unsupported_case_structural_check"] = not response.citations

    failures: list[str] = []
    if not tool_match:
        failures.append(f"tools: expected {case.expected_tools}, observed {response.tools_used}")
    if precision != 1.0:
        failures.append("response included citations outside the expected set")
    if recall != 1.0:
        failures.append("response omitted one or more expected citations")
    if unsupported not in (None, 0):
        failures.append("response included a citation absent from retrieval provenance")
    if metrics.get("policy_grounded") is False:
        failures.append("retrieval-dependent response lacked expected retrieved evidence")
    if metrics.get("unsupported_case_structural_check") is False:
        failures.append("unsupported case returned a policy citation")
    return EvalCaseResult(
        case_id=case.case_id,
        status="passed" if not failures else "failed",
        metrics=metrics,
        observed_tools=response.tools_used,
        observed_citations=response.citations,
        retrieved_citations=list(retrieved_citations) if retrieved_citations is not None else None,
        failure_reasons=failures,
    )


def evaluate_live_agent_cases(
    cases: Sequence[EvalCase],
    context_factory: Callable[[], FinanceAgentContext],
    model: str,
    runner: FinanceAgentRunner | None = None,
) -> list[EvalCaseResult]:
    """Run every case with a fresh context and score provenance from that same context."""
    results = []
    for case in cases:
        context = context_factory()
        response = run_finance_agent(case.question, context, model, runner=runner)
        results.append(evaluate_agent_response(case, response, context.available_citations))
    return results
