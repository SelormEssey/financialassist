"""Single-agent orchestration for transaction analysis and policy retrieval."""

import re
from typing import Protocol

from agents import Agent, Runner

from app.agents.context import FinanceAgentContext
from app.agents.tools import finance_agent_tools
from app.models.agent import FinanceAgentResponse

SYSTEM_INSTRUCTIONS = (
    "You are financialassist, a controlled finance assistant. Use transaction tools "
    "for spending, merchants, categories, totals, averages, and period changes. "
    "Use document search for Northstar Financial policies, fees, fraud, disputes, "
    "rewards, and account or card terms. Use both when a question needs both "
    "personal transaction facts and policy evidence. Never calculate transaction "
    "arithmetic yourself when a transaction tool can do it. Do not invent transaction "
    "data, policy details, or citations. Cite policy evidence only with citation "
    "identifiers returned by document search. Distinguish transaction facts from policy "
    "details, say when available data is insufficient, and do not provide investment, "
    "tax, or legal advice."
)

_CITATION_PATTERN = re.compile(r"\[[a-z0-9][a-z0-9-]*#[a-z0-9][a-z0-9-]*\]")


class FinanceAgentRunner(Protocol):
    """A seam that keeps orchestration tests deterministic and offline."""

    def run(
        self, agent: Agent[FinanceAgentContext], question: str, context: FinanceAgentContext
    ) -> FinanceAgentResponse | str:
        """Run the agent and return structured output or JSON text."""


class AgentsSdkRunner:
    """Production runner backed by the official OpenAI Agents SDK."""

    def run(
        self, agent: Agent[FinanceAgentContext], question: str, context: FinanceAgentContext
    ) -> FinanceAgentResponse | str:
        """Run the configured agent synchronously through the SDK Runner."""
        return Runner.run_sync(agent, question, context=context).final_output


def create_finance_agent(model: str) -> Agent[FinanceAgentContext]:
    """Create the single controlled agent and its three local function tools."""
    if not model.strip():
        raise ValueError("agent model must not be blank")
    return Agent[FinanceAgentContext](
        name="FinancialAssist",
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        tools=finance_agent_tools(),
        output_type=FinanceAgentResponse,
    )


def run_finance_agent(
    question: str,
    context: FinanceAgentContext,
    model: str,
    runner: FinanceAgentRunner | None = None,
) -> FinanceAgentResponse:
    """Run one grounded turn and validate observed evidence metadata."""
    context.reset_run_state()
    if not question.strip():
        raise ValueError("question must not be blank")

    agent = create_finance_agent(model)
    raw_response = (runner or AgentsSdkRunner()).run(agent, question, context)
    return _validate_response(_parse_response(raw_response), context)


def _parse_response(raw_response: FinanceAgentResponse | str) -> FinanceAgentResponse:
    if isinstance(raw_response, FinanceAgentResponse):
        return raw_response
    try:
        return FinanceAgentResponse.model_validate_json(raw_response)
    except ValueError as error:
        raise ValueError("agent runner returned an invalid finance-agent response") from error


def _validate_response(
    response: FinanceAgentResponse, context: FinanceAgentContext
) -> FinanceAgentResponse:
    requested_citations = set(response.citations)
    allowed_citations = set(context.available_citations)
    citations = [
        citation
        for citation in context.available_citations
        if citation in requested_citations
    ]
    answer = _CITATION_PATTERN.sub(
        lambda match: match.group(0) if match.group(0) in allowed_citations else "",
        response.answer,
    )
    tools_used = list(context.tools_used)
    return response.model_copy(
        update={
            "answer": answer,
            "citations": citations,
            "tools_used": tools_used,
            "source_count": len(citations),
            "has_transaction_analysis": any(
                tool_name in {"summarize_transactions", "compare_transaction_periods"}
                for tool_name in tools_used
            ),
            "has_retrieval": "search_financial_documents" in tools_used,
        }
    )
