"""Typed grounded output for the finance agent."""

from pydantic import BaseModel, Field


class FinanceAgentResponse(BaseModel):
    """An inspectable answer with validated citations and observed tool usage."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    source_count: int = 0
    has_transaction_analysis: bool = False
    has_retrieval: bool = False
