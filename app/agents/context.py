"""Local runtime dependencies and observability state for the finance agent."""

from dataclasses import dataclass, field

from app.models.retrieval import LocalVectorIndex
from app.models.transactions import Transaction
from app.retrieval.embeddings import EmbeddingProvider


@dataclass
class FinanceAgentContext:
    """Dependencies that function tools use locally during one agent run."""

    transactions: list[Transaction]
    retrieval_index: LocalVectorIndex
    embedding_provider: EmbeddingProvider
    tools_used: list[str] = field(default_factory=list)
    available_citations: list[str] = field(default_factory=list)

    def reset_run_state(self) -> None:
        """Clear only per-run observability state before starting an agent run."""
        self.tools_used.clear()
        self.available_citations.clear()

    def record_tool(self, tool_name: str) -> None:
        """Record a tool invocation in the order it actually occurs."""
        self.tools_used.append(tool_name)

    def record_citations(self, citations: list[str]) -> None:
        """Record citation identifiers returned by the retrieval tool."""
        for citation in citations:
            if citation not in self.available_citations:
                self.available_citations.append(citation)
