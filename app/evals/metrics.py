"""Small, pure metric helpers used by evaluation runners."""

from collections.abc import Sequence


def percentage(numerator: int, denominator: int) -> float | None:
    """Return a percentage, or ``None`` when there are no eligible observations."""
    return (numerator / denominator * 100) if denominator else None


def exact_match(expected: Sequence[str], observed: Sequence[str]) -> bool:
    """Return whether ordered sequences match exactly."""
    return list(expected) == list(observed)


def recall_at_k(expected: set[str], observed: Sequence[str], k: int) -> float | None:
    """Return per-case Recall@k, unavailable when no relevant citations are expected."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not expected:
        return None
    return len(expected & set(observed[:k])) / len(expected)


def citation_precision(expected: set[str], observed: Sequence[str]) -> float:
    """Relevant expected citations returned divided by total final citations returned."""
    if not observed:
        return 1.0 if not expected else 0.0
    return sum(citation in expected for citation in observed) / len(observed)


def citation_recall(expected: set[str], observed: Sequence[str]) -> float:
    """Relevant expected citations returned divided by expected citations."""
    if not expected:
        return 1.0 if not observed else 0.0
    return len(expected & set(observed)) / len(expected)


def unexpected_citation_count(expected: set[str], observed: Sequence[str]) -> int:
    """Count final citations outside the golden expected-citation set."""
    return sum(citation not in expected for citation in observed)


def unsupported_citation_count(
    observed: Sequence[str], retrieved_citations: Sequence[str] | None
) -> int | None:
    """Count final citations absent from same-run retrieval provenance, if available."""
    if retrieved_citations is None:
        return None
    provenance = set(retrieved_citations)
    return sum(citation not in provenance for citation in observed)


def exact_tool_selection(expected: Sequence[str], observed: Sequence[str]) -> bool:
    """Primary tool metric: observed invocation order must exactly match golden data."""
    return exact_match(expected, observed)


def policy_grounded(expected: set[str], observed_tools: Sequence[str], observed_citations: Sequence[str]) -> bool:
    """Structural proxy: retrieval ran and returned at least one expected citation."""
    return "search_financial_documents" in observed_tools and bool(expected & set(observed_citations))
