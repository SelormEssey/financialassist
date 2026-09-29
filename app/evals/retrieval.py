"""Retrieval evaluation that measures citation Recall@k, never semantic "accuracy"."""

from app.evals.metrics import recall_at_k
from app.evals.models import EvalCase, EvalCaseResult
from app.models.retrieval import LocalVectorIndex
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.index import retrieve


def evaluate_retrieval_case(
    case: EvalCase,
    index: LocalVectorIndex,
    embedding_provider: EmbeddingProvider,
    top_k: int = 5,
) -> EvalCaseResult:
    """Retrieve one case and score relevant citation identifiers at 1, 3, and 5."""
    results = retrieve(case.question, index, embedding_provider, top_k=top_k)
    citations = [result.citation for result in results]
    expected = set(case.expected_citations)
    recalls = {f"retrieval_recall_at_{k}": recall_at_k(expected, citations, k) for k in (1, 3, 5)}
    passed = recalls["retrieval_recall_at_5"] == 1.0
    return EvalCaseResult(
        case_id=case.case_id,
        status="passed" if passed else "failed",
        metrics=recalls,
        observed_citations=citations,
        failure_reasons=[] if passed else ["Not all expected citations appeared in the first five results"],
    )
