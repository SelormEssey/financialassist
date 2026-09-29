"""Offline deterministic transaction evaluation against hand-authored expectations."""

from decimal import Decimal

from app.evals.models import EvalCase, EvalCaseResult
from app.models.transactions import Transaction
from app.tools.transactions import compare_spending_periods, summarize_spending


def evaluate_transaction_case(case: EvalCase, transactions: list[Transaction]) -> EvalCaseResult:
    """Run the real deterministic financial operation and compare Decimal fields exactly."""
    if case.transaction_evaluation is None:
        raise ValueError(f"Case '{case.case_id}' has no transaction evaluation")
    spec = case.transaction_evaluation
    if spec.operation == "summary":
        observed = summarize_spending(transactions, spec.first_start, spec.first_end)
        values: dict[str, Decimal | None] = {
            "total_spent": observed.total_spent,
            "transaction_count": Decimal(observed.transaction_count),
            "average_transaction": observed.average_transaction,
        }
    else:
        if spec.second_start is None or spec.second_end is None:
            raise ValueError(f"Comparison case '{case.case_id}' requires a second date range")
        observed = compare_spending_periods(
            transactions, spec.first_start, spec.first_end, spec.second_start, spec.second_end
        )
        values = {
            "first_total_spent": observed.first_period.total_spent,
            "second_total_spent": observed.second_period.total_spent,
            "absolute_change": observed.absolute_change,
            "percentage_change": observed.percentage_change,
        }

    failures = [
        f"{name}: expected {expected}, observed {values.get(name)}"
        for name, expected in case.expected_numeric.items()
        if values.get(name) != expected
    ]
    checks = len(case.expected_numeric)
    passed = not failures
    return EvalCaseResult(
        case_id=case.case_id,
        status="passed" if passed else "failed",
        metrics={"transaction_numeric_checks": checks - len(failures), "transaction_numeric_total": checks},
        failure_reasons=failures,
    )
