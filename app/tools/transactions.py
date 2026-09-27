"""Deterministic CSV loading and spending calculations for financial transactions."""

import csv
from collections import defaultdict
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from app.models.transactions import (
    PeriodComparison,
    SpendingBreakdownItem,
    SpendingSummary,
    Transaction,
)

REQUIRED_TRANSACTION_COLUMNS = (
    "transaction_id",
    "date",
    "merchant",
    "category",
    "amount",
    "transaction_type",
)
MONEY_QUANTUM = Decimal("0.01")


class TransactionCsvError(ValueError):
    """Raised when a transaction CSV cannot be parsed into valid transactions."""


def load_transactions_csv(path: str | Path) -> list[Transaction]:
    """Load and validate transaction rows from a CSV file.

    The CSV must include every column in ``REQUIRED_TRANSACTION_COLUMNS``. Every
    data row is validated; malformed input raises ``TransactionCsvError`` rather
    than being omitted from the result.
    """
    csv_path = Path(path)
    try:
        file_handle = csv_path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise TransactionCsvError(f"Unable to read transaction CSV: {csv_path}") from error

    with file_handle:
        reader = csv.DictReader(file_handle)
        _validate_columns(reader.fieldnames, csv_path)
        transactions = []
        for line_number, row in enumerate(reader, start=2):
            transactions.append(_parse_transaction_row(row, line_number, csv_path))

    return transactions


def summarize_spending(
    transactions: Iterable[Transaction], start_date: date, end_date: date
) -> SpendingSummary:
    """Summarize debit spending in an inclusive date range using ``Decimal`` math."""
    _validate_date_range(start_date, end_date)
    selected = [
        transaction
        for transaction in transactions
        if transaction.transaction_type == "debit" and start_date <= transaction.date <= end_date
    ]
    total_spent = sum((transaction.amount for transaction in selected), Decimal("0"))
    count = len(selected)
    average = (
        (total_spent / count).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        if count
        else Decimal("0")
    )

    return SpendingSummary(
        start_date=start_date,
        end_date=end_date,
        total_spent=total_spent,
        transaction_count=count,
        average_transaction=average,
        by_category=_breakdown(selected, "category"),
        by_merchant=_breakdown(selected, "merchant"),
    )


def compare_spending_periods(
    transactions: Iterable[Transaction],
    first_start: date,
    first_end: date,
    second_start: date,
    second_end: date,
) -> PeriodComparison:
    """Compare debit spending for two inclusive date ranges.

    ``percentage_change`` is ``None`` when the first period has zero spending,
    because a percentage change from zero is undefined.
    """
    transaction_list = list(transactions)
    first_period = summarize_spending(transaction_list, first_start, first_end)
    second_period = summarize_spending(transaction_list, second_start, second_end)
    absolute_change = second_period.total_spent - first_period.total_spent
    percentage_change = None
    if first_period.total_spent != Decimal("0"):
        percentage_change = ((absolute_change / first_period.total_spent) * Decimal("100")).quantize(
            MONEY_QUANTUM, rounding=ROUND_HALF_UP
        )

    first_categories = {item.name: item.amount for item in first_period.by_category}
    second_categories = {item.name: item.amount for item in second_period.by_category}
    category_changes = [
        SpendingBreakdownItem(
            name=category,
            amount=second_categories.get(category, Decimal("0"))
            - first_categories.get(category, Decimal("0")),
        )
        for category in first_categories.keys() | second_categories.keys()
    ]
    category_changes.sort(key=lambda item: (-abs(item.amount), item.name))

    return PeriodComparison(
        first_period=first_period,
        second_period=second_period,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        category_changes=category_changes,
    )


def _validate_columns(fieldnames: list[str] | None, csv_path: Path) -> None:
    if fieldnames is None:
        raise TransactionCsvError(f"Transaction CSV has no header row: {csv_path}")
    missing_columns = sorted(set(REQUIRED_TRANSACTION_COLUMNS) - set(fieldnames))
    if missing_columns:
        columns = ", ".join(missing_columns)
        raise TransactionCsvError(f"Transaction CSV is missing required columns: {columns}")


def _parse_transaction_row(
    row: dict[str, str | None], line_number: int, csv_path: Path
) -> Transaction:
    if None in row:
        raise TransactionCsvError(
            f"Unexpected extra values on line {line_number} in {csv_path}"
        )

    try:
        transaction_date = date.fromisoformat(_required_value(row, "date"))
    except ValueError as error:
        raise TransactionCsvError(
            f"Invalid date on line {line_number} in {csv_path}; use YYYY-MM-DD"
        ) from error

    try:
        amount = Decimal(_required_value(row, "amount"))
    except (InvalidOperation, ValueError) as error:
        raise TransactionCsvError(
            f"Invalid amount on line {line_number} in {csv_path}; use a decimal value"
        ) from error

    try:
        return Transaction(
            transaction_id=_required_value(row, "transaction_id"),
            date=transaction_date,
            merchant=_required_value(row, "merchant"),
            category=_required_value(row, "category"),
            amount=amount,
            transaction_type=_required_value(row, "transaction_type"),
        )
    except (ValidationError, ValueError) as error:
        raise TransactionCsvError(f"Invalid transaction on line {line_number} in {csv_path}: {error}") from error


def _required_value(row: dict[str, str | None], column: str) -> str:
    value = row[column]
    if value is None or not value.strip():
        raise ValueError(f"{column} must not be blank")
    return value.strip()


def _validate_date_range(start_date: date, end_date: date) -> None:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")


def _breakdown(
    transactions: Iterable[Transaction], attribute: str
) -> list[SpendingBreakdownItem]:
    totals: defaultdict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for transaction in transactions:
        totals[str(getattr(transaction, attribute))] += transaction.amount
    breakdown = [SpendingBreakdownItem(name=name, amount=amount) for name, amount in totals.items()]
    breakdown.sort(key=lambda item: (-item.amount, item.name))
    return breakdown
