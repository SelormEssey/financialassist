"""Unit tests for deterministic transaction loading and analysis."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from app.models.transactions import Transaction
from app.tools.transactions import (
    TransactionCsvError,
    compare_spending_periods,
    load_transactions_csv,
    summarize_spending,
)


@pytest.fixture
def transactions() -> list[Transaction]:
    return [
        Transaction(
            transaction_id="one",
            date=date(2026, 7, 1),
            merchant="Cafe",
            category="Dining",
            amount=Decimal("10.10"),
            transaction_type="debit",
        ),
        Transaction(
            transaction_id="two",
            date=date(2026, 7, 15),
            merchant="Market",
            category="Groceries",
            amount=Decimal("20.20"),
            transaction_type="debit",
        ),
        Transaction(
            transaction_id="three",
            date=date(2026, 7, 20),
            merchant="Cafe",
            category="Dining",
            amount=Decimal("5.00"),
            transaction_type="credit",
        ),
        Transaction(
            transaction_id="four",
            date=date(2026, 8, 1),
            merchant="Cafe",
            category="Dining",
            amount=Decimal("30.30"),
            transaction_type="debit",
        ),
        Transaction(
            transaction_id="five",
            date=date(2026, 8, 2),
            merchant="Train",
            category="Transportation",
            amount=Decimal("9.90"),
            transaction_type="debit",
        ),
    ]


def test_load_transactions_csv_parses_typed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "transactions.csv"
    csv_path.write_text(
        "transaction_id,date,merchant,category,amount,transaction_type\n"
        "txn-1,2026-07-01,Cafe,Dining,12.34,debit\n",
        encoding="utf-8",
    )

    loaded = load_transactions_csv(csv_path)

    assert loaded == [
        Transaction(
            transaction_id="txn-1",
            date=date(2026, 7, 1),
            merchant="Cafe",
            category="Dining",
            amount=Decimal("12.34"),
            transaction_type="debit",
        )
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "transaction_id,date,merchant,category,amount\n"
            "txn-1,2026-07-01,Cafe,Dining,12.34\n",
            "missing required columns: transaction_type",
        ),
        (
            "transaction_id,date,merchant,category,amount,transaction_type\n"
            "txn-1,July 1,Cafe,Dining,12.34,debit\n",
            "Invalid date on line 2",
        ),
        (
            "transaction_id,date,merchant,category,amount,transaction_type\n"
            "txn-1,2026-07-01,Cafe,Dining,not-money,debit\n",
            "Invalid amount on line 2",
        ),
        (
            "transaction_id,date,merchant,category,amount,transaction_type\n"
            "txn-1,2026-07-01,Cafe,Dining,12.34,transfer\n",
            "Invalid transaction on line 2",
        ),
        (
            "transaction_id,date,merchant,category,amount,transaction_type\n"
            "txn-1,2026-07-01,Cafe,Dining,NaN,debit\n",
            "Invalid transaction on line 2",
        ),
    ],
)
def test_load_transactions_csv_rejects_malformed_input(
    tmp_path: Path, content: str, message: str
) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(content, encoding="utf-8")

    with pytest.raises(TransactionCsvError, match=message):
        load_transactions_csv(csv_path)


def test_summary_counts_debits_only_and_uses_inclusive_dates(
    transactions: list[Transaction],
) -> None:
    summary = summarize_spending(transactions, date(2026, 7, 1), date(2026, 7, 15))

    assert summary.total_spent == Decimal("30.30")
    assert summary.transaction_count == 2
    assert summary.average_transaction == Decimal("15.15")
    assert [item.model_dump() for item in summary.by_category] == [
        {"name": "Groceries", "amount": Decimal("20.20")},
        {"name": "Dining", "amount": Decimal("10.10")},
    ]
    assert [item.model_dump() for item in summary.by_merchant] == [
        {"name": "Market", "amount": Decimal("20.20")},
        {"name": "Cafe", "amount": Decimal("10.10")},
    ]


def test_summary_preserves_decimal_accuracy_and_excludes_credits(
    transactions: list[Transaction],
) -> None:
    summary = summarize_spending(transactions, date(2026, 7, 1), date(2026, 7, 31))

    assert summary.total_spent == Decimal("30.30")
    assert summary.average_transaction == Decimal("15.15")
    assert isinstance(summary.total_spent, Decimal)
    assert isinstance(summary.average_transaction, Decimal)


def test_summary_rounds_average_to_cents_with_half_up() -> None:
    transactions = [
        Transaction(
            transaction_id="one",
            date=date(2026, 7, 1),
            merchant="Cafe",
            category="Dining",
            amount=Decimal("10.00"),
            transaction_type="debit",
        ),
        Transaction(
            transaction_id="two",
            date=date(2026, 7, 2),
            merchant="Cafe",
            category="Dining",
            amount=Decimal("10.01"),
            transaction_type="debit",
        ),
    ]

    summary = summarize_spending(transactions, date(2026, 7, 1), date(2026, 7, 31))

    assert summary.total_spent == Decimal("20.01")
    assert summary.average_transaction == Decimal("10.01")
    assert isinstance(summary.average_transaction, Decimal)


def test_empty_period_has_zero_total_and_average(transactions: list[Transaction]) -> None:
    summary = summarize_spending(transactions, date(2026, 6, 1), date(2026, 6, 30))

    assert summary.total_spent == Decimal("0")
    assert summary.transaction_count == 0
    assert summary.average_transaction == Decimal("0")
    assert summary.by_category == []
    assert summary.by_merchant == []


def test_compare_periods_calculates_changes_by_category(
    transactions: list[Transaction],
) -> None:
    comparison = compare_spending_periods(
        transactions,
        date(2026, 7, 1),
        date(2026, 7, 31),
        date(2026, 8, 1),
        date(2026, 8, 31),
    )

    assert comparison.first_period.total_spent == Decimal("30.30")
    assert comparison.second_period.total_spent == Decimal("40.20")
    assert comparison.absolute_change == Decimal("9.90")
    assert comparison.percentage_change == Decimal("32.67")
    assert isinstance(comparison.percentage_change, Decimal)
    assert [item.model_dump() for item in comparison.category_changes] == [
        {"name": "Dining", "amount": Decimal("20.20")},
        {"name": "Groceries", "amount": Decimal("-20.20")},
        {"name": "Transportation", "amount": Decimal("9.90")},
    ]


def test_compare_periods_uses_null_percentage_for_zero_first_period(
    transactions: list[Transaction],
) -> None:
    comparison = compare_spending_periods(
        transactions,
        date(2026, 6, 1),
        date(2026, 6, 30),
        date(2026, 8, 1),
        date(2026, 8, 31),
    )

    assert comparison.absolute_change == Decimal("40.20")
    assert comparison.percentage_change is None


def test_compare_periods_rounds_percentage_to_cents_with_half_up() -> None:
    transactions = [
        Transaction(
            transaction_id="first",
            date=date(2026, 7, 1),
            merchant="Market",
            category="Groceries",
            amount=Decimal("200.00"),
            transaction_type="debit",
        ),
        Transaction(
            transaction_id="second",
            date=date(2026, 8, 1),
            merchant="Market",
            category="Groceries",
            amount=Decimal("225.01"),
            transaction_type="debit",
        ),
    ]

    comparison = compare_spending_periods(
        transactions,
        date(2026, 7, 1),
        date(2026, 7, 31),
        date(2026, 8, 1),
        date(2026, 8, 31),
    )

    assert comparison.absolute_change == Decimal("25.01")
    assert comparison.percentage_change == Decimal("12.51")
    assert isinstance(comparison.percentage_change, Decimal)


def test_invalid_date_range_is_rejected(transactions: list[Transaction]) -> None:
    with pytest.raises(ValueError, match="start_date must be on or before end_date"):
        summarize_spending(transactions, date(2026, 8, 2), date(2026, 8, 1))
