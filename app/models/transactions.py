"""Typed domain models for structured financial transactions."""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, field_validator

TransactionType = Literal["debit", "credit"]


class Transaction(BaseModel):
    """A financial transaction with a positive monetary amount."""

    transaction_id: str
    date: date
    merchant: str
    category: str
    amount: Decimal
    transaction_type: TransactionType

    @field_validator("transaction_id", "merchant", "category")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        """Reject blank transaction identifiers and labels."""
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("amount")
    @classmethod
    def require_positive_amount(cls, value: Decimal) -> Decimal:
        """Amounts are stored as positive values; type determines their direction."""
        if not value.is_finite() or value <= Decimal("0"):
            raise ValueError("must be a finite value greater than zero")
        return value


class SpendingBreakdownItem(BaseModel):
    """A named monetary total used in spending breakdowns and comparisons."""

    name: str
    amount: Decimal


class SpendingSummary(BaseModel):
    """Debit spending totals and breakdowns for an inclusive date range."""

    start_date: date
    end_date: date
    total_spent: Decimal
    transaction_count: int
    average_transaction: Decimal
    by_category: list[SpendingBreakdownItem]
    by_merchant: list[SpendingBreakdownItem]


class PeriodComparison(BaseModel):
    """The difference in debit spending between two inclusive date ranges."""

    first_period: SpendingSummary
    second_period: SpendingSummary
    absolute_change: Decimal
    percentage_change: Decimal | None
    category_changes: list[SpendingBreakdownItem]
