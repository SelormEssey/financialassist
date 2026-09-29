"""Typed models for transparent financialassist evaluation runs."""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EvalCategory = Literal["transaction", "retrieval", "combined", "unsupported"]
TransactionOperation = Literal["summary", "comparison"]
CaseStatus = Literal["passed", "failed", "skipped"]
MetricAvailability = Literal["evaluated", "unavailable"]


class TransactionEvaluation(BaseModel):
    """The deterministic operation a numeric golden case asks the evaluator to run."""

    model_config = ConfigDict(extra="forbid")
    operation: TransactionOperation
    first_start: date
    first_end: date
    second_start: date | None = None
    second_end: date | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "TransactionEvaluation":
        if self.first_end < self.first_start:
            raise ValueError("first_end must be on or after first_start")
        if self.operation == "comparison" and (
            self.second_start is None or self.second_end is None
        ):
            raise ValueError("comparison requires second_start and second_end")
        if self.second_start and self.second_end and self.second_end < self.second_start:
            raise ValueError("second_end must be on or after second_start")
        return self


class EvalCase(BaseModel):
    """One hand-authored golden expectation derived from synthetic source data."""

    model_config = ConfigDict(extra="forbid")
    case_id: str
    category: EvalCategory
    question: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    expected_top_citations: list[str] = Field(default_factory=list)
    expected_numeric: dict[str, Decimal] = Field(default_factory=dict)
    transaction_evaluation: TransactionEvaluation | None = None
    expect_no_supported_policy_answer: bool = False
    expect_insufficient_data: bool = False
    notes: str | None = None

    @field_validator("case_id", "question")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_category_requirements(self) -> "EvalCase":
        transaction_tools = {"summarize_transactions", "compare_transaction_periods"}
        if self.category == "transaction":
            if self.transaction_evaluation is None or not self.expected_numeric:
                raise ValueError("transaction cases require transaction_evaluation and expected_numeric")
            expected_tool = (
                "summarize_transactions"
                if self.transaction_evaluation.operation == "summary"
                else "compare_transaction_periods"
            )
            if self.expected_tools != [expected_tool]:
                raise ValueError(f"{self.transaction_evaluation.operation} cases require [{expected_tool!r}]")
        elif self.category == "retrieval":
            if not self.expected_citations:
                raise ValueError("retrieval cases require expected_citations")
            if self.expected_tools != ["search_financial_documents"]:
                raise ValueError("retrieval cases require ['search_financial_documents']")
        elif self.category == "combined":
            if self.expected_tools not in (
                ["summarize_transactions", "search_financial_documents"],
                ["compare_transaction_periods", "search_financial_documents"],
            ):
                raise ValueError("combined cases require one transaction tool followed by retrieval")
            if not self.expected_citations:
                raise ValueError("combined cases require expected_citations")
        elif self.category == "unsupported" and not (
            self.expect_no_supported_policy_answer or self.expect_insufficient_data
        ):
            raise ValueError("unsupported cases require an explicit unsupported-behavior expectation")
        return self


class EvalMetricSummary(BaseModel):
    """A metric result that distinguishes unavailable from measured values."""

    name: str
    availability: MetricAvailability
    numerator: int | None = None
    denominator: int | None = None
    percentage: float | None = None
    aggregation: str | None = None


class EvalCaseResult(BaseModel):
    """Inspectable observations and status for one dataset case."""

    case_id: str
    status: CaseStatus
    metrics: dict[str, bool | float | None] = Field(default_factory=dict)
    observed_tools: list[str] = Field(default_factory=list)
    observed_citations: list[str] = Field(default_factory=list)
    retrieved_citations: list[str] | None = None
    failure_reasons: list[str] = Field(default_factory=list)
    skip_reason: str | None = None


class EvalRunReport(BaseModel):
    """Serializable report for deterministic or explicitly requested live evaluation."""

    timestamp: datetime
    mode: str
    dataset_total: int
    executed_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    metric_summaries: list[EvalMetricSummary]
    results: list[EvalCaseResult]
