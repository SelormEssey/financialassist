"""Golden JSONL dataset loading with strict, ordered validation."""

import json
from pathlib import Path

from pydantic import ValidationError

from app.evals.models import EvalCase


class GoldenDatasetError(ValueError):
    """Raised when a golden dataset cannot be loaded exactly as authored."""


def load_golden_dataset(path: str | Path) -> list[EvalCase]:
    """Load ordered JSONL cases; malformed, duplicate, and invalid rows fail loudly."""
    dataset_path = Path(path)
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GoldenDatasetError(f"Unable to read golden dataset: {dataset_path}") from error

    cases: list[EvalCase] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise GoldenDatasetError(f"Blank line in golden dataset at line {line_number}")
        try:
            raw_case = json.loads(line)
        except json.JSONDecodeError as error:
            raise GoldenDatasetError(f"Malformed JSON at line {line_number}") from error
        try:
            case = EvalCase.model_validate(raw_case)
        except ValidationError as error:
            raise GoldenDatasetError(f"Invalid evaluation case at line {line_number}: {error}") from error
        if case.case_id in case_ids:
            raise GoldenDatasetError(f"Duplicate case_id '{case.case_id}' at line {line_number}")
        case_ids.add(case.case_id)
        cases.append(case)
    return cases
