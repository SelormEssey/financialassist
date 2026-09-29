"""Run safe offline evaluations, or explicitly requested live evaluation modes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evals.runner import (
    REPOSITORY_ROOT,
    run_deterministic_evaluations,
    run_live_agent_evaluations,
    run_live_retrieval_evaluations,
)


def format_metric_summary(summary) -> str:
    """Format evaluated percentages, evaluated counts, and unavailable metrics safely."""
    if summary.availability == "unavailable":
        return f"{summary.name}: NOT EVALUATED"
    if summary.percentage is None:
        return f"{summary.name}: {summary.numerator} across {summary.denominator} eligible cases"
    if summary.numerator is not None:
        return f"{summary.name}: {summary.numerator}/{summary.denominator} ({summary.percentage:.1f}%)"
    return f"{summary.name}: {summary.percentage:.1f}% ({summary.aggregation})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run financialassist evaluations.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live-agent", action="store_true", help="Run live OpenAI agent evaluation.")
    modes.add_argument("--live-retrieval", action="store_true", help="Run live OpenAI retrieval evaluation.")
    parser.add_argument("--json-output", type=Path, help="Write the calculated report as JSON.")
    args = parser.parse_args()

    if args.live_agent:
        report = run_live_agent_evaluations()
    elif args.live_retrieval:
        report = run_live_retrieval_evaluations()
    else:
        report = run_deterministic_evaluations()
    print(f"financialassist evaluation ({report.mode})")
    print(
        "Cases: "
        f"dataset={report.dataset_total}, executed={report.executed_cases}, "
        f"passed={report.passed_cases}, failed={report.failed_cases}, skipped={report.skipped_cases}"
    )
    for summary in report.metric_summaries:
        print(format_metric_summary(summary))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
