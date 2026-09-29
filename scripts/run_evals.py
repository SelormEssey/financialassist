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
        if summary.availability == "unavailable":
            print(f"{summary.name}: NOT EVALUATED")
        elif summary.numerator is not None:
            print(f"{summary.name}: {summary.numerator}/{summary.denominator} ({summary.percentage:.1f}%)")
        else:
            print(f"{summary.name}: {summary.percentage:.1f}% ({summary.aggregation})")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
