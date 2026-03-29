#!/usr/bin/env python3
"""Manual RAGAS evaluation script.

Loads a JSON dataset (default: tests/fixtures/eval_dataset.json),
evaluates with RAGAS, prints the score report, and optionally saves to
PostgreSQL and/or a JSON file.

Usage::

    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --dataset path/to/dataset.json
    python scripts/run_evaluation.py --tenant-id <uuid> --output reports/eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_evaluation(
    dataset_path: str,
    tenant_id: str = "",
    save_to_db: bool = False,
) -> dict:
    """Load dataset from JSON, run RAGAS evaluation, return report dict."""
    dataset_path = str(Path(dataset_path).resolve())
    with open(dataset_path) as f:
        dataset = json.load(f)

    if not dataset:
        return {"error": "Empty dataset"}

    queries       = [item["question"]                   for item in dataset]
    responses     = [item["answer"]                     for item in dataset]
    contexts      = [item.get("contexts", [])           for item in dataset]
    ground_truths = [item.get("ground_truth", "")       for item in dataset]

    # Only pass ground_truths if all items have them
    gt_arg = ground_truths if any(ground_truths) else None

    from evaluation.evaluator import evaluate_batch
    scores = await evaluate_batch(
        queries=queries,
        responses=responses,
        contexts=contexts,
        ground_truths=gt_arg,
        tenant_id=tenant_id,
        save_to_db=save_to_db,
    )

    return {
        "run_at":    datetime.now(timezone.utc).isoformat(),
        "dataset":   dataset_path,
        "n_samples": len(dataset),
        "scores":    scores,
    }


def _print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Report")
    print("=" * 60)
    print(f"  Dataset  : {report['dataset']}")
    print(f"  Samples  : {report['n_samples']}")
    print(f"  Run at   : {report['run_at']}")
    print("-" * 60)
    scores = report.get("scores", {})
    for metric, value in scores.items():
        if metric == "error":
            print(f"  {'ERROR':<25} {value}")
        else:
            bar = "█" * int(value * 20)
            print(f"  {metric:<25} {value:.4f}  {bar}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/eval_dataset.json",
        help="Path to eval dataset JSON",
    )
    parser.add_argument(
        "--tenant-id",
        default="",
        help="Tenant UUID (enables DB persistence when set)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save scores to PostgreSQL (requires POSTGRES_HOST)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Write JSON report to this file path",
    )
    args = parser.parse_args()

    report = asyncio.run(
        run_evaluation(args.dataset, args.tenant_id, args.save)
    )

    _print_report(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved -> {out_path}", file=sys.stderr)

    # Return non-zero if evaluation errored
    if "error" in report.get("scores", {}):
        sys.exit(1)


if __name__ == "__main__":
    main()
