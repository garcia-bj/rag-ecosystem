#!/usr/bin/env python3
"""RAG system benchmark — latency P50/P95, cost, RAGAS scores.

Runs the 20 eval-dataset questions against the live system (or mocked
pipeline for CI), measures timing, and writes a benchmark report.
If a previous report exists it compares the two.

Usage::

    # Against real system (needs POSTGRES_HOST, QDRANT_URL, etc.)
    python scripts/benchmark.py

    # Dry-run with mocked pipeline (no external services needed)
    python scripts/benchmark.py --dry-run

    # Custom dataset / output path
    python scripts/benchmark.py \\
        --dataset tests/fixtures/eval_dataset.json \\
        --output  reports/benchmark_latest.json \\
        --tenant-id <uuid> \\
        --token <jwt>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

REPORTS_DIR = Path(__file__).parent.parent / "reports"


# ── Query runner ──────────────────────────────────────────────────────────────

async def _run_single_query(
    question: str,
    tenant_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Run one question through the retrieval + LLM pipeline.

    In dry-run mode the pipeline is mocked so no external services are needed.
    """
    t0 = time.perf_counter()

    if dry_run:
        # Simulate a realistic answer for benchmarking purposes
        answer   = f"[MOCK] Answer to: {question[:60]}"
        chunks   = ["Mock context chunk 1.", "Mock context chunk 2."]
        cost_usd = 0.0
    else:
        from core.tenant import get_tenant_collection, get_tenant_index
        from retrieval.pipeline import RetrievalPipeline
        from llm.router import LLMRouter

        pipeline = RetrievalPipeline(
            qdrant_collection=get_tenant_collection(tenant_id),
            es_index=get_tenant_index(tenant_id),
        )
        try:
            ret_result = await pipeline.run(question, top_k=5)
        finally:
            await pipeline.close()

        if ret_result.cache_hit:
            answer   = ret_result.chunks[0].text[:500] if ret_result.chunks else ""
            chunks   = [c.text for c in ret_result.chunks]
            cost_usd = 0.0
        else:
            router = LLMRouter()
            llm_resp = await router.complete(
                query=question,
                chunks=ret_result.chunks,
                tenant_id=tenant_id,
            )
            answer   = llm_resp.text
            chunks   = [c.text for c in ret_result.chunks]
            cost_usd = llm_resp.cost_usd

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "question":   question,
        "answer":     answer,
        "contexts":   chunks if dry_run else chunks,
        "cost_usd":   cost_usd,
        "latency_ms": round(latency_ms, 1),
    }


# ── Main benchmark ────────────────────────────────────────────────────────────

async def run_benchmark(
    dataset_path: str,
    tenant_id: str = "",
    dry_run: bool = False,
    run_ragas: bool = True,
) -> dict:
    """Execute benchmark and return full report dict."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"Running benchmark on {len(dataset)} questions "
          f"({'dry-run' if dry_run else 'live'}) …")

    results: list[dict] = []
    for i, item in enumerate(dataset, 1):
        print(f"  [{i:02d}/{len(dataset)}] {item['question'][:55]} …", end="", flush=True)
        res = await _run_single_query(item["question"], tenant_id, dry_run)
        results.append(res)
        print(f" {res['latency_ms']:.0f}ms")

    latencies  = [r["latency_ms"] for r in results]
    total_cost = sum(r["cost_usd"] for r in results)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    ragas_scores: dict = {}
    if run_ragas and not dry_run:
        print("\nRunning RAGAS evaluation …")
        from evaluation.evaluator import evaluate_batch
        ground_truths = [item.get("ground_truth", "") for item in dataset]
        ragas_scores  = await evaluate_batch(
            queries      =[r["question"] for r in results],
            responses    =[r["answer"]   for r in results],
            contexts     =[r.get("contexts", []) for r in results],
            ground_truths=ground_truths if any(ground_truths) else None,
            tenant_id    =tenant_id,
            save_to_db   =bool(tenant_id),
        )
    elif dry_run:
        # Placeholder for dry-run mode
        ragas_scores = {
            "faithfulness":      0.0,
            "answer_relevancy":  0.0,
            "context_precision": 0.0,
            "note":              "dry-run — RAGAS skipped",
        }

    report = {
        "run_at":      datetime.now(timezone.utc).isoformat(),
        "dataset":     dataset_path,
        "n_questions": len(dataset),
        "dry_run":     dry_run,
        "latency": {
            "p50_ms":  round(p50, 1),
            "p95_ms":  round(p95, 1),
            "min_ms":  round(min(latencies), 1),
            "max_ms":  round(max(latencies), 1),
        },
        "cost": {
            "total_usd":    round(total_cost, 6),
            "avg_per_query": round(total_cost / len(results), 6) if results else 0,
        },
        "ragas":   ragas_scores,
        "details": results,
    }
    return report


def _compare_reports(current: dict, previous: dict) -> None:
    """Print a diff table between the current and previous benchmark."""
    print("\n── Comparison with previous benchmark ──────────────────")
    prev_run = previous.get("run_at", "unknown")[:19]
    print(f"   Previous run: {prev_run}")

    def _delta(key_path: str, fmt: str = ".1f") -> str:
        keys = key_path.split(".")
        try:
            cur  = current
            prev = previous
            for k in keys:
                cur  = cur[k]
                prev = prev[k]
            diff = float(cur) - float(prev)
            sign = "+" if diff >= 0 else ""
            return f"{cur:{fmt}} ({sign}{diff:{fmt}})"
        except Exception:
            return "N/A"

    print(f"   P50 latency   : {_delta('latency.p50_ms')} ms")
    print(f"   P95 latency   : {_delta('latency.p95_ms')} ms")
    print(f"   Total cost    : ${_delta('cost.total_usd', '.4f')}")
    for metric in ("faithfulness", "answer_relevancy", "context_precision"):
        val = _delta(f"ragas.{metric}", ".4f")
        print(f"   {metric:<22}: {val}")
    print()


def _print_report(report: dict) -> None:
    print("\n" + "=" * 65)
    print("  RAG System Benchmark Report")
    print("=" * 65)
    print(f"  Run at   : {report['run_at'][:19]}")
    print(f"  Dataset  : {report['dataset']}")
    print(f"  Samples  : {report['n_questions']}  (dry-run={report['dry_run']})")
    print("-" * 65)
    lat = report["latency"]
    print(f"  Latency  P50={lat['p50_ms']:.0f}ms  P95={lat['p95_ms']:.0f}ms  "
          f"min={lat['min_ms']:.0f}ms  max={lat['max_ms']:.0f}ms")
    cost = report["cost"]
    print(f"  Cost     total=${cost['total_usd']:.4f}  avg/q=${cost['avg_per_query']:.4f}")
    print("-" * 65)
    ragas = report.get("ragas", {})
    if ragas:
        for metric, value in ragas.items():
            if isinstance(value, float):
                bar = "█" * int(value * 20)
                print(f"  {metric:<25} {value:.4f}  {bar}")
            elif metric != "note":
                print(f"  {metric:<25} {value}")
    if "note" in ragas:
        print(f"  Note: {ragas['note']}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG System Benchmark")
    parser.add_argument("--dataset",   default="tests/fixtures/eval_dataset.json")
    parser.add_argument("--tenant-id", default="")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Mock pipeline — no external services needed")
    parser.add_argument("--no-ragas",  action="store_true",
                        help="Skip RAGAS evaluation (faster)")
    parser.add_argument("--output",    default="",
                        help="JSON output path (default: reports/benchmark_YYYY-MM-DD.json)")
    args = parser.parse_args()

    report = asyncio.run(run_benchmark(
        dataset_path=args.dataset,
        tenant_id=args.tenant_id,
        dry_run=args.dry_run,
        run_ragas=not args.no_ragas,
    ))

    _print_report(report)

    # Determine output path
    output_path = args.output or str(
        REPORTS_DIR / f"benchmark_{date.today().isoformat()}.json"
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compare with most recent previous report
    existing = sorted(REPORTS_DIR.glob("benchmark_*.json"))
    if existing and not args.output:
        try:
            with open(existing[-1]) as f:
                previous = json.load(f)
            if previous.get("run_at") != report["run_at"]:
                _compare_reports(report, previous)
        except Exception:
            pass

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved -> {output_path}")


if __name__ == "__main__":
    main()
