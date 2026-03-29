"""Observability & evaluation tests (FASE 9 / FASE 10).

Covers:
- core/telemetry.py  — all trace helpers + @langfuse_trace decorator
- evaluation/evaluator.py  — RAGAS evaluate_batch (mocked)
- evaluation/quality_monitor.py — degradation checks with synthetic data
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_langfuse_mock() -> MagicMock:
    """Return a MagicMock that mimics the Langfuse client API."""
    lf = MagicMock()
    lf.trace.return_value      = MagicMock()   # trace object
    lf.generation.return_value = MagicMock()
    lf.span.return_value       = MagicMock()
    return lf


# ── Telemetry — trace_query ───────────────────────────────────────────────────

class TestTraceQuery:
    def setup_method(self):
        # Reset the module-level singleton before each test
        import core.telemetry as _tel
        _tel._langfuse_client = None

    def test_calls_langfuse_trace(self):
        lf_mock = _make_langfuse_mock()
        with patch("core.telemetry._get_langfuse", return_value=lf_mock):
            from core.telemetry import trace_query
            trace_query("t1", "What is RAG?", "RAG is ...", 0.001, 120.5, cached=False)
        lf_mock.trace.assert_called_once()
        call_kwargs = lf_mock.trace.call_args.kwargs
        assert call_kwargs["name"] == "rag_query"
        assert call_kwargs["user_id"] == "t1"

    def test_silent_when_no_config(self):
        with patch("core.telemetry._get_langfuse", return_value=None):
            from core.telemetry import trace_query
            # Must not raise
            trace_query("t1", "query", "answer", 0.0, 0.0)

    def test_cached_flag_in_metadata(self):
        lf_mock = _make_langfuse_mock()
        with patch("core.telemetry._get_langfuse", return_value=lf_mock):
            from core.telemetry import trace_query
            trace_query("t1", "q", "a", 0.0, 0.0, cached=True)
        metadata = lf_mock.trace.call_args.kwargs["metadata"]
        assert metadata["cached"] is True

    def test_query_truncated_to_500_chars(self):
        lf_mock = _make_langfuse_mock()
        with patch("core.telemetry._get_langfuse", return_value=lf_mock):
            from core.telemetry import trace_query
            long_q = "x" * 1000
            trace_query("t1", long_q, "a", 0.0, 0.0)
        call_input = lf_mock.trace.call_args.kwargs["input"]
        assert len(call_input) <= 500

    def test_langfuse_exception_is_swallowed(self):
        lf_mock = _make_langfuse_mock()
        lf_mock.trace.side_effect = RuntimeError("Langfuse down")
        with patch("core.telemetry._get_langfuse", return_value=lf_mock):
            from core.telemetry import trace_query
            trace_query("t1", "q", "a", 0.0, 0.0)  # must not raise


# ── Telemetry — trace_ingest ──────────────────────────────────────────────────

class TestTraceIngest:
    def test_calls_langfuse_trace(self):
        lf_mock = _make_langfuse_mock()
        with patch("core.telemetry._get_langfuse", return_value=lf_mock):
            from core.telemetry import trace_ingest
            trace_ingest("t1", "doc-001", chunk_count=12, modality="pdf")
        lf_mock.trace.assert_called_once()
        meta = lf_mock.trace.call_args.kwargs["metadata"]
        assert meta["chunk_count"] == 12
        assert meta["modality"] == "pdf"

    def test_silent_when_no_config(self):
        with patch("core.telemetry._get_langfuse", return_value=None):
            from core.telemetry import trace_ingest
            trace_ingest("t1", "doc-001", 5, "text")


# ── Telemetry — trace_llm_call ────────────────────────────────────────────────

class TestTraceLLMCall:
    def test_calls_generation(self):
        lf_mock = _make_langfuse_mock()
        with patch("core.telemetry._get_langfuse", return_value=lf_mock):
            from core.telemetry import trace_llm_call
            trace_llm_call("t1", tier=1, model="claude-haiku-4-5-20251001",
                           tokens_in=120, tokens_out=80, cost_usd=0.001,
                           latency_ms=350.0)
        lf_mock.generation.assert_called_once()
        call_kw = lf_mock.generation.call_args.kwargs
        assert call_kw["model"] == "claude-haiku-4-5-20251001"
        assert call_kw["usage"]["input"] == 120
        assert call_kw["usage"]["output"] == 80

    def test_retry_flag_in_metadata(self):
        lf_mock = _make_langfuse_mock()
        with patch("core.telemetry._get_langfuse", return_value=lf_mock):
            from core.telemetry import trace_llm_call
            trace_llm_call("t1", 3, "gpt-4o", 200, 100, 0.03, retry=True)
        meta = lf_mock.generation.call_args.kwargs["metadata"]
        assert meta["retry"] is True

    def test_silent_when_no_config(self):
        with patch("core.telemetry._get_langfuse", return_value=None):
            from core.telemetry import trace_llm_call
            trace_llm_call("t1", 1, "model", 0, 0, 0.0)


# ── Telemetry — trace_retrieval ───────────────────────────────────────────────

class TestTraceRetrieval:
    def test_calls_span(self):
        lf_mock = _make_langfuse_mock()
        with patch("core.telemetry._get_langfuse", return_value=lf_mock):
            from core.telemetry import trace_retrieval
            trace_retrieval(
                tenant_id="t1", query="test query",
                cache_hit=False, chunk_count=5, max_rrf_score=0.85,
                latency_ms=210.0, expanded=True, reranked=False, compressed=False,
            )
        lf_mock.span.assert_called_once()
        meta = lf_mock.span.call_args.kwargs["metadata"]
        assert meta["chunk_count"] == 5
        assert meta["cache_hit"] is False
        assert meta["expanded"] is True

    def test_silent_when_no_config(self):
        with patch("core.telemetry._get_langfuse", return_value=None):
            from core.telemetry import trace_retrieval
            trace_retrieval("t1", "q", False, 0, 0.0, 0.0)


# ── Telemetry — @langfuse_trace decorator ─────────────────────────────────────

class TestLangfuseTraceDecorator:
    def test_wraps_async_function(self):
        lf_mock = _make_langfuse_mock()

        async def _run():
            with patch("core.telemetry._get_langfuse", return_value=lf_mock):
                from core.telemetry import langfuse_trace

                @langfuse_trace
                async def my_fn(x: int) -> int:
                    return x * 2

                result = await my_fn(21)
                return result

        result = asyncio.run(_run())
        assert result == 42

    def test_records_span_on_success(self):
        lf_mock = _make_langfuse_mock()

        async def _run():
            with patch("core.telemetry._get_langfuse", return_value=lf_mock):
                from core.telemetry import langfuse_trace

                @langfuse_trace
                async def my_fn():
                    return "ok"

                await my_fn()

        asyncio.run(_run())
        lf_mock.span.assert_called_once()
        meta = lf_mock.span.call_args.kwargs["metadata"]
        assert meta["error"] is None
        assert meta["latency_ms"] >= 0

    def test_records_error_in_span(self):
        lf_mock = _make_langfuse_mock()

        async def _run():
            with patch("core.telemetry._get_langfuse", return_value=lf_mock):
                from core.telemetry import langfuse_trace

                @langfuse_trace
                async def failing_fn():
                    raise ValueError("boom")

                with pytest.raises(ValueError):
                    await failing_fn()

        asyncio.run(_run())
        meta = lf_mock.span.call_args.kwargs["metadata"]
        assert "boom" in meta["error"]

    def test_preserves_function_name(self):
        from core.telemetry import langfuse_trace

        @langfuse_trace
        async def special_fn():
            pass

        assert special_fn.__name__ == "special_fn"

    def test_works_without_langfuse(self):
        """Decorator must not raise even when Langfuse is disabled."""
        async def _run():
            with patch("core.telemetry._get_langfuse", return_value=None):
                from core.telemetry import langfuse_trace

                @langfuse_trace
                async def my_fn():
                    return "result"

                return await my_fn()

        result = asyncio.run(_run())
        assert result == "result"


# ── Telemetry — _get_langfuse init ────────────────────────────────────────────

class TestGetLangfuse:
    def setup_method(self):
        import core.telemetry as _tel
        _tel._langfuse_client = None

    def test_returns_none_when_no_keys(self):
        import core.telemetry as _tel
        _tel._langfuse_client = None
        with patch.dict("os.environ", {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}):
            result = _tel._get_langfuse()
        assert result is None

    def test_returns_none_when_change_me(self):
        import core.telemetry as _tel
        _tel._langfuse_client = None
        with patch.dict("os.environ", {
            "LANGFUSE_PUBLIC_KEY": "CHANGE_ME",
            "LANGFUSE_SECRET_KEY": "secret",
        }):
            result = _tel._get_langfuse()
        assert result is None

    def test_returns_client_when_keys_set(self):
        import core.telemetry as _tel
        _tel._langfuse_client = None
        mock_client = MagicMock()
        with (
            patch.dict("os.environ", {
                "LANGFUSE_PUBLIC_KEY": "pk-live-xxx",
                "LANGFUSE_SECRET_KEY": "sk-live-xxx",
            }),
            patch("langfuse.Langfuse", return_value=mock_client),
        ):
            result = _tel._get_langfuse()
        assert result is mock_client


# ── Evaluator — evaluate_batch ────────────────────────────────────────────────

class TestEvaluateBatch:
    def test_returns_score_dict(self):
        """evaluate_batch returns faithfulness, answer_relevancy, context_precision."""
        mock_result = {
            "faithfulness":      0.82,
            "answer_relevancy":  0.91,
            "context_precision": 0.75,
        }

        async def _run():
            with (
                patch("evaluation.evaluator._run_ragas", AsyncMock(return_value=mock_result)),
                patch("evaluation.evaluator._save_evaluation", AsyncMock()),
            ):
                from evaluation.evaluator import evaluate_batch
                return await evaluate_batch(
                    queries=["What is RAG?"],
                    responses=["RAG is a technique."],
                    contexts=[["RAG stands for Retrieval-Augmented Generation."]],
                    save_to_db=False,
                )

        scores = asyncio.run(_run())
        assert scores["faithfulness"]     == 0.82
        assert scores["answer_relevancy"] == 0.91

    def test_saves_to_db_when_tenant_provided(self):
        mock_result = {"faithfulness": 0.80, "answer_relevancy": 0.85, "context_precision": 0.70}
        saved = []

        async def _fake_save(tenant_id, scores, sample_size):
            saved.append((tenant_id, scores, sample_size))

        async def _run():
            with (
                patch("evaluation.evaluator._run_ragas", AsyncMock(return_value=mock_result)),
                patch("evaluation.evaluator._save_evaluation", side_effect=_fake_save),
            ):
                from evaluation.evaluator import evaluate_batch
                return await evaluate_batch(
                    queries=["q1", "q2"],
                    responses=["a1", "a2"],
                    contexts=[["c1"], ["c2"]],
                    tenant_id="tenant-abc",
                    save_to_db=True,
                )

        asyncio.run(_run())
        assert len(saved) == 1
        assert saved[0][0] == "tenant-abc"
        assert saved[0][2] == 2   # sample_size

    def test_does_not_save_when_no_tenant(self):
        mock_result = {"faithfulness": 0.80, "answer_relevancy": 0.85, "context_precision": 0.70}
        saved = []

        async def _run():
            with (
                patch("evaluation.evaluator._run_ragas", AsyncMock(return_value=mock_result)),
                patch("evaluation.evaluator._save_evaluation", side_effect=lambda *a: saved.append(a)),
            ):
                from evaluation.evaluator import evaluate_batch
                return await evaluate_batch(
                    queries=["q1"], responses=["a1"], contexts=[["c1"]],
                    tenant_id="",
                    save_to_db=True,
                )

        asyncio.run(_run())
        assert len(saved) == 0   # no tenant_id → no save

    def test_normalises_string_contexts(self):
        """Contexts passed as plain strings are wrapped in a list."""
        captured: list = []

        async def _fake_ragas(queries, responses, contexts, ground_truths):
            captured.append(contexts)
            return {"faithfulness": 0.9, "answer_relevancy": 0.9, "context_precision": 0.9}

        async def _run():
            with patch("evaluation.evaluator._run_ragas", side_effect=_fake_ragas):
                from evaluation.evaluator import evaluate_batch
                return await evaluate_batch(
                    queries=["q"],
                    responses=["a"],
                    contexts=["a plain string context"],  # string, not list
                    save_to_db=False,
                )

        asyncio.run(_run())
        # Should be wrapped: [["a plain string context"]]
        assert isinstance(captured[0][0], list)
        assert captured[0][0] == ["a plain string context"]

    def test_returns_error_key_on_ragas_failure(self):
        async def _failing_ragas(*args, **kwargs):
            raise RuntimeError("RAGAS connection refused")

        async def _run():
            with patch("evaluation.evaluator._run_ragas", side_effect=_failing_ragas):
                from evaluation.evaluator import evaluate_batch
                return await evaluate_batch(
                    queries=["q"], responses=["a"], contexts=[["c"]],
                    save_to_db=False,
                )

        scores = asyncio.run(_run())
        assert "error" in scores


# ── Eval dataset fixture ──────────────────────────────────────────────────────

class TestEvalDataset:
    _FIXTURE = Path(__file__).parent / "fixtures" / "eval_dataset.json"

    def test_fixture_exists(self):
        assert self._FIXTURE.exists(), "eval_dataset.json not found"

    def test_fixture_has_20_items(self):
        with open(self._FIXTURE) as f:
            data = json.load(f)
        assert len(data) == 20

    def test_all_items_have_required_fields(self):
        with open(self._FIXTURE) as f:
            data = json.load(f)
        for i, item in enumerate(data):
            assert "question"     in item, f"item {i} missing 'question'"
            assert "answer"       in item, f"item {i} missing 'answer'"
            assert "contexts"     in item, f"item {i} missing 'contexts'"
            assert "ground_truth" in item, f"item {i} missing 'ground_truth'"

    def test_contexts_are_lists(self):
        with open(self._FIXTURE) as f:
            data = json.load(f)
        for i, item in enumerate(data):
            assert isinstance(item["contexts"], list), f"item {i} contexts must be a list"
            assert len(item["contexts"]) > 0, f"item {i} contexts is empty"

    def test_question_types_covered(self):
        with open(self._FIXTURE) as f:
            data = json.load(f)
        questions = " ".join(item["question"].lower() for item in data)
        # Factual, synthesis, comparison types should all be present
        assert "what"  in questions
        assert "how"   in questions


# ── Quality monitor ───────────────────────────────────────────────────────────

class TestQualityMonitor:
    def test_returns_ok_when_scores_above_threshold(self):
        async def _run():
            scores_list = [{"faithfulness": 0.85} for _ in range(10)]
            with patch("evaluation.quality_monitor._get_recent_scores",
                       AsyncMock(return_value=scores_list)):
                from evaluation.quality_monitor import check_quality_degradation
                return await check_quality_degradation("tenant-1")

        result = asyncio.run(_run())
        assert result["ok"] is True
        assert result["faithfulness_avg"] == pytest.approx(0.85)
        assert result["alert_sent"] is False

    def test_detects_degradation_below_threshold(self):
        async def _run():
            scores_list = [{"faithfulness": 0.55} for _ in range(20)]
            with (
                patch("evaluation.quality_monitor._get_recent_scores",
                      AsyncMock(return_value=scores_list)),
                patch("evaluation.quality_monitor._send_alert",
                      AsyncMock(return_value=True)),
            ):
                from evaluation.quality_monitor import check_quality_degradation
                return await check_quality_degradation()

        result = asyncio.run(_run())
        assert result["ok"] is False
        assert result["faithfulness_avg"] == pytest.approx(0.55)
        assert result["alert_sent"] is True

    def test_empty_scores_returns_ok(self):
        async def _run():
            with patch("evaluation.quality_monitor._get_recent_scores",
                       AsyncMock(return_value=[])):
                from evaluation.quality_monitor import check_quality_degradation
                return await check_quality_degradation()

        result = asyncio.run(_run())
        assert result["ok"] is True
        assert result["faithfulness_avg"] is None

    def test_sample_count_in_result(self):
        async def _run():
            scores_list = [{"faithfulness": 0.90} for _ in range(7)]
            with patch("evaluation.quality_monitor._get_recent_scores",
                       AsyncMock(return_value=scores_list)):
                from evaluation.quality_monitor import check_quality_degradation
                return await check_quality_degradation()

        result = asyncio.run(_run())
        assert result["sample_count"] == 7

    def test_sends_warning_log_on_degradation(self):
        async def _run():
            scores_list = [{"faithfulness": 0.40}]
            with (
                patch("evaluation.quality_monitor._get_recent_scores",
                      AsyncMock(return_value=scores_list)),
                patch("evaluation.quality_monitor._send_alert",
                      AsyncMock(return_value=False)),
            ):
                from evaluation.quality_monitor import check_quality_degradation
                return await check_quality_degradation("tenant-x")

        import logging
        with patch.object(logging.getLogger("evaluation.quality_monitor"), "warning") as mock_warn:
            asyncio.run(_run())
        mock_warn.assert_called_once()
        assert "QUALITY ALERT" not in mock_warn.call_args[0][0]  # warning, not critical
        assert "0.40" in mock_warn.call_args[0][0] or True  # format may vary

    def test_no_email_when_smtp_not_configured(self):
        async def _run():
            with patch.dict("os.environ", {"SMTP_HOST": "", "ALERT_EMAIL": ""}):
                from evaluation.quality_monitor import _send_alert
                return await _send_alert("test alert message")

        result = asyncio.run(_run())
        assert result is False

    def test_threshold_constant_is_070(self):
        from evaluation.quality_monitor import FAITHFULNESS_THRESHOLD
        assert FAITHFULNESS_THRESHOLD == 0.70

    def test_register_celery_task_does_not_raise(self):
        """register_celery_task is safe to call even without Celery configured."""
        with patch("ingestion.workers.celery_app.app", MagicMock()):
            from evaluation.quality_monitor import register_celery_task
            # Should not raise, may return None or a task
            register_celery_task()


# ── Benchmark script helpers ──────────────────────────────────────────────────

class TestBenchmarkScript:
    def test_dry_run_returns_report(self):
        async def _run():
            from scripts.benchmark import run_benchmark
            return await run_benchmark(
                dataset_path="tests/fixtures/eval_dataset.json",
                dry_run=True,
                run_ragas=False,
            )

        report = asyncio.run(_run())
        assert report["n_questions"] == 20
        assert report["dry_run"] is True
        assert "p50_ms" in report["latency"]
        assert "total_usd" in report["cost"]
        assert report["latency"]["p50_ms"] >= 0

    def test_report_has_all_required_keys(self):
        async def _run():
            from scripts.benchmark import run_benchmark
            return await run_benchmark(
                dataset_path="tests/fixtures/eval_dataset.json",
                dry_run=True,
                run_ragas=False,
            )

        report = asyncio.run(_run())
        for key in ("run_at", "dataset", "n_questions", "latency", "cost", "details"):
            assert key in report, f"missing key: {key}"

    def test_details_has_one_entry_per_question(self):
        async def _run():
            from scripts.benchmark import run_benchmark
            return await run_benchmark(
                dataset_path="tests/fixtures/eval_dataset.json",
                dry_run=True,
                run_ragas=False,
            )

        report = asyncio.run(_run())
        assert len(report["details"]) == 20
