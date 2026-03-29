"""Query router — POST /query, POST /query/stream, GET /query/history."""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.middleware.auth import get_current_user
from api.middleware.rate_limit import get_rate_limiter
from api.schemas import ChunkOut, QueryRequest, QueryResponse, QueryHistoryItem
from core.tenant import TenantContext

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    ctx: TenantContext = Depends(get_current_user),
):
    """Answer a query using hybrid retrieval + LLM routing."""
    from api.middleware.rate_limit import get_rate_limiter
    from core.tenant import get_tenant_collection, get_tenant_index
    from retrieval.pipeline import RetrievalPipeline
    from llm.router import LLMRouter
    from llm.hallucination_guard import HallucinationGuard

    # Rate limit check (fetches plan from DB or defaults to free)
    plan = await _get_tenant_plan(ctx.tenant_id)
    await get_rate_limiter().check_query(ctx.tenant_id, plan)

    t0 = time.perf_counter()

    # Retrieval
    pipeline = RetrievalPipeline(
        qdrant_collection=get_tenant_collection(ctx.tenant_id),
        es_index=get_tenant_index(ctx.tenant_id),
    )
    try:
        ret_result = await pipeline.run(req.query, top_k=req.top_k)
    finally:
        await pipeline.close()

    if ret_result.cache_hit:
        # Reconstruct answer from cached chunks (avoid LLM cost)
        answer = _cached_answer(ret_result.chunks)
        sources = _chunks_to_out(ret_result.chunks)
        return QueryResponse(
            answer=answer,
            sources=sources,
            cost_usd=0.0,
            cached=True,
            model="cache",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # LLM routing
    router_llm = LLMRouter()
    llm_resp = await router_llm.complete(
        query=req.query,
        chunks=ret_result.chunks,
        tenant_name="",  # can be enriched from DB
        tenant_id=ctx.tenant_id,
    )

    # Hallucination guard
    guard = HallucinationGuard()

    async def _regenerate(prompt: str) -> str:
        import litellm
        r = await litellm.acompletion(
            model=llm_resp.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return r.choices[0].message.content or ""

    from llm.prompt_builder import PromptBuilder
    prompt, _ = PromptBuilder().build(req.query, ret_result.chunks, llm_resp.tier)
    guard_result = await guard.check(
        response=llm_resp.text,
        chunks=ret_result.chunks,
        router_fn=_regenerate,
        query=req.query,
        prompt=prompt,
    )

    final_answer = guard_result.final_response or llm_resp.text
    total_latency = round((time.perf_counter() - t0) * 1000, 1)

    # Persist query to history (fire-and-forget)
    _record_history(ctx, req.query, final_answer, llm_resp.cost_usd)

    # Telemetry trace (non-blocking, fails silently)
    try:
        from core.telemetry import trace_query
        trace_query(
            tenant_id=ctx.tenant_id,
            query=req.query,
            response=final_answer,
            cost_usd=llm_resp.cost_usd,
            latency_ms=total_latency,
            cached=False,
        )
    except Exception:
        pass

    return QueryResponse(
        answer=final_answer,
        sources=_chunks_to_out(ret_result.chunks),
        cost_usd=llm_resp.cost_usd,
        cached=False,
        model=llm_resp.model,
        latency_ms=total_latency,
    )


@router.post("/stream")
async def query_stream(
    req: QueryRequest,
    ctx: TenantContext = Depends(get_current_user),
):
    """Stream LLM answer via SSE."""
    plan = await _get_tenant_plan(ctx.tenant_id)
    await get_rate_limiter().check_query(ctx.tenant_id, plan)

    async def _generate() -> AsyncGenerator[str, None]:
        import litellm
        from core.tenant import get_tenant_collection, get_tenant_index
        from retrieval.pipeline import RetrievalPipeline
        from llm.router import LLMRouter

        pipeline = RetrievalPipeline(
            qdrant_collection=get_tenant_collection(ctx.tenant_id),
            es_index=get_tenant_index(ctx.tenant_id),
            use_expansion=False,
        )
        try:
            ret_result = await pipeline.run(req.query, top_k=req.top_k)
        finally:
            await pipeline.close()

        router_llm = LLMRouter()
        tier = router_llm.classify_tier(req.query)
        from llm.router import _TIER_MODELS
        model = _TIER_MODELS.get(tier, _TIER_MODELS[1])

        from llm.prompt_builder import PromptBuilder
        prompt, _ = PromptBuilder().build(req.query, ret_result.chunks, tier)

        # Send sources first
        sources = [
            {"source_id": c.source_id, "score": c.rrf_score}
            for c in ret_result.chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        # Stream tokens
        stream = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps({'type': 'token', 'data': delta})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/history", response_model=list[QueryHistoryItem])
async def query_history(
    ctx: TenantContext = Depends(get_current_user),
    limit: int = 50,
):
    """Return the last N queries for this tenant."""
    try:
        from core.database import get_connection
        conn = await get_connection()
        try:
            rows = await conn.fetch(
                """
                SELECT query, answer, cost_usd, cached, created_at
                FROM query_history
                WHERE tenant_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                ctx.tenant_id, limit,
            )
            return [
                QueryHistoryItem(
                    query=r["query"],
                    answer=r["answer"],
                    cost_usd=float(r["cost_usd"]),
                    cached=r["cached"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("query_history DB error: %s", exc)
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunks_to_out(chunks) -> list[ChunkOut]:
    return [
        ChunkOut(
            text=c.text,
            source_id=c.source_id,
            chunk_index=c.chunk_index,
            modality=c.modality,
            score=round(c.rrf_score, 6),
            sources=c.sources,
        )
        for c in chunks
    ]

def _cached_answer(chunks) -> str:
    if not chunks:
        return "No relevant information found."
    return chunks[0].text[:500]

def _record_history(ctx: TenantContext, query: str, answer: str, cost: float) -> None:
    """Fire-and-forget history persistence."""
    import asyncio
    async def _persist():
        try:
            from core.database import get_connection
            conn = await get_connection()
            try:
                await conn.execute(
                    """
                    INSERT INTO query_history (tenant_id, user_id, query, answer, cost_usd)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    ctx.tenant_id, ctx.user_id, query, answer[:2000], cost,
                )
            finally:
                await conn.close()
        except Exception as exc:
            logger.debug("History persist failed: %s", exc)

    asyncio.create_task(_persist())

async def _get_tenant_plan(tenant_id: str) -> str:
    try:
        from core.database import get_tenant_by_id
        tenant = await get_tenant_by_id(tenant_id)
        return tenant.plan if tenant else "free"
    except Exception:
        return "free"
