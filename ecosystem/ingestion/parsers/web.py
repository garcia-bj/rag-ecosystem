"""Web parser — Firecrawl client with rate limiting and retry."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from urllib.parse import urlparse

from .base import BaseParser, ParsedChunk

logger = logging.getLogger(__name__)

try:
    from firecrawl import FirecrawlApp
    _FIRECRAWL_OK = True
except ImportError:
    _FIRECRAWL_OK = False

# Max concurrent Firecrawl requests
_MAX_CONCURRENCY = 3
# Delay between requests (seconds)
_REQUEST_DELAY = 1.0


class WebParser(BaseParser):
    """Scrape and parse web pages via Firecrawl.

    Supports single URL, sitemap crawl, and batch URL lists.
    Rate limiting: max _MAX_CONCURRENCY concurrent requests,
    _REQUEST_DELAY seconds between bursts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_depth: int = 1,
        page_options: dict[str, Any] | None = None,
    ) -> None:
        if not _FIRECRAWL_OK:
            raise ImportError("firecrawl-py is required: pip install firecrawl-py")
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self.max_depth = max_depth
        self.page_options = page_options or {
            "onlyMainContent": True,
            "includeHtml": False,
        }
        self._sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        self._app: "FirecrawlApp | None" = None

    def _get_app(self) -> "FirecrawlApp":
        if self._app is None:
            self._app = FirecrawlApp(api_key=self.api_key)
        return self._app

    async def parse(
        self,
        source: str | list[str],  # single URL or list of URLs
        source_id: str | None = None,
    ) -> list[ParsedChunk]:
        if isinstance(source, list):
            return await self._parse_batch(source)

        url = str(source)
        sid = source_id or url
        return await self._scrape_url(url, sid)

    async def crawl(self, base_url: str, source_id: str | None = None) -> list[ParsedChunk]:
        """Crawl a site up to max_depth, respecting robots.txt."""
        sid = source_id or base_url
        return await asyncio.to_thread(self._crawl_sync, base_url, sid)

    # ------------------------------------------------------------------ internals

    async def _scrape_url(self, url: str, source_id: str) -> list[ParsedChunk]:
        async with self._sem:
            result = await asyncio.to_thread(
                self._get_app().scrape_url,
                url,
                params={"pageOptions": self.page_options},
            )
            await asyncio.sleep(_REQUEST_DELAY)

        return self._result_to_chunks(result, url, source_id)

    async def _parse_batch(self, urls: list[str]) -> list[ParsedChunk]:
        tasks = [self._scrape_url(url, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        chunks: list[ParsedChunk] = []
        for url, res in zip(urls, results):
            if isinstance(res, Exception):
                logger.warning("Failed to scrape %s: %s", url, res)
                continue
            chunks.extend(res)
        return chunks

    def _crawl_sync(self, base_url: str, source_id: str) -> list[ParsedChunk]:
        app = self._get_app()
        result = app.crawl_url(
            base_url,
            params={
                "crawlerOptions": {
                    "maxDepth": self.max_depth,
                    "limit": 100,
                },
                "pageOptions": self.page_options,
            },
        )
        chunks: list[ParsedChunk] = []
        pages = result if isinstance(result, list) else result.get("data", [])
        for i, page in enumerate(pages):
            page_chunks = self._result_to_chunks(page, page.get("url", base_url), source_id)
            for c in page_chunks:
                c.chunk_index = i
            chunks.extend(page_chunks)
        return chunks

    @staticmethod
    def _result_to_chunks(
        result: dict[str, Any], url: str, source_id: str
    ) -> list[ParsedChunk]:
        content = (
            result.get("content")
            or result.get("markdown")
            or result.get("text")
            or ""
        ).strip()

        if not content:
            logger.debug("Empty content from %s", url)
            return []

        parsed = urlparse(url)
        meta: dict[str, Any] = {
            "source_id": source_id,
            "url": url,
            "domain": parsed.netloc,
            "title": result.get("metadata", {}).get("title", ""),
            "description": result.get("metadata", {}).get("description", ""),
            "og_image": result.get("metadata", {}).get("ogImage", ""),
            "word_count": len(content.split()),
        }

        return [
            ParsedChunk(
                text=content,
                metadata=meta,
                modality="web",
                source_id=source_id,
                chunk_index=0,
            )
        ]
