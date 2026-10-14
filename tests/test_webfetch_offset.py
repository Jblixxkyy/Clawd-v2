from __future__ import annotations

import io
import socket
import tempfile
import unittest
import urllib.request
import uuid
from pathlib import Path
from unittest.mock import patch

from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from src.tool_system.tools import WebFetchTool
from src.tool_system.tools import web_fetch as web_fetch_module

PAGE_CHARS = 250_000
PUBLIC_ADDR = [(None, None, None, None, ("93.184.216.34", 0))]


class _Resp(io.BytesIO):
    headers = {"Content-Type": "text/plain"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _page(text: str = "a", size: int = PAGE_CHARS) -> _Resp:
    return _Resp((text * size).encode("utf-8"))


class WebFetchOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self._url_counter = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _next_url(self) -> str:
        self._url_counter += 1
        return f"https://example.com/offset-page-{uuid.uuid4().hex}/"

    def _run(self, url: str, **tool_input):
        with patch.object(socket, "getaddrinfo", return_value=PUBLIC_ADDR):
            with patch.object(urllib.request, "urlopen", return_value=_page()):
                return WebFetchTool().run({"url": url, **tool_input}, self.ctx).output

    def test_first_call_truncates_with_accurate_counts(self) -> None:
        url = self._next_url()
        out = self._run(url)
        expected = (
            "a" * 100_000
            + "\n\n... [truncated: showing 0–100000 of 250000 chars; "
            + "pass offset=100000 to continue]"
        )
        self.assertEqual(out["content"], expected)
        self.assertEqual(out["content_type"], "text/plain")
        self.assertEqual(out["url"], url)

    def test_offset_returns_next_chunk(self) -> None:
        url = self._next_url()
        self._run(url)
        out = self._run(url, offset=100_000)
        expected = (
            "a" * 100_000
            + "\n\n... [truncated: showing 100000–200000 of 250000 chars; "
            + "pass offset=200000 to continue]"
        )
        self.assertEqual(out["content"], expected)

    def test_final_chunk_has_no_truncation_notice(self) -> None:
        url = self._next_url()
        self._run(url)
        out = self._run(url, offset=200_000)
        self.assertEqual(out["content"], "a" * 50_000)
        self.assertNotIn("[truncated", out["content"])

    def test_offset_uses_cache_without_refetch(self) -> None:
        url = self._next_url()
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return _page()

        with patch.object(socket, "getaddrinfo", return_value=PUBLIC_ADDR):
            with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
                WebFetchTool().run({"url": url}, self.ctx)
                out = WebFetchTool().run({"url": url, "offset": 100_000}, self.ctx).output

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(out["content"].split("\n\n... [truncated")[0]), 100_000)

    def test_offset_beyond_end_returns_empty_with_clean_message(self) -> None:
        url = self._next_url()
        out = self._run(url, offset=300_000)
        self.assertEqual(
            out["content"],
            "... [end of content: offset 300000 is past the end of the page (250000 chars total)]",
        )

    def test_offset_at_exact_end_returns_clean_message(self) -> None:
        url = self._next_url()
        out = self._run(url, offset=250_000)
        self.assertIn("past the end of the page", out["content"])

    def test_negative_offset_rejected(self) -> None:
        with self.assertRaises(ToolInputError):
            WebFetchTool().run({"url": "https://example.com/", "offset": -1}, self.ctx)

    def test_non_integer_offset_rejected(self) -> None:
        with self.assertRaises(ToolInputError):
            WebFetchTool().run({"url": "https://example.com/", "offset": "100"}, self.ctx)

    def test_short_page_needs_no_offset(self) -> None:
        short = "hello world"
        with patch.object(socket, "getaddrinfo", return_value=PUBLIC_ADDR):
            with patch.object(urllib.request, "urlopen", return_value=_Resp(short.encode("utf-8"))):
                out = WebFetchTool().run({"url": self._next_url()}, self.ctx).output
        self.assertEqual(out["content"], short)
        self.assertNotIn("[truncated", out["content"])

    def test_cache_evicts_oldest_url_after_five(self) -> None:
        urls = [self._next_url() for _ in range(6)]
        # 100k-char pages: 6 fit under the 1MB byte cap, so eviction is by URL count.
        with patch.object(socket, "getaddrinfo", return_value=PUBLIC_ADDR):
            with patch.object(urllib.request, "urlopen", return_value=_page(size=100_000)):
                for url in urls:
                    WebFetchTool().run({"url": url}, self.ctx)
        self.assertNotIn(urls[0], web_fetch_module._fetch_cache)
        for url in urls[1:]:
            self.assertIn(url, web_fetch_module._fetch_cache)

    def test_cache_keeps_most_recent_five(self) -> None:
        self.assertLessEqual(len(web_fetch_module._fetch_cache), 5)
        total = sum(len(text) for text, _ in web_fetch_module._fetch_cache.values())
        self.assertLessEqual(total, 1_000_000)


if __name__ == "__main__":
    unittest.main()
