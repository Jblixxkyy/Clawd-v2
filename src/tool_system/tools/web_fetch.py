from __future__ import annotations

import html
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
from typing import Any
from collections import OrderedDict

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..protocol import ToolResult
from ..registry import ToolSpec


_TAG_RE = re.compile(r"<[^>]+>")

# Max chars returned per WebFetch call before truncation.
_CHUNK_SIZE = 100_000

# Bounded session cache of full extracted page text per URL, so offset reads
# don't re-fetch. Last N URLs, hard cap on total cached bytes.
_CACHE_MAX_URLS = 5
_CACHE_MAX_BYTES = 1_000_000

_fetch_cache: "OrderedDict[str, tuple[str, str]]" = OrderedDict()


def _cache_get(url: str) -> tuple[str, str] | None:
    entry = _fetch_cache.get(url)
    if entry is not None:
        # refresh recency
        _fetch_cache.move_to_end(url)
    return entry


def _cache_store(url: str, text: str, content_type: str) -> None:
    if len(text) > _CACHE_MAX_BYTES:
        return
    _fetch_cache.pop(url, None)
    _fetch_cache[url] = (text, content_type)
    total = sum(len(v[0]) for v in _fetch_cache.values())
    while len(_fetch_cache) > _CACHE_MAX_URLS or total > _CACHE_MAX_BYTES:
        _, evicted = _fetch_cache.popitem(last=False)
        total -= len(evicted[0])


def _is_private_host(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


def _html_to_text(raw: str) -> str:
    without_tags = _TAG_RE.sub(" ", raw)
    without_tags = re.sub(r"\s+", " ", without_tags).strip()
    return html.unescape(without_tags)


class WebFetchTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="WebFetch",
            description=(
                "Fetches a URL and returns its extracted text. Use it to read documentation or pages the "
                "user links to. It cannot access pages that need a login. Content is returned in 100,000 "
                "character chunks; if the output says it was truncated, pass `offset` with the suggested "
                "value to continue reading from where the previous chunk ended. If it fails or the content "
                "is truncated, say so rather than guessing at the page contents."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {"type": "string"},
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Character offset to continue reading a truncated page from. "
                            "Use the offset value given in the truncation message."
                        ),
                    },
                },
                "required": ["url"],
            },
            is_read_only=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        url = tool_input["url"]
        if not isinstance(url, str) or not url:
            raise ToolInputError("url must be a non-empty string")

        offset = tool_input.get("offset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolInputError("offset must be a non-negative integer")

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolPermissionError("only http/https URLs are allowed")
        if not parsed.netloc:
            raise ToolInputError("url must include a network location")

        hostname = parsed.hostname or ""
        if hostname in {"localhost"} or hostname.endswith(".localhost") or _is_private_host(hostname):
            raise ToolPermissionError("refusing to fetch localhost/private network URLs")

        entry = _cache_get(url)
        if entry is None:
            req = urllib.request.Request(url, headers={"User-Agent": "clawd-codex/0.1"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_bytes = resp.read(1_000_000)
                content_type = resp.headers.get("Content-Type", "")

            text = raw_bytes.decode("utf-8", errors="replace")
            if "text/html" in content_type:
                text = _html_to_text(text)
            _cache_store(url, text, content_type)
        else:
            text, content_type = entry

        total = len(text)
        if offset >= total:
            content = f"... [end of content: offset {offset} is past the end of the page ({total} chars total)]"
        else:
            chunk = text[offset : offset + _CHUNK_SIZE]
            end = offset + len(chunk)
            if end < total:
                content = (
                    chunk
                    + f"\n\n... [truncated: showing {offset}–{end} of {total} chars; "
                    + f"pass offset={end} to continue]"
                )
            else:
                content = chunk

        return ToolResult(
            name="WebFetch",
            output={"url": url, "content_type": content_type, "content": content},
        )

