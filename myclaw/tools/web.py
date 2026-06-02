from __future__ import annotations

import html
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlsplit
from urllib.request import Request, urlopen

from myclaw.tools.base import Tool


class WebFetchTool(Tool):
    read_only = True
    exclusive = False

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch a public http/https URL and return simplified text content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["url"],
        }

    async def execute(self, url: str | None = None, max_chars: int = 6000, **kwargs: Any) -> dict[str, Any] | str:
        error = _validate_public_http_url(url)
        if error is not None:
            return error
        assert url is not None
        try:
            limit = max(1, int(max_chars))
        except (TypeError, ValueError):
            return "Error: max_chars must be an integer"
        try:
            text, content_type = _fetch_text(url)
        except Exception as exc:
            return f"Error fetching URL: {exc}"
        title = _html_title(text)
        content = _html_to_text(text) if "html" in content_type.lower() else text
        return {
            "url": url,
            "title": title,
            "content": content[:limit],
        }


class WebSearchTool(Tool):
    read_only = True
    exclusive = False

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the public web and return a small set of result titles and URLs."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        }

    async def execute(self, query: str | None = None, max_results: int = 5, **kwargs: Any) -> dict[str, Any] | str:
        if query is None or not str(query).strip():
            return "Error: query is required"
        try:
            limit = max(1, min(10, int(max_results)))
        except (TypeError, ValueError):
            return "Error: max_results must be an integer"
        url = f"https://duckduckgo.com/html/?q={quote_plus(str(query))}"
        try:
            text, _content_type = _fetch_text(url)
        except Exception as exc:
            return f"Error searching web: {exc}"
        return {"query": str(query), "results": _parse_duckduckgo_results(text, limit)}


def _fetch_text(url: str) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": "myclaw/0.1"})
    with urlopen(request, timeout=10) as response:
        payload = response.read()
        content_type = response.headers.get("content-type", "")
    charset_match = re.search(r"charset=([A-Za-z0-9_.-]+)", content_type)
    encoding = charset_match.group(1) if charset_match else "utf-8"
    return payload.decode(encoding, errors="replace"), content_type


def _validate_public_http_url(url: str | None) -> str | None:
    if url is None or not str(url).strip():
        return "Error: url is required"
    parsed = urlsplit(str(url))
    if parsed.scheme not in {"http", "https"}:
        return "Error: only http and https URLs are supported"
    if not parsed.hostname:
        return "Error: URL host is required"
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return "Error: blocked private or local address"
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror:
        return "Error: cannot resolve host"
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return "Error: blocked private or local address"
    return None


def _html_title(text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return html.unescape(_collapse_ws(match.group(1)))


def _html_to_text(text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(_collapse_ws(text))


def _parse_duckduckgo_results(text: str, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for raw_url, raw_title in pattern.findall(text):
        url = _clean_duckduckgo_url(html.unescape(raw_url))
        title = html.unescape(_collapse_ws(re.sub(r"<[^>]+>", " ", raw_title)))
        if not title or not url:
            continue
        results.append({"title": title, "url": url})
        if len(results) >= limit:
            break
    return results


def _clean_duckduckgo_url(url: str) -> str:
    if "duckduckgo.com/l/" not in url:
        return url
    query = parse_qs(urlsplit(url).query)
    target = query.get("uddg", [""])[0]
    return unquote(target)


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())
