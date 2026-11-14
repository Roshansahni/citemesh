"""Search tools: DuckDuckGo (ddgs, no key), arXiv API, Wikipedia API. All free.

Each returns a list of dicts: title, url, snippet, kind (and optional text).
Failures return [] so one flaky source never kills a run.
"""
import logging
import xml.etree.ElementTree as ET

import requests

log = logging.getLogger("citemesh.tools")
UA = "CiteMesh/0.1 (student research project; contact: you@example.com)"


def web_search(query: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS

        rows = DDGS().text(query, max_results=n) or []
    except Exception as e:  # ddgs raises several types (rate limit, network)
        log.warning("web search failed: %s", e)
        return []
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", ""), "kind": "web"}
        for r in rows
        if r.get("href")
    ]


def arxiv_search(query: str, n: int) -> list[dict]:
    try:
        r = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": n},
            headers={"User-Agent": UA}, timeout=20,
        )
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except (requests.RequestException, ET.ParseError) as e:
        log.warning("arxiv failed: %s", e)
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        summary = " ".join((e.findtext("a:summary", "", ns) or "").split())
        url = (e.findtext("a:id", "", ns) or "").strip()
        if url:
            out.append({"title": title, "url": url, "snippet": summary[:300], "text": summary, "kind": "arxiv"})
    return out


def wikipedia_search(query: str, n: int) -> list[dict]:
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": n,
                "prop": "extracts|info", "exintro": 1, "explaintext": 1, "inprop": "url", "format": "json",
            },
            headers={"User-Agent": UA}, timeout=20,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
    except (requests.RequestException, ValueError) as e:
        log.warning("wikipedia failed: %s", e)
        return []
    out = []
    for p in sorted(pages.values(), key=lambda x: x.get("index", 0)):
        text = (p.get("extract") or "").strip()
        if text and p.get("fullurl"):
            out.append({"title": p.get("title", ""), "url": p["fullurl"], "snippet": text[:300], "text": text, "kind": "wikipedia"})
    return out
