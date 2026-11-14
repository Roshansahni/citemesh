"""Toolbox: the only way agents touch the outside world (easy to fake in tests)."""
from dataclasses import dataclass
from typing import Callable

from ..cache import Cache
from .fetch import fetch_text
from .pdfs import load_local_pdfs
from .web import arxiv_search, web_search, wikipedia_search

SEARCHERS = {"web": web_search, "arxiv": arxiv_search, "wikipedia": wikipedia_search}


@dataclass
class Toolbox:
    search: Callable[[str], list[list[dict]]]   # query -> one result list per source type
    fetch: Callable[[str], str]                  # url -> text
    local: Callable[[], list[dict]]              # -> user's PDFs as sources


def default_toolbox(cfg: dict) -> Toolbox:
    r = cfg["research"]
    cache = Cache(r["cache_file"])

    def search(query: str) -> list[list[dict]]:
        groups = []
        for name in r["sources"]:
            key = f"{name}|{query}|{r['results_per_source']}"
            hit = cache.get("search", key)
            if hit is None:
                hit = SEARCHERS[name](query, r["results_per_source"])
                if hit:  # never cache failures
                    cache.set("search", key, hit)
            groups.append(hit)
        return groups

    def fetch(url: str) -> str:
        hit = cache.get("page", url)
        if hit is None:
            hit = fetch_text(url, r["max_page_chars"])
            if hit:
                cache.set("page", url, hit)
        return hit or ""

    return Toolbox(search=search, fetch=fetch, local=lambda: load_local_pdfs(r["local_pdf_dir"]))
