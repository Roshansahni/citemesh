"""Chunking + retrieval. Embeddings via Ollama if available, else keyword overlap."""
import logging
import math
import re

from .llm import LLMError

log = logging.getLogger("citemesh.knowledge")
STOP = set("the a an of to in and or is are was were for on with as by at from that this it be".split())


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP}


def keyword_rank(query: str, chunks: list[str], k: int) -> list[str]:
    q = _tokens(query)
    order = sorted(range(len(chunks)), key=lambda i: (-len(q & _tokens(chunks[i])), i))
    return [chunks[i] for i in sorted(order[:k])]  # keep document order


def _cos(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def retrieve(llm, query: str, chunks: list[str], k: int, use_embeddings: bool = True) -> list[str]:
    if len(chunks) <= k:
        return chunks
    if use_embeddings and getattr(llm, "embed_ok", False):
        try:
            vecs = llm.embed([query] + chunks)
            scores = [(_cos(vecs[0], v), i) for i, v in enumerate(vecs[1:])]
            top = sorted(scores, reverse=True)[:k]
            return [chunks[i] for _, i in sorted(top, key=lambda x: x[1])]
        except LLMError as e:
            llm.embed_ok = False  # don't retry every source
            log.warning("embeddings unavailable, using keywords: %s", e)
    return keyword_rank(query, chunks, k)
