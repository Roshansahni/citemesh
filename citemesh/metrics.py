"""Deterministic evaluation metrics for generated reports (no LLM judge, fully reproducible).

LIMITS (state these in your report): claim support is a LEXICAL proxy (word overlap
between a sentence and its cited quote), not true semantic entailment. Validate it
by hand-auditing a sample of claims.
"""
import re
from urllib.parse import urlparse

CITE = re.compile(r"\[E(\d+)\]")
SUPPORT_THRESHOLD = 0.4  # share of a claim's content words that must appear in its cited quote(s)
STOP = set("the a an of to in and or is are was were for on with as by at from that this it be "
           "can has have not which their its into than also such these those more most other".split())
SKIP = ("no evidence", "insufficient", "does not provide", "do not provide", "not explicitly",
        "not enough", "cannot be determined")  # honest "I don't know" lines are not claims


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", CITE.sub(" ", s).lower()) if len(w) > 2 and w not in STOP}


def claim_sentences(text: str) -> list[str]:
    """Factual sentences of a report body (excludes headings, Limitations, references)."""
    body = re.split(r"^##\s*Evidence", text, flags=re.M)[0]
    out, section = [], ""
    for line in body.splitlines():
        s = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).strip()
        if not s:
            continue
        if s.startswith("#"):
            section = s.lower()
            continue
        if "limitation" in section:
            continue
        # do not split before trailing citations: "... search. [E4] [E5]"
        for sent in re.split(r"(?<=[.!?])\s+(?!\[E\d+\])", s):
            sent = sent.strip()
            if len(sent) >= 40 and not any(p in sent.lower() for p in SKIP):
                out.append(sent)
    return out


def evaluate_report(text: str, evidence: list[dict], sources: list[dict]) -> dict:
    ev = {e["id"]: e for e in evidence}
    src = {s["id"]: s for s in sources}
    cites = CITE.findall(text)
    valid = [c for c in cites if f"E{c}" in ev]
    claims = claim_sentences(text)
    cited = supported = 0
    used: set[str] = set()
    for c in claims:
        ids = [f"E{m}" for m in CITE.findall(c) if f"E{m}" in ev]
        if not ids:
            continue
        cited += 1
        used.update(ids)
        quote_words = set().union(*(words(ev[i]["quote"]) for i in ids))
        w = words(c)
        if w and len(w & quote_words) / len(w) >= SUPPORT_THRESHOLD:
            supported += 1
    cited_src = [src[ev[i]["source_id"]] for i in used if ev[i]["source_id"] in src]
    n = len(claims)
    return {
        "claims": n,
        "citation_validity": (len(valid) / len(cites)) if cites else None,
        "cited_claim_rate": cited / n if n else 0.0,
        "claim_support_rate": supported / n if n else 0.0,
        "evidence_used": len(used) / len(ev) if ev else 0.0,
        "cited_sources": len({s["id"] for s in cited_src}),
        "cited_kinds": len({s["kind"] for s in cited_src}),
        "cited_domains": len({urlparse(s["url"]).netloc for s in cited_src}),
    }


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
