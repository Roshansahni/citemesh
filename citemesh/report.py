"""Assemble the final Markdown report: strip bad citations, append evidence list."""
import re
from pathlib import Path

from .utils import slugify

CITE = re.compile(r"\[E(\d+)\]")


def build_final(draft: str, evidence: list[dict], sources: list[dict], critique: dict, iterations: int) -> str:
    valid = {e["id"] for e in evidence}
    body = CITE.sub(lambda m: m.group(0) if f"E{m.group(1)}" in valid else "", draft).strip()
    cited = [e for e in evidence if f"[{e['id']}]" in body]
    by_src = {s["id"]: s for s in sources}
    lines = ["", "## Evidence & References", ""]
    for e in cited:
        s = by_src[e["source_id"]]
        lines.append(f'- **{e["id"]}** "{e["quote"]}" - [{s["title"] or s["url"]}]({s["url"]}) ({s["kind"]}, {s["id"]})')
    if not cited:
        lines.append("_The draft cited no evidence; treat it with caution._")
    out = body + "\n" + "\n".join(lines)
    n_issues = len(critique.get("invalid_ids", [])) + len(critique.get("uncited", [])) + len(critique.get("issues", []))
    if critique.get("verdict") == "revise" and n_issues:
        out += (f"\n\n> **Warning:** automated checks still flagged {n_issues} issue(s) after "
                f"{iterations} pass(es). Verify claims against the evidence list.")
    return out + "\n"


def save_report(text: str, question: str, folder: str) -> str:
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(folder) / f"{slugify(question)}.md"
    path.write_text(text, encoding="utf-8")
    return str(path)
