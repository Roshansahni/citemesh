"""Shared state passed between agents (LangGraph TypedDict schema)."""
from typing import TypedDict


class ResearchState(TypedDict, total=False):
    question: str
    plan: list[str]          # approved sub-questions
    sources: list[dict]      # {id:'S1', title, url, kind, snippet, text?}
    evidence: list[dict]     # {id:'E1', source_id, quote, sub_question}
    critique: dict           # {verdict, invalid_ids, uncited, issues}
    draft: str
    final_report: str
    iteration: int           # completed critic passes
    events: list[str]        # human-readable progress for CLI/UI
    stats: dict
