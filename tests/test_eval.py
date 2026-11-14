import json

import run_eval
from citemesh.metrics import claim_sentences, evaluate_report
from tests.helpers import FakeLLM, QUOTE, fake_tools, make_cfg

EV = [{"id": "E1", "source_id": "S1", "quote": QUOTE}]
SRC = [{"id": "S1", "title": "t", "url": "https://example.org/rag", "kind": "web"}]
DRAFT = ("# T\n\n## Summary\nRetrieval augmented generation grounds answers in retrieved documents. [E1]\n"
         "Large language models always hallucinate in every conceivable situation whatsoever.\n"
         "Another claim that cites a missing identifier and is long enough to count [E9].\n"
         "The provided evidence does not provide information on benchmarks at all really.\n"
         "## Limitations\nThis sentence is long enough but sits in the limitations section.\n")


def test_trailing_citation_stays_with_sentence():
    assert any(s.endswith("[E1]") for s in claim_sentences(DRAFT))
    assert not any("limitations section" in s for s in claim_sentences(DRAFT))
    assert not any("does not provide" in s for s in claim_sentences(DRAFT))


def test_evaluate_report_metrics():
    m = evaluate_report(DRAFT, EV, SRC)
    assert m["claims"] == 3
    assert m["cited_claim_rate"] == 1 / 3 and m["claim_support_rate"] == 1 / 3
    assert m["citation_validity"] == 0.5  # [E1] valid, [E9] invalid
    assert m["cited_domains"] == 1 and m["evidence_used"] == 1.0


def test_baseline_has_zero_traceability():
    m = evaluate_report(DRAFT.replace("[E1]", "").replace("[E9]", ""), [], [])
    assert m["cited_claim_rate"] == 0.0 and m["citation_validity"] is None


def test_run_one_and_summarize(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "OUT", tmp_path)
    good = ("# R\n\n## Summary\nRetrieval augmented generation grounds answers in retrieved documents [E1].\n")
    q = {"id": "q01", "category": "mechanism", "question": "What is RAG?", "expect_refusal": False}
    cfg = make_cfg(tmp_path)
    rows = [run_eval.run_one(n, q, cfg, fake_tools(), llm_factory=lambda c: FakeLLM([good])) for n in run_eval.CONFIGS]
    assert [r["cited_claim_rate"] for r in rows] == [0.0, 1.0, 1.0]  # baseline, no_revision, full
    table = run_eval.summarize(rows)
    assert "| full |" in table and "| baseline |" in table
    json.dumps(rows)  # rows must be JSON-serialisable for runs.jsonl
