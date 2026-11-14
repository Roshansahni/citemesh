from citemesh.runner import run_research
from tests.helpers import FakeLLM, QUOTE, fake_tools, make_cfg

GOOD = ("# RAG\n\n## Summary\nRAG grounds answers in retrieved documents and so reduces unsupported claims [E1].\n\n"
        "## What is RAG?\nIt retrieves documents before generating text, which grounds the answer [E1].\n\n"
        "## Limitations\nSmall evidence base.\n")
UNCITED = "# RAG\n\n## Summary\nRAG is a technique that makes language models much more accurate in practice.\n"


def run(llm, tools=None, tmp=None, **kw):
    return run_research("Does RAG reduce hallucination?", make_cfg(tmp), llm=llm, tools=tools or fake_tools(), **kw)


def test_happy_path(tmp_path):
    s = run(FakeLLM([GOOD]), tmp=tmp_path)
    assert s["critique"]["verdict"] == "pass" and s["iteration"] == 1
    assert "## Evidence & References" in s["final_report"] and QUOTE in s["final_report"]
    assert s["stats"]["evidence"] == 1  # duplicate quote across sources is deduplicated
    assert (tmp_path / "does-rag-reduce-hallucination.md").exists()


def test_hallucinated_quote_is_rejected_and_no_evidence_stops(tmp_path):
    llm = FakeLLM([GOOD], reader_quotes=["This sentence never appeared in any source at all"])
    s = run(llm, tmp=tmp_path)
    assert s["evidence"] == [] and "No verifiable evidence" in s["final_report"]
    assert s["draft"] == ""  # writer never ran


def test_revision_loop_fixes_uncited_draft(tmp_path):
    s = run(FakeLLM([UNCITED, GOOD]), tmp=tmp_path)
    assert s["iteration"] == 2 and s["critique"]["verdict"] == "pass"


def test_loop_is_bounded_and_warns(tmp_path):
    s = run(FakeLLM([UNCITED]), tmp=tmp_path)  # writer never improves
    assert s["iteration"] == 2  # max_iterations
    assert "Warning" in s["final_report"]


def test_invalid_citation_is_stripped(tmp_path):
    bad = GOOD + "\nExtra claim that is completely made up by the model itself [E99].\n"
    s = run(FakeLLM([bad]), tmp=tmp_path, )
    assert "[E99]" not in s["final_report"]


def test_plan_approval_hook_can_edit_plan(tmp_path):
    s = run(FakeLLM([GOOD]), tmp=tmp_path, approve=lambda p: ["Only this question"])
    assert s["plan"] == ["Only this question"]


def test_prefilled_plan_skips_planner_llm(tmp_path):
    s = run(FakeLLM([GOOD]), tmp=tmp_path, plan=["A custom sub question"])
    assert s["plan"] == ["A custom sub question"]
