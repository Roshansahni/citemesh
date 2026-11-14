"""UI smoke tests with Streamlit's headless AppTest (no browser, no LLM)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import citemesh.agents
import citemesh.runner

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setattr(citemesh.agents, "make_plan", lambda llm, q, n: ["Sub one?", "Sub two?"])
    monkeypatch.setattr(citemesh.runner, "run_research", lambda q, cfg, plan=None, on_event=None, **k: (
        on_event("writer", "fake"), {"final_report": "# Report\n\n- claim [E1]", "stats": {"llm_calls": 1}})[1])


def test_full_flow_draft_plan_then_run():
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert not at.exception and at.sidebar.selectbox[0].value == "groq"  # no Ollama in CI -> cloud provider
    at.text_input[0].set_value("What is RAG?")
    at.button[0].click().run()
    assert at.text_area[0].value == "Sub one?\nSub two?"
    at.button[1].click().run()
    assert not at.exception
    assert any("claim [E1]" in m.value for m in at.markdown)


def test_password_gate(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(APP, default_timeout=20).run()
    assert at.text_input[0].label == "Password" and not at.button  # app body is hidden
    at.text_input[0].set_value("wrong").run()
    assert at.error and not at.button
    at.text_input[0].set_value("s3cret").run()
    assert at.button  # unlocked


def test_run_limit(monkeypatch):
    monkeypatch.setenv("MAX_RUNS_PER_SESSION", "1")
    at = AppTest.from_file(APP, default_timeout=20).run()
    at.text_input[0].set_value("What is RAG?")
    at.button[0].click().run()
    at.button[1].click().run()
    at.button[1].click().run()
    assert any("Run limit reached" in w.value for w in at.warning)
