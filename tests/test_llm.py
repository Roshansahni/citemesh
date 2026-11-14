import pytest

from citemesh import llm as llm_mod
from citemesh.config import load_config
from citemesh.llm import LLM, LLMError


class Resp:
    def __init__(self, code, data):
        self.status_code, self._d, self.text = code, data, str(data)
    def json(self):
        return self._d


def cfg(provider="ollama"):
    c = load_config()
    c["provider"] = provider
    c["llm"]["retries"] = 2
    return c


def test_ollama_chat_and_json_repair(monkeypatch):
    replies = iter(["not json at all", '{"ok": true}'])
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: Resp(200, {
        "message": {"content": next(replies)}, "prompt_eval_count": 5, "eval_count": 7}))
    m = LLM(cfg())
    assert m.chat_json("sys", "hi") == {"ok": True}  # first reply broken -> repaired
    assert m.calls == 2 and m.tokens == 24


def test_retry_on_429_then_success(monkeypatch):
    seq = iter([Resp(429, "slow down"), Resp(200, {"message": {"content": "hi"}})])
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: next(seq))
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    assert LLM(cfg()).chat("s", "u") == "hi"


def test_bad_model_fails_fast_and_cloud_needs_key(monkeypatch):
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: Resp(404, "model not found"))
    with pytest.raises(LLMError, match="404"):
        LLM(cfg()).chat("s", "u")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        LLM(cfg("groq")).chat("s", "u")


def test_groq_response_shape(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: Resp(200, {
        "choices": [{"message": {"content": "yo"}}], "usage": {"total_tokens": 9}}))
    m = LLM(cfg("groq"))
    assert m.chat("s", "u") == "yo" and m.tokens == 9
