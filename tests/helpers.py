"""Fake LLM + fake tools so tests need no Ollama and no internet."""
from citemesh.config import load_config
from citemesh.llm import LLMError
from citemesh.tools import Toolbox

QUOTE = "Retrieval augmented generation grounds answers in retrieved documents"
TEXT = f"Intro text here. {QUOTE}, which reduces unsupported statements. More filler text follows."


class FakeLLM:
    embed_ok = False

    def __init__(self, drafts, reader_quotes=None, critic_issues=None):
        self.drafts = list(drafts)
        self.reader_quotes = reader_quotes if reader_quotes is not None else [QUOTE]
        self.critic_issues = critic_issues or []
        self.calls = 0
        self.tokens = 0

    def chat_json(self, system, user, default=None):
        self.calls += 1
        if "Planner" in system:
            return {"sub_questions": ["What is RAG?", "Does RAG reduce hallucination?"]}
        if "evidence extractor" in system:
            return {"evidence": [{"quote": q, "sub_question": 1} for q in self.reader_quotes]}
        if "fact-checker" in system:
            return {"issues": self.critic_issues}
        return default or {}

    def chat(self, system, user, json_mode=False):
        self.calls += 1
        return self.drafts.pop(0) if len(self.drafts) > 1 else self.drafts[0]

    def embed(self, texts):
        raise LLMError("no embeddings in tests")


def fake_tools(sources=None):
    src = sources if sources is not None else [
        {"title": "RAG overview", "url": "https://example.org/rag", "snippet": "s", "text": TEXT, "kind": "web"},
        {"title": "RAG paper", "url": "https://arxiv.org/abs/1", "snippet": "s", "text": TEXT + " Extra.", "kind": "arxiv"},
    ]
    return Toolbox(search=lambda q: [list(src)], fetch=lambda url: "", local=lambda: [])


def make_cfg(tmp_path):
    cfg = load_config()
    cfg["research"]["output_dir"] = str(tmp_path)
    cfg["research"]["max_sources"] = 4
    return cfg
