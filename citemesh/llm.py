"""LLM provider layer: Ollama (local) or Groq/Gemini (free cloud, OpenAI-compatible).

One class, one switch (`provider` in config.yaml). All agents share it.
Cloud providers are NOT private: never send sensitive data to them.
"""
import json
import logging
import os
import re
import time

import requests

log = logging.getLogger("citemesh.llm")

OPENAI_COMPAT = {  # VERIFY endpoints in each provider's docs
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}
KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}


class LLMError(RuntimeError):
    pass


def parse_json(raw: str):
    """Best-effort JSON object extraction from model output; None on failure."""
    raw = re.sub(r"```(?:json)?", "", raw or "")
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class LLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = cfg["provider"]
        if self.provider not in ("ollama", *OPENAI_COMPAT):
            raise LLMError(f"Unknown provider '{self.provider}'")
        self.calls = 0
        self.tokens = 0
        self.embed_ok = True

    # ---- HTTP with retry/backoff -------------------------------------------------
    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        s = self.cfg["llm"]
        last = None
        for attempt in range(s["retries"]):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=s["timeout"])
            except requests.ConnectionError as e:
                hint = " Is Ollama running? Open the Ollama app or run `ollama serve`." if self.provider == "ollama" else ""
                last = LLMError(f"Cannot connect to {url}.{hint}")
                log.warning("connection error: %s", e)
            except requests.Timeout:
                last = LLMError("LLM call timed out (try a smaller model or raise llm.timeout).")
            else:
                if r.status_code in (429, 500, 502, 503, 504):
                    last = LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                elif r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")  # bad model/key: don't retry
                else:
                    return r.json()
            time.sleep(2 * (2**attempt))
        raise last or LLMError("LLM call failed")

    # ---- chat ----------------------------------------------------------------------
    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        t0 = time.time()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        temp = self.cfg["llm"]["temperature"]
        if self.provider == "ollama":
            o = self.cfg["ollama"]
            payload = {
                "model": o["chat_model"], "messages": msgs, "stream": False,
                "options": {"temperature": temp, "num_ctx": self.cfg["llm"]["num_ctx"]},
            }
            if json_mode:
                payload["format"] = "json"
            data = self._post(f"{o['host']}/api/chat", payload)
            text = data["message"]["content"]
            self.tokens += data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        else:
            key = os.getenv(KEY_ENV[self.provider])
            if not key:
                raise LLMError(f"Set {KEY_ENV[self.provider]} in .env to use '{self.provider}'.")
            payload = {"model": self.cfg[self.provider]["model"], "messages": msgs, "temperature": temp}
            data = self._post(OPENAI_COMPAT[self.provider], payload, {"Authorization": f"Bearer {key}"})
            text = data["choices"][0]["message"]["content"]
            self.tokens += data.get("usage", {}).get("total_tokens", 0)
        self.calls += 1
        log.info("llm call #%d %.1fs in=%d out=%d chars", self.calls, time.time() - t0, len(user), len(text))
        return text

    def chat_json(self, system: str, user: str, default: dict | None = None) -> dict:
        """Ask for JSON; on bad output try one repair call; else return `default`."""
        raw = self.chat(system + "\nReply with ONE valid JSON object and nothing else.", user, json_mode=True)
        obj = parse_json(raw)
        if obj is None:
            fixed = self.chat("Repair this into one valid JSON object. Output only JSON.", raw[:3000], json_mode=True)
            obj = parse_json(fixed)
        return obj if obj is not None else (default if default is not None else {})

    # ---- embeddings (always local Ollama; callers fall back to keywords) ------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        o = self.cfg["ollama"]
        try:
            r = requests.post(f"{o['host']}/api/embed", json={"model": o["embed_model"], "input": texts}, timeout=120)
            r.raise_for_status()
            return r.json()["embeddings"]
        except (requests.RequestException, KeyError, ValueError) as e:
            raise LLMError(f"Embedding failed: {e}") from e
