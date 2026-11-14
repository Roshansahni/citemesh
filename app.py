"""Streamlit UI:  streamlit run app.py   (also the entry point on Streamlit Community Cloud)"""
import copy
import hmac
import os

import requests
import streamlit as st

st.set_page_config(page_title="CiteMesh", page_icon="🔎", layout="wide")

# Streamlit Cloud stores secrets in st.secrets; the rest of the code reads environment variables.
for _key in ("LLM_PROVIDER", "GROQ_API_KEY", "GEMINI_API_KEY", "APP_PASSWORD", "MAX_RUNS_PER_SESSION"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
    except Exception:  # no secrets file locally: fine
        pass

from citemesh.agents import make_plan  # noqa: E402  (must import after secrets are in the environment)
from citemesh.config import load_config  # noqa: E402
from citemesh.llm import LLM, LLMError  # noqa: E402
from citemesh.runner import run_research, setup_logging  # noqa: E402

base = load_config()
setup_logging(base)


@st.cache_data(ttl=30)
def ollama_up(host: str) -> bool:
    try:
        return requests.get(f"{host}/api/tags", timeout=1.5).ok
    except requests.RequestException:
        return False


# Optional password gate so a public link cannot burn your free API quota.
password = os.getenv("APP_PASSWORD")
if password and not st.session_state.get("authed"):
    entered = st.text_input("Password", type="password")
    if entered and hmac.compare_digest(entered, password):
        st.session_state.authed = True
        st.rerun()
    if entered:
        st.error("Wrong password.")
    st.stop()

st.title("🔎 CiteMesh - multi-agent research assistant")
st.caption("Every claim in the report is tied to a verbatim quote from a real source.")

local_ok = ollama_up(base["ollama"]["host"])
providers = [p for p in ("ollama", "groq", "gemini") if p != "ollama" or local_ok]
default = providers.index(base["provider"]) if base["provider"] in providers else 0
provider = st.sidebar.selectbox("LLM provider", providers, index=default)
st.sidebar.info("Ollama runs locally and privately. Groq/Gemini are free cloud tiers: do not send sensitive data.")
cfg = copy.deepcopy(base)
cfg["provider"] = provider
cfg["research"]["use_embeddings"] = base["research"]["use_embeddings"] and local_ok

max_runs = int(os.getenv("MAX_RUNS_PER_SESSION", "5"))
runs_used = st.session_state.get("runs", 0)
question = st.text_input("Research question", placeholder="e.g. How do retrieval-augmented LLMs reduce hallucinations?")

if st.button("1. Draft plan") and question:
    try:
        with st.spinner("Planner is thinking..."):
            st.session_state.plan = make_plan(LLM(cfg), question, cfg["research"]["max_subquestions"])
        st.session_state.pop("result", None)
    except LLMError as e:
        st.error(str(e))

if "plan" in st.session_state:
    text = st.text_area("2. Approve or edit the plan (one sub-question per line)",
                        "\n".join(st.session_state.plan), height=120)
    if st.button("3. Run research"):
        if runs_used >= max_runs:
            st.warning(f"Run limit reached ({max_runs} per session). Reload the page to start a new session.")
        else:
            plan = [ln.strip() for ln in text.splitlines() if ln.strip()]
            try:
                with st.status("Agents working...", expanded=True) as status:
                    st.session_state.result = run_research(
                        question, cfg, plan=plan, on_event=lambda node, msg: st.write(f"**{node}** - {msg}"))
                    status.update(label="Done", state="complete")
                st.session_state.runs = runs_used + 1
            except LLMError as e:
                st.error(str(e))

res = st.session_state.get("result")
if res:
    st.divider()
    st.markdown(res["final_report"])
    st.download_button("Download Markdown", res["final_report"], file_name="report.md", mime="text/markdown")
    st.caption(f"Stats: {res['stats']}")
