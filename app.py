"""Streamlit UI:  streamlit run app.py"""
import copy

import streamlit as st

from citemesh.agents import make_plan
from citemesh.config import load_config
from citemesh.llm import LLM, LLMError
from citemesh.runner import run_research, setup_logging

st.set_page_config(page_title="CiteMesh", page_icon="🔎", layout="wide")
base = load_config()
setup_logging(base)

st.title("🔎 CiteMesh - multi-agent research assistant")
st.caption("Every claim in the report is tied to a verbatim quote from a real source.")

providers = ["ollama", "groq", "gemini"]
provider = st.sidebar.selectbox("LLM provider", providers, index=providers.index(base["provider"]))
st.sidebar.info("Ollama runs locally and privately. Groq/Gemini are free cloud tiers: do not send sensitive data.")
cfg = copy.deepcopy(base)
cfg["provider"] = provider

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
        plan = [ln.strip() for ln in text.splitlines() if ln.strip()]
        try:
            with st.status("Agents working...", expanded=True) as status:
                st.session_state.result = run_research(
                    question, cfg, plan=plan, on_event=lambda node, msg: st.write(f"**{node}** - {msg}"))
                status.update(label="Done", state="complete")
        except LLMError as e:
            st.error(str(e))

res = st.session_state.get("result")
if res:
    st.divider()
    st.markdown(res["final_report"])
    st.download_button("Download Markdown", res["final_report"], file_name="report.md", mime="text/markdown")
    st.caption(f"Stats: {res['stats']}")
