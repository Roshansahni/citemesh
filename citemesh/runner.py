"""Run the graph, stream progress events, save the report."""
import logging
import time
from pathlib import Path

from .graph import build_graph
from .llm import LLM
from .report import save_report
from .tools import default_toolbox


def setup_logging(cfg: dict) -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(filename="logs/citemesh.log", level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")


def run_research(question, cfg, llm=None, tools=None, plan=None, approve=None, on_event=lambda node, msg: None):
    llm = llm or LLM(cfg)
    tools = tools or default_toolbox(cfg)
    app = build_graph(llm, cfg, tools, approve or (lambda p: p))
    state = {"question": question, "plan": plan or [], "sources": [], "evidence": [], "critique": {},
             "draft": "", "final_report": "", "iteration": 0, "events": [], "stats": {}}
    log, t0 = [], time.time()
    for update in app.stream(state, stream_mode="updates"):
        for node, delta in update.items():
            state.update(delta)
            for msg in delta.get("events", []):
                log.append(f"{node}: {msg}")
                on_event(node, msg)
    state["events"] = log
    state["stats"] = {"llm_calls": llm.calls, "tokens": llm.tokens, "seconds": round(time.time() - t0, 1),
                      "sources": len(state["sources"]), "evidence": len(state["evidence"])}
    state["report_path"] = save_report(state["final_report"], question, cfg["research"]["output_dir"])
    return state
