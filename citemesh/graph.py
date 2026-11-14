"""LangGraph wiring:  planner -> searcher -> reader -> writer <-> critic -> finalize."""
from langgraph.graph import END, START, StateGraph

from .agents import build_agents
from .state import ResearchState


def build_graph(llm, cfg: dict, tools, approve=lambda plan: plan):
    planner, searcher, reader, writer, critic, finalize = build_agents(llm, cfg, tools, approve)
    max_iter = cfg["research"]["max_iterations"]

    def after_reader(state) -> str:
        return "writer" if state.get("evidence") else "finalize"  # no source, no report

    def after_critic(state) -> str:
        revise = state["critique"]["verdict"] == "revise"
        return "writer" if revise and state["iteration"] < max_iter else "finalize"  # loop guard

    g = StateGraph(ResearchState)
    for name, fn in [("planner", planner), ("searcher", searcher), ("reader", reader),
                     ("writer", writer), ("critic", critic), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "searcher")
    g.add_edge("searcher", "reader")
    g.add_conditional_edges("reader", after_reader, {"writer": "writer", "finalize": "finalize"})
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", after_critic, {"writer": "writer", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()
