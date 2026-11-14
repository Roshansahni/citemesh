"""CLI:  python main.py "your research question"   (add --yes to skip plan approval)"""
import argparse

from citemesh.config import load_config
from citemesh.llm import LLMError
from citemesh.runner import run_research, setup_logging


def approve_cli(plan: list[str]) -> list[str]:
    print("\nProposed research plan:")
    for i, q in enumerate(plan, 1):
        print(f"  {i}. {q}")
    reply = input("\nPress Enter to approve, or type replacement sub-questions separated by ';': ").strip()
    return [p.strip() for p in reply.split(";") if p.strip()] if reply else plan


def main() -> None:
    ap = argparse.ArgumentParser(description="CiteMesh multi-agent research assistant")
    ap.add_argument("question", nargs="*", help="research question")
    ap.add_argument("--provider", choices=["ollama", "groq", "gemini"], help="override config.yaml")
    ap.add_argument("--yes", action="store_true", help="auto-approve the plan")
    args = ap.parse_args()
    question = " ".join(args.question) or input("Research question: ").strip()
    cfg = load_config()
    if args.provider:
        cfg["provider"] = args.provider
    setup_logging(cfg)
    try:
        state = run_research(question, cfg, approve=None if args.yes else approve_cli,
                             on_event=lambda node, msg: print(f"[{node:8}] {msg}", flush=True))
    except LLMError as e:
        raise SystemExit(f"\nLLM error: {e}")
    print("\n" + "=" * 70 + "\n" + state["final_report"])
    print(f"Saved: {state['report_path']}\nStats: {state['stats']}")


if __name__ == "__main__":
    main()
