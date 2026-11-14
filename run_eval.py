"""Benchmark runner (resumable).

  python run_eval.py --n 3                      # quick trial: 3 questions x 3 configs
  python run_eval.py                            # all 20 questions (leave it running)
  python run_eval.py --provider groq            # much faster on free cloud
  python run_eval.py --summarize                # rebuild eval_results/results.md only

Configs:  baseline    = one LLM call, no retrieval (the "just ask the model" baseline)
          no_revision = full agent team but NO critic revision loop (ablation)
          full        = planner + searcher + reader + writer + critic loop
"""
import argparse
import copy
import json
import time
from pathlib import Path

from citemesh.config import ROOT, load_config
from citemesh.llm import LLM
from citemesh.metrics import evaluate_report, mean
from citemesh.report import save_report
from citemesh.runner import run_research, setup_logging
from citemesh.tools import default_toolbox

OUT = ROOT / "eval_results"
CONFIGS = {"baseline": {}, "no_revision": {"max_iterations": 1, "llm_critic": False}, "full": {}}
BASELINE_PROMPT = ("You are a research assistant. Answer the question with a structured Markdown report: "
                   "'# title', '## Summary', one '## ' section per aspect, '## Limitations'. Cite sources inline.")


def run_one(name: str, q: dict, base_cfg: dict, tools, llm_factory=LLM) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["research"].update(CONFIGS[name])
    folder = str(OUT / "reports" / name)
    cfg["research"]["output_dir"] = folder
    llm = llm_factory(cfg)
    t0 = time.time()
    if name == "baseline":
        text = llm.chat(BASELINE_PROMPT, q["question"])
        save_report(text, q["question"], folder)
        metrics, refused, n_src, n_ev = evaluate_report(text, [], []), False, 0, 0
    else:
        state = run_research(q["question"], cfg, llm=llm, tools=tools)
        refused = not state["evidence"]
        metrics = evaluate_report(state["draft"], state["evidence"], state["sources"])
        n_src, n_ev = len(state["sources"]), len(state["evidence"])
    return {"config": name, "id": q["id"], "category": q["category"], "question": q["question"],
            "expect_refusal": q["expect_refusal"], "refused": refused, "seconds": round(time.time() - t0, 1),
            "tokens": llm.tokens, "llm_calls": llm.calls, "sources": n_src, "evidence": n_ev, **metrics}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def summarize(rows: list[dict]) -> str:
    rows = [r for r in rows if "error" not in r]
    lines = ["| Config | Runs | Cited-claim rate | Claim support (lexical) | Citation validity | Evidence used | Distinct domains | Avg sec | Avg tokens |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in CONFIGS:
        rs = [r for r in rows if r["config"] == name and not r["expect_refusal"] and not r["refused"]]
        if not rs:
            continue
        lines.append(f"| {name} | {len(rs)} | {pct(mean(r['cited_claim_rate'] for r in rs))} | "
                     f"{pct(mean(r['claim_support_rate'] for r in rs))} | {pct(mean(r['citation_validity'] for r in rs))} | "
                     f"{pct(mean(r['evidence_used'] for r in rs))} | {mean(r['cited_domains'] for r in rs):.1f} | "
                     f"{mean(r['seconds'] for r in rs):.0f} | {mean(r['tokens'] for r in rs):.0f} |")
    lines += ["", "_Quality rows use answerable questions where the system did not refuse. Baseline has no evidence list, so its claims are untraceable by construction._", "",
              "| Config | Correct refusals (unanswerable Qs) | False refusals (answerable Qs) |", "|---|---|---|"]
    for name in ("no_revision", "full"):
        un = [r for r in rows if r["config"] == name and r["expect_refusal"]]
        an = [r for r in rows if r["config"] == name and not r["expect_refusal"]]
        if un or an:
            lines.append(f"| {name} | {sum(r['refused'] for r in un)}/{len(un)} | {sum(r['refused'] for r in an)}/{len(an)} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="only the first N questions")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--provider", choices=["ollama", "groq", "gemini"])
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    runs = OUT / "runs.jsonl"
    if not args.summarize:
        cfg = load_config()
        if args.provider:
            cfg["provider"] = args.provider
        setup_logging(cfg)
        tools = default_toolbox(cfg)
        questions = json.loads((ROOT / "eval" / "questions.json").read_text())[: args.n]
        done = {(r["config"], r["id"]) for r in load_rows(runs) if "error" not in r}
        for q in questions:
            for name in args.configs.split(","):
                if (name, q["id"]) in done:
                    continue
                print(f"[{q['id']}] {name} ...", flush=True)
                try:
                    row = run_one(name, q, cfg, tools)
                    print(f"      {row['seconds']}s, claims={row['claims']}, cited={pct(row['cited_claim_rate'])}, refused={row['refused']}")
                except Exception as e:  # keep an overnight run alive
                    row = {"config": name, "id": q["id"], "error": str(e)}
                    print(f"      ERROR: {e}")
                with open(runs, "a") as f:
                    f.write(json.dumps(row) + "\n")
    table = summarize(load_rows(runs))
    (OUT / "results.md").write_text(table + "\n")
    print("\n" + table + f"\n\nSaved: {OUT / 'results.md'}")


if __name__ == "__main__":
    main()
