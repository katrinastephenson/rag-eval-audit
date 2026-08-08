"""
run_eval.py — run the agent against every question in the golden set.

For each (question, run) pair, save one JSON to data/eval_runs/ capturing
what the answer was, what the agent saw in its context, and how the loop
terminated. Downstream scripts read these files.

Run:  python code/run_eval.py [--runs N]
"""

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

from agent import run_agent

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_PATH = DATA_DIR / "golden_set.json"
EVAL_RUNS_DIR = DATA_DIR / "eval_runs"

# Matches (source.txt) or (source.txt, index=N) — case-insensitive on the ext.
_CITE_RE = re.compile(r"\(([A-Za-z0-9_]+\.txt)\b", re.IGNORECASE)


def _sources_cited(answer_text):
    """Extract source filenames the agent cited in its final answer."""
    return sorted({m.lower() for m in _CITE_RE.findall(answer_text)})


def _chunks_in_context(trace):
    """Union of every chunk returned to the agent across all rounds.

    Dedupe by chunk_id — the same chunk can surface from multiple queries,
    but the agent only sees it once (conceptually) for groundedness.
    """
    seen = {}
    for step in trace["steps"]:
        for tu in step["tool_uses"]:
            for r in tu["results"]:
                seen.setdefault(r["chunk_id"], r)
    return list(seen.values())


def run_one(item, run_no):
    trace = run_agent(item["question"])
    result = {
        "question_id": item["id"],
        "question": item["question"],
        "category": item["category"],
        "run_number": run_no,
        "final_answer": trace["final_answer"],
        "sources_cited": _sources_cited(trace["final_answer"]),
        "chunks_in_context": _chunks_in_context(trace),
        "rounds_used": trace["rounds_used"],
        "stop_reason": trace["stop_reason"],
        "hit_round_cap": trace["hit_round_cap"],
        "total_usage": trace["total_usage"],
    }
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = EVAL_RUNS_DIR / f"q{item['id']:02d}_run{run_no:02d}_{ts}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of times to run each question (default 1).")
    args = parser.parse_args()

    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    n_ok = 0
    n_fail = 0
    # Count by category so answerable vs hard-negative outcomes stay
    # separable when reading the console output.
    by_cat_ok = {}
    by_cat_fail = {}
    for item in golden:
        for run_no in range(1, args.runs + 1):
            preview = item["question"][:60]
            print(f"[q{item['id']:02d} {item['category']:14s} "
                  f"run {run_no}/{args.runs}] {preview}...",
                  flush=True)
            try:
                path = run_one(item, run_no)
                print(f"  -> {path.name}", flush=True)
                n_ok += 1
                by_cat_ok[item["category"]] = \
                    by_cat_ok.get(item["category"], 0) + 1
            except Exception as e:
                print(f"  ! FAILED: {e}", file=sys.stderr)
                traceback.print_exc()
                n_fail += 1
                by_cat_fail[item["category"]] = \
                    by_cat_fail.get(item["category"], 0) + 1

    print(f"\ndone: {n_ok} ok, {n_fail} failed")
    for cat in sorted(set(by_cat_ok) | set(by_cat_fail)):
        print(f"  {cat:14s}  ok={by_cat_ok.get(cat, 0)}  "
              f"failed={by_cat_fail.get(cat, 0)}")
