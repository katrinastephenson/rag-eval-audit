"""
analyze.py — summarize every judgment in data/judgments/.

Groups judgments by category first, then by question. Within each
question, lists the individual scores across all judge runs (and
across all eval runs, if the question was answered multiple times).

Nothing is averaged. Nothing is pooled across categories. Where every
run produced the same score we say so explicitly instead of printing a
standard deviation of zero — a std of 0 across 3 runs is not the same
signal as a std of 0 across 30, and hiding the sample size behind a
summary statistic invites overconfidence.

Run:  python code/analyze.py
"""

from collections import defaultdict
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JUDGMENTS_DIR = DATA_DIR / "judgments"

# Print order matters — answerable first (should score high), then the
# staleness trap (specifically constructed to see which document wins),
# then hard-negatives (should abstain).
CATEGORY_ORDER = ["answerable", "staleness-trap", "hard-negative"]


def _load_judgments():
    """Load every judgment file, tolerant of missing new-schema fields."""
    records = []
    for path in sorted(JUDGMENTS_DIR.glob("*.json")):
        try:
            j = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[skip] {path.name}: {e}")
            continue
        # If the file was written before the flat-schema rework, skip
        # it rather than silently mis-report. Re-run judge.py to refresh.
        if "groundedness_score" not in j:
            print(f"[skip] {path.name}: pre-flat-schema, please re-run "
                  f"judge.py")
            continue
        records.append((path.name, j))
    return records


def _fmt_scores(values, label):
    """Zero-variance: say so explicitly. Otherwise show the raw list."""
    if not values:
        return f"{label:32s}  (no runs)"
    unique = set(values)
    if len(unique) == 1:
        return f"{label:32s}  all {len(values)} runs: {values[0]}"
    return (f"{label:32s}  {list(values)}  "
            f"(distinct values: {sorted(unique)})")


def _print_question(qid, question, category, records):
    # Records for one question — possibly across multiple eval runs
    # and multiple judge runs. Group by (eval_run_file, judge_model)
    # so the reader sees where the variance is coming from.
    eval_runs = sorted({r["eval_run_file"] for r in records})
    judge_models = sorted({r["judge_model"] for r in records})

    print(f"\n--- Q{qid}: {question[:70]}{'...' if len(question) > 70 else ''} ---")
    print(f"  eval_runs={len(eval_runs)}  "
          f"judge_models={judge_models}  "
          f"total_judgments={len(records)}")

    grounded = [r["groundedness_score"] for r in records]
    print("  " + _fmt_scores(grounded, "groundedness_score"))

    if category == "hard-negative":
        abstained = [r["abstained"] for r in records]
        print("  " + _fmt_scores(abstained, "abstained"))
        # citation_check_programmatic is meaningfully None here — no
        # expected sources to check.
        print(f"  {'citation_check_programmatic':32s}  "
              f"N/A (no expected sources)")
    else:
        correctness = [r["correctness_score"] for r in records]
        cites = [r["cites_expected_source"] for r in records]
        prog = [r["citation_check_programmatic"] for r in records]
        print("  " + _fmt_scores(correctness, "correctness_score"))
        print("  " + _fmt_scores(cites, "cites_expected_source (judge)"))
        print("  " + _fmt_scores(prog, "citation_check_programmatic"))


def main():
    if not JUDGMENTS_DIR.exists():
        print(f"no judgments directory at {JUDGMENTS_DIR}")
        return
    all_records = _load_judgments()
    if not all_records:
        print("no judgment records found")
        return

    # Group: category -> question_id -> [records]
    grouped = defaultdict(lambda: defaultdict(list))
    questions = {}
    for _name, j in all_records:
        grouped[j["category"]][j["question_id"]].append(j)
        questions[j["question_id"]] = j["question"]

    for cat in CATEGORY_ORDER:
        if cat not in grouped:
            continue
        qids = sorted(grouped[cat].keys())
        n_judgments = sum(len(grouped[cat][q]) for q in qids)
        print("=" * 78)
        print(f"{cat.upper()}  ({len(qids)} questions, "
              f"{n_judgments} total judgments)")
        print("=" * 78)
        for qid in qids:
            _print_question(qid, questions[qid], cat, grouped[cat][qid])

    # Warn if a category we don't expect appears — better than silently
    # dropping it.
    unexpected = set(grouped.keys()) - set(CATEGORY_ORDER)
    if unexpected:
        print("\n[!] Skipped unrecognized categories: "
              f"{sorted(unexpected)}")


if __name__ == "__main__":
    main()
