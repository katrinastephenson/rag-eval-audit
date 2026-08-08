"""
negative_control.py — sanity check for the judge.

Take a real eval-run answer and construct three deliberately corrupted
versions of it, then judge each version (plus the original) N times and
print the scores side-by-side. If the judge can't distinguish the
corrupted answers from the original, the judge is broken and no
absolute score coming out of it can be trusted.

The three corruptions target three different failure modes:

  1. fabricated_claim         — appends a specific-sounding invented
                                fact, citing a source that WAS
                                retrieved (fools naive citation checks).
                                Groundedness should drop.

  2. cites_unretrieved_source — cites a source filename that was never
                                returned by the retriever. Groundedness
                                should drop; the programmatic citation
                                check may still pass if the fake cite
                                happens to be an expected source.

  3. contradicts_context      — appends an assertion that the retrieved
                                context establishes the OPPOSITE
                                conclusion. Groundedness AND correctness
                                (or abstention) should drop.

Run:  python code/negative_control.py <eval_run.json> [--runs N] \\
                                                      [--judge-model MODEL]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import anthropic

from judge import (
    DEFAULT_JUDGE_MODEL,
    _programmatic_cite_check,
    score_abstain,
    score_correctness,
    score_groundedness,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_PATH = DATA_DIR / "golden_set.json"
NEG_CONTROL_DIR = DATA_DIR / "negative_control"


def _load_golden_by_id():
    return {x["id"]: x for x in json.loads(
        GOLDEN_PATH.read_text(encoding="utf-8"))}


def _all_source_filenames():
    return sorted(p.name for p in DATA_DIR.glob("*.txt"))


def make_fabricated_claim(original, chunks):
    """Append an invented fact, citing a source that WAS retrieved.

    Fools superficial citation checks — the source exists in context,
    but the specific claim is nowhere in it. This is the failure mode
    that matters most: a plausible-sounding fabrication with a real
    citation.
    """
    real_source = chunks[0]["source"] if chunks else "unknown.txt"
    real_index = chunks[0]["index"] if chunks else 0
    fabricated = (
        "Additionally, covered institutions must conduct quarterly "
        "algorithmic bias audits and file annual attestations with the "
        "Federal Reserve Board, with civil penalties starting at "
        f"$250,000 per violation ({real_source}, index={real_index})."
    )
    return original + "\n\n" + fabricated


def make_wrong_source_cite(original, chunks):
    """Cite a source filename that was NEVER returned by the retriever."""
    retrieved = {c["source"] for c in chunks}
    unused = [s for s in _all_source_filenames() if s not in retrieved]
    fake_source = unused[0] if unused else "unretrieved_source.txt"
    appendix = (
        f"This requirement is further reinforced by parallel guidance in "
        f"({fake_source}, index=5), which extends the same principles to "
        f"related supervisory contexts."
    )
    return original + "\n\n" + appendix


def make_contradicts_context(original, chunks):
    """Assert the retrieved context establishes the opposite conclusion.

    Deliberately generic — a downstream reader (or judge) does not need
    to know the specific claim to see the contradiction, because the
    corruption asserts its own reversal.
    """
    contradiction = (
        "Correction on the above: on further review of the retrieved "
        "context, the opposite conclusion is actually the case. The "
        "cited passages in fact establish the reverse of what is stated "
        "above; the answer above should be inverted."
    )
    return original + "\n\n" + contradiction


CORRUPTIONS = [
    ("fabricated_claim", make_fabricated_claim),
    ("cites_unretrieved_source", make_wrong_source_cite),
    ("contradicts_context", make_contradicts_context),
]


def judge_variant(client, question, answer, chunks, item, model):
    """One judge pass — returns a flat dict of scores/flags."""
    grounded, _ = score_groundedness(client, answer, chunks, model)
    record = {
        "groundedness_score": grounded["score"],
        "groundedness_rationale": grounded["rationale"],
    }
    if item["category"] == "hard-negative":
        abstain, _ = score_abstain(client, question, answer, model)
        record["abstained"] = abstain["abstained"]
        record["abstain_rationale"] = abstain["rationale"]
    else:
        correct, _ = score_correctness(
            client, question, item["expected_answer"],
            item["expected_sources"], answer, model)
        record["correctness_score"] = correct["score"]
        record["correctness_rationale"] = correct["rationale"]
        record["cites_expected_source"] = correct["cites_expected_source"]
    record["citation_check_programmatic"] = _programmatic_cite_check(
        answer, item["expected_sources"])
    return record


def _fmt_list(values):
    """List values, with an 'all N: X' shortcut when they're identical."""
    if not values:
        return "(no runs)"
    if len(set(values)) == 1:
        return f"all {len(values)}: {values[0]}"
    return str(list(values))


def print_comparison(question, category, all_scores):
    print("=" * 78)
    print(f"QUESTION: {question}")
    print(f"category: {category}")
    print("=" * 78)

    variants = list(all_scores.keys())
    axes = ["groundedness_score"]
    if category == "hard-negative":
        axes.append("abstained")
    else:
        axes.append("correctness_score")
        axes.append("cites_expected_source")
    axes.append("citation_check_programmatic")

    for variant in variants:
        runs = all_scores[variant]
        print(f"\n--- {variant} ({len(runs)} judge runs) ---")
        for axis in axes:
            values = [r.get(axis) for r in runs]
            print(f"  {axis:32s}  {_fmt_list(values)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_run", type=Path,
                        help="An eval-run JSON file to corrupt and re-judge.")
    parser.add_argument("--runs", type=int, default=3,
                        help="Judge each variant this many times (default 3).")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model (default {DEFAULT_JUDGE_MODEL}).")
    args = parser.parse_args()

    run = json.loads(args.eval_run.read_text(encoding="utf-8"))
    item = _load_golden_by_id()[run["question_id"]]
    chunks = run["chunks_in_context"]

    variants = {"original": run["final_answer"]}
    for name, fn in CORRUPTIONS:
        variants[name] = fn(run["final_answer"], chunks)

    client = anthropic.Anthropic()
    NEG_CONTROL_DIR.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    for name, answer in variants.items():
        print(f"[judging {name} x {args.runs} "
              f"(model={args.judge_model})]", flush=True)
        variant_records = []
        for k in range(1, args.runs + 1):
            scores = judge_variant(
                client, run["question"], answer, chunks, item,
                args.judge_model)
            variant_records.append(scores)

            record = {
                "eval_run_file": args.eval_run.name,
                "question_id": run["question_id"],
                "question": run["question"],
                "category": item["category"],
                "variant": name,
                "judge_run_number": k,
                "judge_model": args.judge_model,
                "answer": answer,
                **scores,
            }
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            model_slug = args.judge_model.replace(".", "_")
            out_path = NEG_CONTROL_DIR / (
                f"{args.eval_run.stem}_{name}_{model_slug}"
                f"_judge{k:02d}_{ts}.json")
            out_path.write_text(json.dumps(record, indent=2),
                                encoding="utf-8")
            print(f"  -> {out_path.name}", flush=True)
        all_scores[name] = variant_records

    print()
    print_comparison(run["question"], item["category"], all_scores)
