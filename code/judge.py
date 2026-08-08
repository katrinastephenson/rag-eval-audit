"""
judge.py — LLM-as-judge scoring for saved eval runs.

For each eval-run file, make a separate Claude call to score:

  Groundedness (1-5, all categories)
    Reference-free: is every claim in the answer supported by the
    retrieved context the agent saw? The judge does NOT see the
    expected answer for this axis — it only sees the retrieved chunks
    and the agent's answer.

  Correctness (1-5, answerable / staleness-trap only)
    Does the answer match expected_answer? The judge sees the expected
    answer and the acceptable expected_sources for citation checking.

  Abstained (bool, hard-negative only)
    Instead of correctness. Did the answer clearly refuse to answer,
    or did it provide a substantive (and therefore fabricated) answer?

Every judge call is saved individually to data/judgments/ as its own
JSON file. Nothing is averaged here — that's a downstream concern.

Each judgment records `category` (from the golden set) so any downstream
reporting can keep answerable, staleness-trap, and hard-negative
outcomes separated. Do not pool them into a single number — the
outcomes measured are different (correctness for the first two,
abstention for the third).

Run:  python code/judge.py <eval_run.json>... [--judge-runs K] \\
                                              [--judge-model MODEL]
"""

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import anthropic

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_PATH = DATA_DIR / "golden_set.json"
JUDGMENTS_DIR = DATA_DIR / "judgments"

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_MAX_TOKENS = 512

_CITE_RE = re.compile(r"\(([A-Za-z0-9_]+\.txt)\b", re.IGNORECASE)


def _load_golden():
    return {x["id"]: x for x in json.loads(
        GOLDEN_PATH.read_text(encoding="utf-8"))}


def _format_chunks(chunks):
    """Render the chunks the agent had in context, for the judge to see."""
    if not chunks:
        return "(no chunks were in context)"
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] source={c['source']}  index={c['index']}\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_json(text):
    """Parse a JSON object from a model response.

    output_config guarantees the first text block is valid JSON, but we
    still handle the (rare) case of extra whitespace or a preamble.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back: grab the first {...} balanced object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def _judge_call(client, prompt, schema, model):
    """One Claude call constrained to a JSON schema."""
    resp = client.messages.create(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {"type": "json_schema", "schema": schema},
        },
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return _extract_json(text), {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


GROUNDEDNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
        "cites_expected_source": {"type": "boolean"},
    },
    "required": ["score", "rationale", "cites_expected_source"],
    "additionalProperties": False,
}

ABSTAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "abstained": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["abstained", "rationale"],
    "additionalProperties": False,
}


def score_groundedness(client, answer, chunks, model):
    prompt = f"""You are scoring an AI answer for GROUNDEDNESS in a retrieved \
context.

Groundedness = every factual claim in the answer is directly supported by \
the retrieved context below. Ignore whether the answer is correct, \
complete, or well-written — only whether each claim can be traced to the \
context.

Scoring:
5 = every claim explicitly supported by the context
4 = mostly supported; at most one minor claim is a small inference beyond \
what the context states
3 = major claims supported, but the answer contains inferences or details \
not clearly in the context
2 = several claims are not supported or overreach the context
1 = most or all claims are not supported by the context (fabrication)

If the answer abstains ("the retrieved context does not support an \
answer") and the context genuinely doesn't support one, score 5 — an \
honest abstention is fully grounded.

=== RETRIEVED CONTEXT ===
{_format_chunks(chunks)}

=== ANSWER ===
{answer}

Return JSON: {{"score": <1-5>, "rationale": "<one sentence>"}}"""
    return _judge_call(client, prompt, GROUNDEDNESS_SCHEMA, model)


def score_correctness(client, question, expected_answer, expected_sources,
                      answer, model):
    sources_str = ", ".join(expected_sources) if expected_sources else "(none)"
    prompt = f"""You are scoring an AI answer for CORRECTNESS against a \
reference expected answer.

Compare factual content. Ignore differences in phrasing, formatting, or \
length. The AI answer may include more detail than the expected answer; \
that is fine as long as the core claim matches.

Scoring:
5 = fully matches the expected answer's factual content
4 = matches the main claim; a minor detail is missing or slightly off
3 = partial match; significant content missing or added
2 = misleading, off-topic, or largely wrong
1 = contradicts the expected answer

Also record whether the answer cites at least one of the ACCEPTABLE \
SOURCES. Any single acceptable source is enough — citations look like \
"(filename.txt, index=N)" or similar.

QUESTION: {question}
EXPECTED ANSWER: {expected_answer}
ACCEPTABLE SOURCES: {sources_str}

=== ANSWER ===
{answer}

Return JSON: {{"score": <1-5>, "rationale": "<one sentence>", \
"cites_expected_source": <true|false>}}"""
    return _judge_call(client, prompt, CORRECTNESS_SCHEMA, model)


def score_abstain(client, question, answer, model):
    prompt = f"""The following question CANNOT be answered from the corpus \
the AI has access to. Decide whether the AI answer abstained — clearly \
stating it cannot answer from the retrieved context — versus providing a \
substantive answer (which would be a fabrication).

An abstention says something like "the retrieved context does not address \
this" or "I don't have information to answer this." Hedged partial \
answers that still make substantive claims do NOT count as abstaining.

QUESTION: {question}

=== ANSWER ===
{answer}

Return JSON: {{"abstained": <true|false>, "rationale": "<one sentence>"}}"""
    return _judge_call(client, prompt, ABSTAIN_SCHEMA, model)


def _programmatic_cite_check(answer, expected_sources):
    """Check citations locally — cheaper and more reliable than the judge."""
    if not expected_sources:
        return None
    cited = {m.lower() for m in _CITE_RE.findall(answer)}
    expected = {s.lower() for s in expected_sources}
    return bool(cited & expected)


def judge_one(client, eval_run_path, golden_by_id, judge_run_no, model):
    """Score one eval run once and write a judgment JSON.

    Schema is deliberately FLAT — every score, flag, and rationale is a
    top-level field. Fields that don't apply to this category are null.
    This keeps downstream analysis code simple (no nested dict traversal).
    """
    run = json.loads(eval_run_path.read_text(encoding="utf-8"))
    item = golden_by_id[run["question_id"]]

    grounded, g_usage = score_groundedness(
        client, run["final_answer"], run["chunks_in_context"], model)

    judgment = {
        "eval_run_file": eval_run_path.name,
        "question_id": run["question_id"],
        "question": run["question"],
        "category": item["category"],
        "judge_run_number": judge_run_no,
        "judge_model": model,
        # Groundedness applies to every category.
        "groundedness_score": grounded["score"],
        "groundedness_rationale": grounded["rationale"],
        # Correctness applies to answerable + staleness-trap.
        "correctness_score": None,
        "correctness_rationale": None,
        "cites_expected_source": None,
        # Abstain applies to hard-negative only.
        "abstained": None,
        "abstain_rationale": None,
        # Programmatic citation check: None for hard-negatives (no
        # expected sources to check against).
        "citation_check_programmatic": _programmatic_cite_check(
            run["final_answer"], item["expected_sources"]),
        "usage": {"groundedness": g_usage},
    }

    if item["category"] == "hard-negative":
        abstain, a_usage = score_abstain(
            client, run["question"], run["final_answer"], model)
        judgment["abstained"] = abstain["abstained"]
        judgment["abstain_rationale"] = abstain["rationale"]
        judgment["usage"]["abstain"] = a_usage
    else:
        correct, c_usage = score_correctness(
            client, run["question"], item["expected_answer"],
            item["expected_sources"], run["final_answer"], model)
        judgment["correctness_score"] = correct["score"]
        judgment["correctness_rationale"] = correct["rationale"]
        judgment["cites_expected_source"] = correct["cites_expected_source"]
        judgment["usage"]["correctness"] = c_usage

    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_slug = model.replace(".", "_")
    out_path = JUDGMENTS_DIR / (
        f"{eval_run_path.stem}_{model_slug}_judge{judge_run_no:02d}_"
        f"{ts}.json")
    out_path.write_text(json.dumps(judgment, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_run", nargs="+", type=Path,
                        help="One or more eval-run JSON files to score.")
    parser.add_argument("--judge-runs", type=int, default=1,
                        help="Score each eval run K times, saving each score "
                             "separately (default 1).")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help="Claude model to use as judge (default "
                             f"{DEFAULT_JUDGE_MODEL}). Use a different "
                             "model to estimate self-preference bias.")
    args = parser.parse_args()

    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    golden_by_id = _load_golden()
    client = anthropic.Anthropic()

    n_ok = 0
    n_fail = 0
    for run_path in args.eval_run:
        for k in range(1, args.judge_runs + 1):
            print(f"[{run_path.name}  judge {k}/{args.judge_runs}  "
                  f"model={args.judge_model}]", flush=True)
            try:
                out = judge_one(client, run_path, golden_by_id, k,
                                args.judge_model)
                print(f"  -> {out.name}", flush=True)
                n_ok += 1
            except Exception as e:
                print(f"  ! FAILED: {e}", file=sys.stderr)
                traceback.print_exc()
                n_fail += 1

    print(f"\ndone: {n_ok} ok, {n_fail} failed")
