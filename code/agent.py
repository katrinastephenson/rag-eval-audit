"""
agent.py — Claude Sonnet 4.6 as a RAG agent over the chunked corpus.

The model is given one tool (search_documents) and decides for itself
what queries to issue and when it has enough context to answer.
We cap the loop at MAX_ROUNDS tool-use turns and record whether the
model stopped naturally or hit the cap.

Why not set temperature=0:
  We need run-to-run variance to be measurable — that's the whole
  point of the eval. Leaving the default preserves the sampling
  behavior the model was trained on.

Why not prompt-cache:
  Each run is a fresh question and a short conversation. There's no
  reusable prefix worth the write premium.

Run:  python code/agent.py "your question here"
      python code/agent.py                # interactive; 'quit' to exit
Out:  data/traces/<timestamp>_<slug>.json
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from retrieve import search_documents

MODEL = "claude-sonnet-4-6"
MAX_ROUNDS = 6
MAX_TOKENS = 4096

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRACES_DIR = DATA_DIR / "traces"

SYSTEM_PROMPT = """You are a research assistant answering questions about US \
bank and consumer-finance regulatory guidance.

Answer only from context returned by the search_documents tool; do not rely \
on prior knowledge. You may call the tool multiple times with different \
queries before answering.

Write plain prose under 200 words. No markdown headers, no bullet lists, no \
tables, no emoji. Cite inline as (source, index) after each claim.

If the retrieved context does not support an answer, say so directly. Do \
not hedge, guess, or pad with generalities."""

TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search the corpus of regulatory documents using TF-IDF over "
            "~285 pre-chunked passages. Returns the top-k most similar "
            "chunks with their source filename, position within that "
            "document (index), unique chunk_id, cosine similarity score, "
            "and text. Use specific keyword-rich queries; the index is "
            "lexical, not semantic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }
]


def _format_results_for_model(results):
    """Render tool results as text the model can read and cite from."""
    if not results:
        return "No results returned."
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[{i}] source={r['source']}  index={r['index']}  "
            f"chunk_id={r['chunk_id']}  score={r['score']:.4f}\n"
            f"{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def run_agent(question, max_rounds=MAX_ROUNDS):
    """Run the agent loop and return the full trace dict."""
    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": question}]
    steps = []
    total_input = 0
    total_output = 0
    stop_reason = "max_rounds"  # overwritten if the model ends naturally

    for round_idx in range(1, max_rounds + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        total_input += response.usage.input_tokens
        total_output += response.usage.output_tokens

        assistant_text_parts = []
        tool_uses_in_round = []
        for block in response.content:
            if block.type == "text":
                assistant_text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses_in_round.append(block)

        # Execute every tool use in this round and collect results.
        tool_results_for_model = []
        tool_uses_for_trace = []
        for tu in tool_uses_in_round:
            query = tu.input.get("query", "")
            k = tu.input.get("k", 5)
            results = search_documents(query, k=k)

            tool_uses_for_trace.append({
                "id": tu.id,
                "query": query,
                "k": k,
                "results": [
                    {
                        "source": r["source"],
                        "index": r["index"],
                        "chunk_id": r["chunk_id"],
                        "score": r["score"],
                        "text": r["text"],
                    }
                    for r in results
                ],
            })
            tool_results_for_model.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": _format_results_for_model(results),
            })

        steps.append({
            "round": round_idx,
            "assistant_text": "\n".join(assistant_text_parts),
            "tool_uses": tool_uses_for_trace,
            "response_stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        })

        # If the model didn't ask for a tool, it's done.
        if response.stop_reason != "tool_use":
            stop_reason = response.stop_reason  # typically "end_turn"
            break

        # Otherwise feed the tool results back and loop.
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results_for_model})

    # Final answer is the assistant text from the last step.
    final_answer = steps[-1]["assistant_text"] if steps else ""

    return {
        "question": question,
        "model": MODEL,
        "max_rounds": max_rounds,
        "rounds_used": len(steps),
        "stop_reason": stop_reason,          # "end_turn" | "max_rounds" | other
        "hit_round_cap": stop_reason == "max_rounds",
        "total_usage": {
            "input_tokens": total_input,
            "output_tokens": total_output,
        },
        "final_answer": final_answer,
        "steps": steps,
    }


def _write_trace(trace):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = re.sub(r"[^a-z0-9]+", "_", trace["question"].lower()).strip("_")[:50]
    path = TRACES_DIR / f"{ts}_{slug}.json"
    path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return path


def _print_trace(trace):
    print("=" * 78)
    print(f"QUESTION: {trace['question']}")
    print(f"model={trace['model']}  rounds={trace['rounds_used']}/"
          f"{trace['max_rounds']}  stop_reason={trace['stop_reason']}")
    print(f"tokens: input={trace['total_usage']['input_tokens']}  "
          f"output={trace['total_usage']['output_tokens']}")
    print("=" * 78)

    for step in trace["steps"]:
        print(f"\n--- Round {step['round']} "
              f"(stop_reason={step['response_stop_reason']}) ---")
        if step["assistant_text"]:
            print(f"\n[assistant]\n{step['assistant_text']}")
        for tu in step["tool_uses"]:
            print(f"\n[tool_use] search_documents(query={tu['query']!r}, "
                  f"k={tu['k']})")
            for i, r in enumerate(tu["results"], start=1):
                preview = " ".join(r["text"].split()[:25])
                print(f"    [{i}] score={r['score']:.4f}  "
                      f"{r['source']}  index={r['index']}  "
                      f"chunk_id={r['chunk_id']}")
                print(f"        {preview}...")

    if trace["hit_round_cap"]:
        print(f"\n[!] Hit the {trace['max_rounds']}-round cap without a "
              f"natural end_turn.")


# Matches (source.txt) or (source.txt, index=N). Same shape used by
# run_eval.py — duplicated here to avoid a circular import.
_CITE_RE = re.compile(r"\(([A-Za-z0-9_]+\.txt)\b", re.IGNORECASE)


def _sources_cited(answer_text):
    return sorted({m.lower() for m in _CITE_RE.findall(answer_text)})


def _interactive():
    """Read questions from stdin until the user types 'quit'."""
    print("Interactive mode. Type 'quit' (or Ctrl-D) to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() == "quit":
            break
        try:
            trace = run_agent(question)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            continue
        path = _write_trace(trace)
        print(f"\n{trace['final_answer']}")
        cited = _sources_cited(trace["final_answer"])
        print(f"\nsources cited: {cited if cited else '(none)'}")
        print(f"(rounds={trace['rounds_used']}/{trace['max_rounds']}  "
              f"stop_reason={trace['stop_reason']}  trace={path.name})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _interactive()
    else:
        question = " ".join(sys.argv[1:])
        trace = run_agent(question)
        path = _write_trace(trace)
        _print_trace(trace)
        print(f"\nwrote trace to {path}")
