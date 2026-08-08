# RAG Eval-Audit

An agentic retrieval-augmented generation system over public banking-regulatory documents —
and an evaluation layer that audits both the system **and the judge grading it**.

The system works. The interesting part is what happened when I measured how well.

---

## The headline finding

Asked *"Are community banks required to perform annual model validation?"*, the system
**cited rescinded 2011 guidance as current authority on 5 of 6 runs.**

Same question, same corpus, same code. The correct answer lives in an OCC bulletin issued in
April 2026; the superseded 2011 guidance says something that reads like the opposite. On five of
six runs the retriever never surfaced the current document at all, and the agent answered
confidently from the rescinded one.

**Groundedness — the metric most RAG evaluations report as their headline number — scored 5/5 on
every one of those failures.** And it was right to. Every claim in the answer *was* supported by
the context it was given. The context was simply the wrong document.

Groundedness asks *"did the answer follow from the evidence?"* It never asks *"was that the right
evidence?"* Reference-free evaluation is blind to this failure class by construction.

**A single-run evaluation had a 1-in-6 chance of reporting this system as passing.**

---

## Why the failure happens

Not model reasoning — retrieval.

The decisive document is **7 chunks out of 288, 2.4% of the corpus**, and it is the only source
containing the sentence that settles the question. Retrieval surfaced it on 1 run in 6.

The chain is: retrieval failure → citation failure → compliance error, with a perfect groundedness
score at every step.

---

## What's here

| Component | File | What it does |
|---|---|---|
| Chunker | `code/chunk.py` | 200-word windows, 50-word overlap → 288 chunks |
| Retrieval | `code/retrieve.py` | TF-IDF, brute-force cosine, optional score threshold |
| Agent | `code/agent.py` | Claude with retrieval exposed as a tool — writes its own queries, decides when to stop |
| Eval runner | `code/run_eval.py` | Runs the agent over the golden set, saves full traces |
| Judge | `code/judge.py` | Scores groundedness (reference-free) and correctness; deterministic citation cross-check |
| Negative control | `code/negative_control.py` | Deliberately corrupts answers to test whether the judge detects corruption |
| Analysis | `code/analyze.py` | Aggregates by category — answerable and hard-negative never pooled |

**This is an agentic workflow, not a fixed pipeline.** The model authors its own search queries,
decides whether one search was enough, and decides when to stop. Control flow is determined at
run time by the model, not written in advance. It is *not* a service — autonomy is within a
single run.

---

## The corpus is a test bed, not a pile

Six public documents from US federal banking regulators, chosen to create four deliberate
test structures:

| Document | Role |
|---|---|
| `occ_2026_13` + `sr_26_2` | Same 2026 guidance from two agencies → **near-duplicate pair** |
| `sr_11_7_superseded` | 2011 guidance, replaced April 2026 → **staleness test** |
| `sr_23_4` + `third_party_risk_2023` | Same third-party guidance, one carrying ~7,000 words of comment-response preamble → **near-duplicate with asymmetric noise** |
| `cfpb_circular_2023_03` | Adverse action notices and AI credit models → **topically distinct** |

Corpus design *is* eval design. The near-duplicates test whether the agent cites the right source;
the superseded document tests whether it can distinguish guidance in force from guidance rescinded.

---

## Other findings

**No retrieval threshold separates answerable from unanswerable questions.** A question the corpus
cannot answer scored 0.2245; a question it can answer scored 0.1528. Excluding the former requires
a threshold that discards the latter. **No cutoff exists** — not "none was found." Relevance and
answerability are different quantities, and scoring the first does not get you the second.

**The judge penalises invented facts more than invented sources** — 5→2 for fabricated content,
5→3/4 for a fabricated citation, consistently across five questions. That is the worst possible
sensitivity profile for a compliance tool, where provenance *is* the product.

**Judge self-consistency depends on what it is asked to detect.** Across 40 judgments of
uncorrupted answers, the judge never disagreed with itself once. But on *subtly* corrupted
answers — ones that contradict their own evidence — it disagreed on all five questions, and twice
scored a deliberately contradictory answer a perfect 5. **Judge reliability is not a single
number**; measure it on easy cases and you will report a reliability you do not have on hard ones.

**Neither citation check catches fabricated provenance.** The deterministic regex is a *presence*
test — it confirms an acceptable source appears, never that no fabricated source does — so a
corrupted answer that adds a bogus citation alongside real ones passes it outright. The judge
deducts a single point. Together they still miss it.

**My own ground truth had a defect.** One question was labelled unanswerable because a keyword
search returned no matches — but the *concept* was in the corpus, and the system's answer was
defensible. Lexical absence is not semantic absence. If hand-built ground truth is unreliable,
any automated threshold claiming to separate answerable from unanswerable deserves suspicion.

Full detail and the numbers behind each of these are in [`NOTES.md`](NOTES.md).

---

## Scorecard

Six runs per question, judged independently:

| Question | Category | Result |
|---|---|---|
| Q1–Q3 | answerable | 6/6 correct |
| Q4 | answerable | 5/6 |
| **Q5** | staleness trap | **1/6 — 83% failure** |
| Q6, Q8 | hard negative | 6/6 correctly abstained |
| **Q7** | hard negative | **0/6 — never abstained** |

Four of eight questions showed run-to-run outcome variation.

---

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="..."      # never committed; .env is gitignored

python code/chunk.py                          # build the index
python code/agent.py "your question here"     # ask it something
python code/run_eval.py --runs 5              # run the golden set
python code/judge.py --judge-runs 1 data/eval_runs/*.json
python code/analyze.py                        # results by category
python code/negative_control.py data/eval_runs/q01_run01_*.json --runs 3
```

---

## Design decisions

**TF-IDF rather than dense embeddings.** Constrained by hardware, then kept on the merits: lexical
retrieval is the BM25 half of hybrid search and is *better* on exact identifiers — regulation
numbers, defined terms, section citations — which a regulatory corpus is full of. Known weakness:
paraphrases. Dense embeddings via an API are the natural upgrade.

**Brute-force cosine, no vector database.** At 288 chunks a single sparse matrix-vector product is
sub-millisecond *and* exact. An approximate index would trade correctness for speed that isn't
needed.

**No orchestration framework.** The abstraction layer would obscure the thing being measured.

**Temperature left at default.** Counterintuitive for anything auditable, but suppressing
run-to-run variance would suppress the quantity under study.

**The judge never sees the expected answer when scoring groundedness.** Otherwise it grades
correctness and calls it groundedness, and the distinction between "reasoned properly from bad
evidence" and "got lucky" is lost.

**Nothing is averaged at write time.** Every individual judge score is saved separately. The
variation *between* runs is the measurement.

---

## Limitations

- **Self-preference bias is not controlled.** The same model generates and judges. A `--judge-model`
  flag exists to estimate the effect; not yet run.
- **n = 8 golden set.** Adequate to demonstrate method, not to establish rates precisely.
- **Judge variability measured at default temperature only.**
- **One hard-negative label is contestable** (see above).
- **Lexical retrieval only** — paraphrase failures are expected and unmeasured.
- **Chunking parameters untuned** — 200/50 is a standard default, not an optimum.

---

## Proposed next step

**Status-aware retrieval.** The retriever currently ranks a rescinded document and its replacement
purely on lexical similarity. Tag each document with `status` and `effective_date`; filter or
down-weight superseded sources; require the agent to flag when it uses one.

This is a metadata problem, not a model problem — no amount of prompt engineering fixes a
retriever with no concept of which document is in force.

Any such fix would be validated with this same harness. That is the point of building it: the
evaluation is what distinguishes *"the fix worked"* from *"the fix worked on the one run I
happened to look at."*

---

## What this cost

Roughly **250 API calls** to evaluate eight questions properly. A single-run evaluation would have
been **eight** — about a 30× multiplier for rigour, at a total of around $5.

That is not a complaint; it is the actual economics, and it explains why the standard practice is
to run once and report the number. What the 30× bought was a compliance failure occurring on 83%
of runs that a single run would have had a 1-in-6 chance of calling a pass.

---

*Corpus documents are US federal government works and are in the public domain.*
