# RAG Eval-Audit — Running Notes

Working notes for the README and any later write-up.
A study of evaluation reliability in agentic retrieval-augmented generation over
banking-regulatory documents. Built 2026-08-08.

**One-line description:** an agentic RAG system over public banking-regulatory documents,
plus an evaluation layer that audits both the system *and the judge grading it*.

---

## 1. What it is

| # | Component | What it does |
|---|---|---|
| 1 | **Corpus** | 6 public banking-regulatory documents, ~43,550 words |
| 2 | **Chunker** (`chunk.py`) | 200-word windows, 50-word overlap → 288 chunks |
| 3 | **Retrieval** (`retrieve.py`) | TF-IDF (1–2 grams, English stopwords), brute-force cosine, top-*k* |
| 4 | **Agent** (`agent.py`) | Claude with `search_documents` exposed as a tool; writes its own queries, decides how many times to search, when to stop. 6-round cap |
| 5 | **Eval runner** (`run_eval.py`) | Runs the agent over the golden set, saves full traces |
| 6 | **Judge** (`judge.py`) | Separate Claude call scoring groundedness (reference-free) and correctness; programmatic citation cross-check |
| 7 | **Negative control** (`negative_control.py`) | Corrupts answers deliberately to test whether the judge can detect corruption |
| 8 | **Analysis** (`analyze.py`) | Aggregates judgments by category, never pooled |

**Not an agent that runs unattended.** The autonomy is *within* a single run: the model
chooses its own search queries and its own stopping point. It is not a background service.

---

## 2. Corpus design

Six documents, chosen to create four deliberate test structures rather than as a random pile:

| Document | Words | Role in the test bed |
|---|---|---|
| `occ_2026_13.txt` | 1,082 | 2026 model risk guidance, OCC version |
| `sr_26_2.txt` | 2,982 | Same guidance, Fed version → **near-duplicate pair** |
| `sr_11_7_superseded.txt` | 9,875 | 2011 guidance, replaced April 2026 → **staleness test** |
| `sr_23_4.txt` | 9,785 | Third-party risk, Fed version |
| `third_party_risk_2023.txt` | 16,585 | Same guidance, Federal Register version with ~7,000 words of comment-response preamble → **second near-duplicate, asymmetric noise** |
| `cfpb_circular_2023_03.txt` | 3,245 | Adverse action notices + AI credit models → **topically distinct** |

Corpus design *is* eval design. The near-duplicates test whether the agent cites the right
source; the superseded document tests whether it can tell current guidance from rescinded
guidance — the canonical failure mode for regulatory Q&A.

---

## 3. Design decisions (and why)

**TF-IDF rather than dense embeddings.** Forced by hardware — PyTorch has published no macOS
x86_64 build since 2.2.2, and `transformers` requires ≥ 2.4, so `sentence-transformers` was
not installable. Reframed honestly: lexical retrieval is the BM25 half of hybrid search and is
*better* on exact identifiers — regulation numbers, defined terms, section citations — which a
regulatory corpus is full of. Known weakness: misses paraphrases. Dense embeddings via an API
(e.g. Voyage) are the natural upgrade.

**Brute-force cosine, no vector database.** At 288 chunks a single sparse matrix-vector product
is sub-millisecond *and* returns the exact top-*k*. An ANN index would trade exactness for speed
that isn't needed.

**No LangChain / LlamaIndex.** The orchestration layer would hide the thing being measured.

**200-word chunks, 50-word (25%) overlap.** Chunk size is a binning choice with the familiar
tradeoff: too coarse and the TF-IDF signal is diluted across unrelated text; too fine and a chunk
lacks the context to answer anything. These are untuned defaults — tuning them properly would
require the eval set that was built afterward.

**Temperature left at default, not 0.** Counterintuitive for anything auditable, but the whole
point is to measure run-to-run variance; temperature 0 would suppress the quantity of interest.
Scope note: what's measured is variability *at default temperature*.

**The judge does not see `expected_answer` when scoring groundedness.** Otherwise it grades
correctness and calls it groundedness, and the ability to distinguish "reasoned properly from bad
evidence" from "got lucky" is lost.

**Nothing is averaged at write time.** Every individual judge score is saved. The variation
*between* runs is the measurement.

**`expected_sources` = acceptable sources; citing any one counts as correct.** Forced by the
near-duplicate pairs — an answer citing the Fed version of guidance that also exists in the
Federal Register is correct, not a miss.

**Agent capped at 6 tool-use rounds**, and whether the cap was hit is recorded — hitting it means
the agent never became confident enough to answer, which is itself a signal.

---

## 4. Findings

### 4.1 Retrieval score cannot separate answerable from unanswerable

The headline result.

| Query | Category | Top score |
|---|---|---|
| What are the requirements for model validation? | on-topic | **0.3349** |
| What must a creditor disclose when using AI to deny credit? | on-topic | 0.2699 |
| How often must banks retrain machine learning models? | **hard negative** | **0.2245** |
| What does the guidance say about cryptocurrency custody? | **hard negative** | 0.1779 |
| How should banks manage third-party risk? | on-topic | **0.1528** |
| What are the capital requirements for community banks? | **hard negative** | 0.1453 |

**A question the corpus cannot answer outscores a question it can.**

Work the arithmetic and it becomes a proof rather than an observation: excluding the worst hard
negative requires a threshold above 0.2245, which also discards a legitimate on-topic question at
0.1528. Keeping every on-topic question requires a threshold at or below 0.1528, which admits all
three hard negatives. **No cutoff separates them** — not "none was found," but none exists.

**Why it matters:** the obvious engineering answer to RAG hallucination is "set a relevance
threshold and refuse below it." This is measured evidence, from this corpus, that retrieval score
is not a sufficient abstention signal. Relevance and answerability are different quantities.

**Why the hard negatives scored high:** the capital-requirements query returned chunks discussing
community banks at length that say nothing about capital requirements. Topically adjacent,
lexically matching, semantically useless — the hallucination mechanism, visible.

**Honest caveat:** part of this overlap is TF-IDF's lexical matching, and dense embeddings would
reduce it. But not eliminate it — any retriever returns its top-*k*, and relevance ranking never
becomes an answerability test.

### 4.2 The eval metric hit a ceiling

All 40 groundedness scores and all 20 correctness scores came back **5**.

Zero variance. Which means the planned measurement **cannot be computed**: for
$\hat{R} = \sqrt{\hat{V}/W}$ the within-run variance $W$ is exactly zero, so the statistic is
$0/0$. The bootstrap interval on the mean is $[5,5]$ — zero width not from certainty but because
there is nothing to resample.

**A metric with no variance carries no information.** It returns the maximum whether the system is
good or bad.

### 4.3 A negative control showed the judge was *not* the problem

Deliberately corrupted versions of one answer, judged 3× each:

| Variant | Groundedness |
|---|---|
| Original | 5 |
| Cites a source that was never retrieved | 4 |
| Contradicts the retrieved context | 3 |
| Fabricates a claim absent from context | 2 |

The judge detects corruption and grades corruption types differently. **So the ceiling was the
test set being too easy, not a broken instrument** — all eight answers scored 5 because all eight
were genuinely good.

Validating the instrument before trusting the measurement is the discipline. It is the same move
as fitting `sklearn.LinearRegression` to check a from-scratch gradient descent, applied to an LLM
judge — and it is not standard practice in LLM evaluation.

### 4.4 Judge sensitivity profile *(n = 1 — replication pending)*

From 4.3: the judge is **most** sensitive to fabricated *content* (5→2) and **least** sensitive to
fabricated *provenance* (5→4). A citation pointing at a document that was never retrieved barely
moves the score.

That is the worst possible sensitivity profile for a compliance tool, where provenance *is* the
product — nobody at a bank asks "is this plausible," they ask "which document says that."

**Status: one question only.** Replication across q02–q05 is running. If the ordering holds across
five questions it is a measured property; if it scrambles it was noise, and the claim gets dropped.

### 4.5 Ground truth is harder to construct than it looks

Q07 — *"How often must banks retrain machine learning models?"* — was labelled a hard negative
because a keyword search for "retrain" returned zero matches. But the *concept* (ongoing monitoring
frequency, triggers for recalibration and redevelopment) is very much in the corpus, and the agent
produced a defensible, well-cited answer saying no fixed schedule is prescribed.

**Lexical absence is not semantic absence.** The labelling procedure was itself unreliable — which
is worth stating rather than fixing by relabelling. If ground truth built by hand has a known
defect, any automated threshold claiming to separate answerable from unanswerable deserves deep
suspicion.

### 4.6 Groundedness means different things in different categories

A 5 on an answerable item means "every claim is supported." A 5 on a hard negative means
"correctly refused" — an honest abstention is perfectly grounded.

**Therefore groundedness must never be averaged across categories.** A pooled score would look
excellent both for a system working well and for a system refusing everything. Report groundedness
for answerable items and abstention rate for hard negatives, separately, always. This is the same
trap as conflating coverage-against-truth with coverage-against-empirical-mean: one metric name
hiding two quantities.

### 4.7 Corpus hygiene is a retrieval problem in disguise

Both Federal Register documents carried a block of site metadata and help text ("Document Headings,"
"See the Document Drafting Handbook") that became **chunk 0** — a retrievable chunk of pure website
furniture sitting in the index, ready to match a query and crowd out real content.

Invisible in the raw files. Only surfaced by printing a chunk and reading it. Retrieval quality is
substantially a data-hygiene problem, and it presents as a model problem.

### 4.8 Trajectory variance *(observed informally — formal measurement pending)*

Same question, two runs:

| | Run 1 | Run 2 |
|---|---|---|
| Rounds | 3 | 2 |
| Searches | 3 | 2 |
| Second query | "generative AI models risk management" | "generative AI model risk" |

Different path, different step count, different self-authored queries — **same correct answer.**
The conclusion was stable; the route was not.

This only exists because the system is an agent rather than a fixed pipeline, and it matters
because per-step reliability $p$ over $n$ steps compounds toward $p^n$. Formal measurement across
5 repeats is pending.

### 4.9 Near-duplicates break exact-match source scoring

`sr_23_4.txt` and `third_party_risk_2023.txt` are the same guidance. Q4's `expected_sources`
initially listed only one, so an answer citing the other would have scored as a miss while being
substantively correct. Real corpora are full of near-duplicates; exact-match source scoring will
silently under-report accuracy on all of them.

---

## 5. Limitations to declare up front

- **Self-preference bias.** The same model (`claude-sonnet-4-6`) generates and judges. This is a
  documented LLM-as-judge failure mode and it is *not* controlled for. A `--judge-model` flag
  exists to estimate the effect by re-judging with a different model; not yet run.
- **n = 8 golden set.** Small. Adequate to demonstrate method, not to establish rates.
- **Judge variability measured at default temperature only.** A deployment at temperature 0 would
  see less.
- **One hard negative (q07) has contestable ground truth.** See 4.5.
- **Lexical retrieval only.** No dense embeddings; paraphrase failures are expected and unmeasured.
- **Untuned chunking parameters.** 200/50 chosen as a standard default, not optimised.

---

## 6. Incidents worth keeping

**Secrets handling.** One API key scoped to this project, held in an environment variable and
never in a file. `.gitignore` excludes `.env` and the virtualenv. Any key that is exposed by any
route is revoked and replaced rather than reused — rotation is cheap, and a leaked key in a public
repository is scraped within minutes.

**An agent deleted the experiment.** Claude Code cleared `data/eval_runs/` and `data/judgments/`
while testing a bug fix, destroying 8 eval runs and 40 judgments (~80¢ of API calls). Nobody asked
it to. It was doing something reasonable with broader permissions than the task required.

This is the clearest possible illustration of why destructive operations belong outside an agent's
autonomy boundary, and why file access should be scoped to what the task actually needs. The
mitigation is unglamorous: `git init` before the first expensive step, commit after each one.

---

## 7. Still to do

- [ ] Replicate the judge sensitivity profile across q02–q05 (4.4)
- [ ] Formal trajectory variance: `run_eval.py --runs 5`, compare paths and answers (4.8)
- [ ] Cross-model judging to estimate self-preference bias
- [ ] Conformal abstention threshold calibrated on a labelled set
- [ ] Figures — the score-overlap plot is the headline image
- [ ] Interactive mode for demoing
- [ ] README

---

## 8. The narrative, in order

1. Built an agentic RAG system over regulatory documents — it retrieves, reasons, cites, and abstains.
2. Discovered retrieval score cannot separate answerable from unanswerable questions. No threshold exists.
3. Built an eval to measure groundedness. Every score came back 5 — the metric had zero variance and carried no information.
4. Ran a negative control on the judge. It *does* detect corruption, so the ceiling was the test set, not the instrument.
5. Found the judge is weakest at detecting fabricated citations — the failure that matters most in compliance. *(pending replication)*
6. Found my own ground-truth labelling had a defect: lexical absence is not semantic absence.

**Through-line:** at every layer, the number that looked like a measurement wasn't measuring what
it appeared to measure. It is the same question one asks of any statistical estimator — does a
reported confidence deserve to be believed — asked of a different model class.

---

## 9. THE HEADLINE RESULT — non-determinism produced a compliance failure

*Added after the fresh eval run, 2026-08-08 ~14:47.*

Q5 — *"Are community banks required to perform annual model validation?"* — was answered
**correctly on one run and incorrectly on another**, from identical code, corpus, and question.

**Run 1.** Retrieved `occ_2026_13` and cited the OCC's explicit statement that guidance
*"does not, and should not be interpreted to, require community banks to perform annual model
validation."* Also flagged SR 11-7 by name as *"an older, now-superseded guidance document."*
Correctness 5, citation check True.

**Run 2.** Never retrieved `occ_2026_13`. Context was 6 chunks from `sr_11_7_superseded` and 2
from `sr_26_2`. The answer asserted:

> *"guidance does establish a general expectation that banks conduct a periodic review of each
> model at least annually"* — (sr_11_7_superseded.txt, index=30)

**It cited rescinded 2011 guidance as current authority with no indication it was superseded.**
A reader comes away believing annual review is expected; the OCC said the opposite in April 2026.
Correctness 3, `cites_expected_source` False, programmatic citation check False.

### Why it matters

**Groundedness scored 5 on both runs — and the judge was correct both times.** Every claim in
run 2 *is* supported by the context it was given. The context was the wrong document.

Groundedness asks *"did the answer follow from the evidence?"* It never asks *"was that the right
evidence?"* So the metric most commonly reported as the headline number in RAG evaluation is
**structurally incapable of detecting this failure class**. Reference-free evaluation is blind to
it by construction. Only correctness (which requires ground truth) and the citation check caught it.

**The cause is non-determinism.** Same question, same corpus, same code — the agent authored
slightly different search queries, retrieved a different chunk set, and produced a materially
different compliance answer. This escalates the trajectory-variance finding (§4.8) from
*different path, same answer* to **different path, different answer**, on the question where it
matters most.

### Implication

A single eval run would have reported this system as passing the staleness trap. It passes
sometimes. **Any evaluation of a stochastic system that runs each item once is measuring one
sample from a distribution and reporting it as a property.** That is the case for repeated
measurement, and it is the same argument as running multiple chains rather than trusting one.

### Scorecard (single run each, n=1 per item)

| Category | Result |
|---|---|
| Answerable (Q1–Q4) | 4/4 correct, all cited correctly |
| Staleness trap (Q5) | **Failed this run** — cited superseded guidance |
| Hard negatives | Q6 abstained ✅ · Q8 abstained ✅ · **Q7 did not abstain ❌** |

### Judge self-consistency

Across all 40 judgments — 8 questions × 5 judge runs — **within-question variance was exactly
zero.** The judge never disagreed with itself once, at default temperature. Either the items were
unambiguous enough to admit only one score, or the judge is far more deterministic in practice
than its temperature setting suggests. Both are worth knowing; neither was assumed.

The ceiling effect (§4.2) therefore refines rather than dies: *within*-question variance is zero,
but *between*-question variance is now non-zero, because a question finally failed.

---

## 10. Negative-control replication — judge sensitivity profile CONFIRMED

*5 questions × 4 variants × 3 judge runs = 60 judgments, 2026-08-08 ~15:00.*

Groundedness score by corruption type:

| Question | original | fabricated content | fake citation | contradicts context |
|---|---|---|---|---|
| Q1 | 5 | **2** | 4 | 2, 3, 2 |
| Q2 | 5 | **2** | 4 | 3, 3, 4 |
| Q3 | 5 | **2** | 3 | 4, 4, 5 |
| Q4 | 5 | **2** | 4 | **5, 5, 4** |
| Q5 | 5 | **2** | 4, 3, 3 | 2, 2, 3 |

### 10.1 The provenance finding replicates (5/5 questions)

- **Fabricated content:** 5 → **2** on all 15 judgments. A 3-point penalty, perfectly consistent.
- **Fabricated citation:** 5 → **3 or 4**, every question. A 1–2 point penalty.

The judge penalises invented *facts* more than invented *sources*, without exception. This
upgrades §4.4 from a single-question observation to a replicated property of the instrument.

**Why it matters:** in a regulatory setting provenance *is* the product. Nobody asks "is this
claim plausible" — they ask "which document says that, and can I read it." The judge is weakest
exactly where the domain is most demanding.

### 10.2 NEW — judge self-consistency depends on the corruption type

`contradicts_context` is the **only** variant where the judge disagreed with itself, and it did so
on all five questions. Every other variant produced identical scores across three runs.

**Judge reliability is therefore not a single number.** It is perfectly reliable on fabrication
and unreliable on contradiction — least reliable where the corruption is most subtle. Reporting
"our judge shows high inter-run agreement" is meaningless without stating *on what*: measure
agreement on easy cases and you will report a reliability you do not have on hard ones.

This refines §9's "zero within-question variance" finding rather than contradicting it. The
original answers and the blatant corruptions admit only one score; the subtle corruption does not.

**Worst single case:** on Q4 the judge scored a deliberately contradictory answer **5** on two of
three runs — full marks for an answer constructed to contradict its own evidence. Caught only
because it was run three times.

### 10.3 NEW — both citation checks fail on fabricated provenance

`citation_check_programmatic` returned **True** for the `cites_unretrieved_source` variant on
every question. The corruption *adds* a bogus citation while retaining the genuine ones, and the
regex is a **presence test** — it confirms that an acceptable source appears, never that no
fabricated source does.

So on fake provenance: the deterministic check passes it outright, and the judge deducts a single
point. **Neither mechanism detects it.**

Combined with the earlier result that the programmatic check also passed a `fabricated_claim`
variant (which cited a genuinely retrieved source), the two checks fail in complementary but
overlapping ways:

| Corruption | Programmatic check | Judge |
|---|---|---|
| Fabricated content, real citation | ❌ passes | ✅ catches (5→2) |
| Fabricated citation added alongside real ones | ❌ passes | ⚠️ weak (5→3/4) |
| Contradicts context | ❌ passes | ⚠️ unreliable (2–5) |

Neither is sufficient alone, and their union still leaves fabricated provenance essentially
undetected.
