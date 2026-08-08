"""
plot_figures.py — three figures for the audit write-up.

Design constraints:
  - No gridlines, no chartjunk (top/right spines off, no background).
  - Okabe-Ito palette (colourblind-safe).
  - Titles state the finding, not the axes.
  - One row per unit-of-analysis on the y-axis where relevant, so the
    reader can see individual runs rather than a summary statistic.

Run:  python code/plot_figures.py
Out:  figures/{01,02,03}_*.png
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Okabe-Ito, chosen for max discriminability under all common CVDs.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
BLACK = "#000000"
GREY = "#999999"

TITLE_KW = dict(loc="left", fontsize=13, fontweight="bold", pad=14)
LABEL_KW = dict(fontsize=11)


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="out", length=4)


# --------------------------------------------------------------------- #
# Figure 1 — retrieval score overlap
# --------------------------------------------------------------------- #

def figure_1_score_overlap():
    csv_path = DATA_DIR / "retrieval_scores.csv"
    by_query = {}
    for row in csv.DictReader(csv_path.open()):
        d = by_query.setdefault(row["query"], {"cat": row["category"],
                                               "scores": []})
        d["scores"].append(float(row["score"]))

    # Sort: on-topic first, then hard-negative. Within each, keep CSV order
    # (which is the order the queries were issued).
    order = ["on_topic", "hard_negative"]
    items = sorted(by_query.items(),
                   key=lambda kv: (order.index(kv[1]["cat"]), kv[0]))

    fig, ax = plt.subplots(figsize=(10, 4.2))

    y_positions = []
    y_labels = []
    y = 0
    prev_cat = None
    for query, d in items:
        if prev_cat is not None and d["cat"] != prev_cat:
            y += 0.8  # visual gap between groups
        color = BLUE if d["cat"] == "on_topic" else VERMILLION
        ax.scatter(d["scores"], [y] * len(d["scores"]),
                   color=color, s=70, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.6)
        y_positions.append(y)
        # truncate label to keep the figure readable
        y_labels.append(query if len(query) <= 55
                        else query[:54] + "…")
        y += 1
        prev_cat = d["cat"]

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()  # top-to-bottom reading
    ax.set_xlabel("cosine similarity (top-5 chunks per query)", **LABEL_KW)
    ax.set_xlim(-0.02, None)
    _clean(ax)

    # Category legend — two labelled dots in the top-right corner.
    ax.scatter([], [], color=BLUE, s=70, label="on-topic")
    ax.scatter([], [], color=VERMILLION, s=70, label="hard-negative")
    ax.legend(frameon=False, loc="lower right", fontsize=10)

    ax.set_title(
        "No retrieval threshold separates answerable from "
        "unanswerable questions", **TITLE_KW)

    out = FIG_DIR / "01_score_overlap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------- #
# Figure 2 — Q5 failure (staleness trap)
# --------------------------------------------------------------------- #

def figure_2_q5_failure():
    eval_runs = []
    for p in sorted((DATA_DIR / "eval_runs").glob("q05_*.json")):
        r = json.loads(p.read_text())
        sources = {c["source"] for c in r["chunks_in_context"]}
        eval_runs.append({
            "file": p.name,
            "retrieved_occ": "occ_2026_13.txt" in sources,
        })

    # Correctness scores per eval-run-file (one file → possibly many judgments).
    scores_by_run = defaultdict(list)
    for p in sorted((DATA_DIR / "judgments").glob("q05_*.json")):
        j = json.loads(p.read_text())
        if j.get("correctness_score") is not None:
            scores_by_run[j["eval_run_file"]].append(j["correctness_score"])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    y_labels = []
    for i, run in enumerate(eval_runs):
        y = i
        color = BLUE if run["retrieved_occ"] else GREY
        # Plot every judgment as its own dot with small vertical jitter so
        # overlapping scores are visible.
        scores = scores_by_run.get(run["file"], [])
        n = len(scores)
        for idx, s in enumerate(scores):
            offset = (idx - (n - 1) / 2) * 0.10 if n > 1 else 0
            ax.scatter([s], [y + offset], color=color, s=110, alpha=0.9,
                       edgecolors="white", linewidths=0.7, zorder=3)
        y_labels.append(
            f"run {i + 1}  " +
            ("OCC 2026-13 retrieved" if run["retrieved_occ"]
             else "OCC 2026-13 MISSING"))

    ax.set_yticks(range(len(eval_runs)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("judge correctness score  (one dot per judgment run)",
                  **LABEL_KW)
    _clean(ax)

    n_ok = sum(r["retrieved_occ"] for r in eval_runs)
    ax.set_title(
        f"When the current-guidance chunk is retrieved (1 of "
        f"{len(eval_runs)} runs), the answer is correct; "
        f"otherwise it isn't", **TITLE_KW)

    out = FIG_DIR / "02_q5_failure.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (retrieval hit rate: {n_ok}/{len(eval_runs)})")


# --------------------------------------------------------------------- #
# Figure 3 — judge sensitivity to corruption type
# --------------------------------------------------------------------- #

def figure_3_judge_sensitivity():
    variants = [
        "original",
        "fabricated_claim",
        "cites_unretrieved_source",
        "contradicts_context",
    ]
    variant_colors = {
        "original": BLACK,
        "fabricated_claim": VERMILLION,
        "cites_unretrieved_source": ORANGE,
        "contradicts_context": PURPLE,
    }
    # Human-readable x-tick labels (match user's phrasing).
    variant_labels = {
        "original": "original",
        "fabricated_claim": "fabricated\ncontent",
        "cites_unretrieved_source": "fake\ncitation",
        "contradicts_context": "contradicts\ncontext",
    }

    by_variant = defaultdict(list)
    for p in sorted((DATA_DIR / "negative_control").glob("*.json")):
        r = json.loads(p.read_text())
        by_variant[r["variant"]].append(r["groundedness_score"])

    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Strip plot: for each variant, jitter the individual scores horizontally
    # around the variant's x position so overlapping dots are visible.
    import random
    random.seed(0)  # deterministic jitter for reproducible figures
    for i, variant in enumerate(variants):
        scores = by_variant.get(variant, [])
        xs = [i + (random.random() - 0.5) * 0.25 for _ in scores]
        ax.scatter(xs, scores, color=variant_colors[variant], s=80,
                   alpha=0.7, edgecolors="white", linewidths=0.6,
                   zorder=3)

    ax.set_xticks(range(len(variants)))
    # Sample size goes inline with the variant label so it doesn't collide
    # with any other axis element.
    ax.set_xticklabels(
        [f"{variant_labels[v]}\n(n={len(by_variant.get(v, []))})"
         for v in variants],
        fontsize=11)
    ax.set_ylabel("groundedness score  (1 = fabricated, 5 = fully grounded)",
                  **LABEL_KW)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    _clean(ax)

    ax.set_title(
        "Judge cleanly demotes fabrication and fake citations; "
        "disagrees on contradictions", **TITLE_KW)

    out = FIG_DIR / "03_judge_sensitivity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    figure_1_score_overlap()
    figure_2_q5_failure()
    figure_3_judge_sensitivity()
