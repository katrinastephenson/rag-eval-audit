"""
retrieve.py — TF-IDF search over the chunked corpus.

Builds a TF-IDF matrix over data/chunks.json and returns the top-k
chunks for a query, ranked by cosine similarity.

Why brute-force cosine over ~285 chunks:
  An ANN index (FAISS, HNSW) trades exactness for speed. At this
  corpus size a single sparse matrix-vector product is already
  sub-millisecond AND returns the true top-k. Nothing to gain.

Why cache the vectorizer:
  Fitting the vocabulary is the expensive step. We do it once at
  import time (module-level) and reuse it for every query.

Run:  python code/retrieve.py
"""

import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
SCORES_PATH = DATA_DIR / "retrieval_scores.csv"

EVAL_QUERIES = {
    "on_topic": [
        "What must a creditor disclose when using AI to deny credit?",
        "How should banks manage risk from third-party service providers?",
        "What are the requirements for model validation?",
    ],
    "hard_negative": [
        "What are the capital requirements for community banks?",
        "What does the guidance say about cryptocurrency custody?",
        "How often must banks retrain machine learning models?",
    ],
}


def _build_index(chunks_path=CHUNKS_PATH):
    """Load chunks and fit a TF-IDF vectorizer over their text."""
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    # stop_words='english' drops common function words that would
    # otherwise dominate the vocabulary without carrying meaning.
    # ngram_range=(1,2) picks up phrases like "adverse action" or
    # "model risk" that are more discriminative than single words.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(c["text"] for c in chunks)
    return chunks, vectorizer, matrix


# Fit once at import. Callers of search_documents() reuse this state.
_CHUNKS, _VECTORIZER, _MATRIX = _build_index()


def search_documents(query, k=5, min_score=None):
    """Return the top-k chunks most similar to `query`.

    Each result is a dict with source, index, chunk_id, score, text.
    If `min_score` is given, results below that cosine similarity are
    dropped — returns [] when nothing clears the bar.
    """
    query_vec = _VECTORIZER.transform([query])
    scores = cosine_similarity(query_vec, _MATRIX)[0]

    # argsort ascending; take the last k and reverse for descending.
    top_idx = scores.argsort()[-k:][::-1]

    results = []
    for i in top_idx:
        score = float(scores[i])
        if min_score is not None and score < min_score:
            continue
        chunk = _CHUNKS[i]
        results.append({
            "source": chunk["source"],
            "index": chunk["index"],
            "chunk_id": chunk["chunk_id"],
            "score": score,
            "text": chunk["text"],
        })
    return results


def write_eval_scores(k=5, out_path=SCORES_PATH):
    """Run every EVAL_QUERIES query and dump top-k results to CSV.

    Columns: query, category, rank, score, source, chunk_id.
    Returns the path written.
    """
    rows = []
    for category, queries in EVAL_QUERIES.items():
        for query in queries:
            for rank, r in enumerate(search_documents(query, k=k), start=1):
                rows.append({
                    "query": query,
                    "category": category,
                    "rank": rank,
                    "score": f"{r['score']:.6f}",
                    "source": r["source"],
                    "chunk_id": r["chunk_id"],
                })

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "category", "rank", "score", "source", "chunk_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return out_path


if __name__ == "__main__":
    on_topic = EVAL_QUERIES["on_topic"]
    hard_negatives = EVAL_QUERIES["hard_negative"]

    def print_scores(label, queries):
        print("=" * 78)
        print(label)
        print("=" * 78)
        print(f"{'#1':>7} {'#2':>7} {'#3':>7} {'#4':>7} {'#5':>7}  query")
        for q in queries:
            results = search_documents(q, k=5)
            scores = [f"{r['score']:.4f}" for r in results]
            scores += [" " * 6] * (5 - len(scores))
            print(f"{scores[0]:>7} {scores[1]:>7} {scores[2]:>7} "
                  f"{scores[3]:>7} {scores[4]:>7}  {q}")
        print()

    print_scores("ON-TOPIC", on_topic)
    print_scores("HARD NEGATIVES", hard_negatives)

    out = write_eval_scores()
    print(f"wrote {out}\n")

    # Detailed dump of one on-topic and one hard-negative query so you
    # can eyeball what actually comes back at these score levels.
    for label, q in [("ON-TOPIC EXAMPLE", on_topic[0]),
                     ("HARD-NEGATIVE EXAMPLE", hard_negatives[0])]:
        print("=" * 78)
        print(f"{label}: {q}")
        print("=" * 78)
        for rank, r in enumerate(search_documents(q, k=5), start=1):
            preview = " ".join(r["text"].split()[:40])
            print(
                f"\n[{rank}] score={r['score']:.4f}  "
                f"source={r['source']}  index={r['index']}  "
                f"chunk_id={r['chunk_id']}"
            )
            print(f"    {preview}...")
        print()
