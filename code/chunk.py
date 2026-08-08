"""
chunk.py — split the corpus into overlapping word-windows.

Each chunk keeps three things:
  text    : the words themselves
  source  : which file it came from   -> needed for citations
  index   : its position in that file -> needed to point at it later

Run:  python code/chunk.py
Out:  data/chunks.json
"""

import json
from pathlib import Path

# ---- parameters -------------------------------------------------------
# 200 words is big enough to contain a complete idea, small enough that
# TF-IDF isn't diluted across unrelated topics.
# 50 words of overlap (25%) means a fact sitting on a boundary still
# appears whole in one of the two neighbouring chunks.
CHUNK_SIZE = 200
OVERLAP = 50

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping windows of `size` words."""
    words = text.split()               # flat list of words; ignores line structure
    step = size - overlap              # how far we advance each time
    chunks = []

    for start in range(0, len(words), step):
        window = words[start:start + size]
        if len(window) < 20:           # drop a tiny leftover tail
            continue
        chunks.append(" ".join(window))
        if start + size >= len(words): # we've consumed the whole document
            break

    return chunks


def build_chunks(data_dir=DATA_DIR):
    """Read every .txt in data/ and return a list of chunk records."""
    records = []

    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, chunk in enumerate(chunk_text(text)):
            records.append({
                "chunk_id": len(records),   # unique across the whole corpus
                "source": path.name,
                "index": i,                 # position within this document
                "text": chunk,
            })

    return records


if __name__ == "__main__":
    chunks = build_chunks()

    out = DATA_DIR / "chunks.json"
    out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    # ---- sanity checks ------------------------------------------------
    print(f"{len(chunks)} chunks written to {out}")

    print("\nchunks per source:")
    counts = {}
    for c in chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    for source, n in sorted(counts.items()):
        print(f"  {n:4d}  {source}")

    lengths = [len(c["text"].split()) for c in chunks]
    print(f"\nwords per chunk: min={min(lengths)} max={max(lengths)}")

    # Print two adjacent chunks so you can SEE the overlap with your own eyes.
    print("\n--- chunk 0, last 60 words ---")
    print(" ".join(chunks[0]["text"].split()[-60:]))
    print("\n--- chunk 1, first 60 words ---")
    print(" ".join(chunks[1]["text"].split()[:60]))
