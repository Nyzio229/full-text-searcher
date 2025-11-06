from __future__ import annotations

import os
import json
import math
import argparse
from collections import Counter
from typing import Dict, List, Tuple

from utils import lex, normalize_en


def read_index(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index not found: {path}")
    text = open(path, "r", encoding="utf-8").read().strip()
    # Kompaktowy JSON (cały słownik)
    if text.startswith("{"):
        return json.loads(text)
    # JSONL (linia per dokument)
    index = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        index[rec["id"]] = {"url": rec.get("url", rec.get("path", "")), "tfidf": rec["tfidf"]}
    return index


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for t, va in a.items():
        vb = b.get(t)
        if vb is not None:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def query_vector(q: str) -> Dict[str, float]:
    toks = lex(normalize_en(q))
    c = Counter(toks)
    total = float(sum(c.values())) or 1.0
    return {t: cnt / total for t, cnt in c.items()}


def search(index_path: str, q: str, k: int = 10) -> List[Tuple[str, str, float]]:
    index = read_index(index_path)
    qv = query_vector(q)
    scored = []
    for doc_id, rec in index.items():
        score = cosine(qv, rec["tfidf"])
        if score > 0:
            scored.append((doc_id, rec.get("url", ""), score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:k]


def main():
    ap = argparse.ArgumentParser(description="Beam TF-IDF Searcher")
    ap.add_argument("--index", required=True, help="Path to tfidf_index.json or jsonl")
    ap.add_argument("--q", required=True, help="Query string (English)")
    ap.add_argument("--k", type=int, default=10, help="Top-K results")
    args = ap.parse_args()

    hits = search(args.index, args.q, args.k)
    if not hits:
        print("No results.")
        return
    print(f"Top {len(hits)} results:")
    for i, (doc_id, url, score) in enumerate(hits, 1):
        print(f"{i:2d}. {score:.6f}  {url}  [{doc_id}]")


if __name__ == "__main__":
    main()
