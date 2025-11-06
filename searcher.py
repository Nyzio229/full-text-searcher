from __future__ import annotations

import os
import json
import math
import argparse
from collections import Counter
from typing import Dict, List, Tuple

from utils import lex, normalize_en


def read_index(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index not found: {path}")
    data = json.load(open(path, "r", encoding="utf-8"))
    # policz normę wektora, ale nic nie modyfikuj
    for d in data:
        d["_norm"] = math.sqrt(sum(v * v for v in d.get("tfidf", {}).values())) or 0.0
    return data


def vectorize_query(q: str, docs: List[Dict]) -> Tuple[Dict[str, float], float]:
    """
    TF-IDF zapytania liczone względem DF/N z indeksu (prosty IDF: log(1 + N/df)).
    """
    N = len(docs)
    if N == 0:
        return {}, 0.0

    # DF odtworzone z indeksu (ile dokumentów ma dany token)
    df: Dict[str, int] = {}
    for d in docs:
        for t in d["tfidf"].keys():
            df[t] = df.get(t, 0) + 1

    q_tokens = normalize_en(lex(q), lemmatize=True)
    if not q_tokens:
        return {}, 0.0

    tf = Counter(q_tokens)  # surowe TF zapytania
    idf = {t: math.log(1.0 + (N / float(df.get(t, 1)))) for t in df.keys()}

    qvec = {t: tf[t] * idf.get(t, 0.0) for t in tf.keys() if t in idf and idf.get(t, 0.0) > 0.0}
    qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 0.0
    return qvec, qnorm


def cosine(qvec: Dict[str, float], qnorm: float, doc: Dict) -> float:
    if qnorm <= 0.0 or doc.get("_norm", 0.0) <= 0.0:
        return 0.0
    # iloczyn tylko po wspólnych kluczach
    dot = 0.0
    dvec = doc.get("tfidf", {})
    for t, w in qvec.items():
        dv = dvec.get(t)
        if dv:
            dot += w * dv
    return dot / (qnorm * doc["_norm"])


def search(index_path: str, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
    docs = read_index(index_path)
    qvec, qnorm = vectorize_query(query, docs)
    if qnorm == 0.0:
        return []

    scored = [(d["id"], d["path"], cosine(qvec, qnorm, d)) for d in docs]
    scored = [s for s in scored if s[2] > 0.0]
    scored.sort(key=lambda x: (-x[2], x[1]))
    return scored[:top_k]


def main():
    ap = argparse.ArgumentParser(description="Search TF-IDF JSON index (EN-only).")
    ap.add_argument("--index", default="tfidf_index.json", help="Path to JSON index file")
    ap.add_argument("--q", required=True, help="Query string (English)")
    ap.add_argument("--k", type=int, default=10, help="Top-K results")
    args = ap.parse_args()

    hits = search(args.index, args.q, args.k)
    if not hits:
        print("No results.")
        return
    print(f"Top {len(hits)} results:")
    for i, (doc_id, path, score) in enumerate(hits, 1):
        print(f"{i:2d}. {score:.6f}  {path}  [{doc_id}]")


if __name__ == "__main__":
    main()