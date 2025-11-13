#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import argparse
import re
from collections import Counter
from typing import Dict, Any, List, Tuple

# prefer your project's utils, but provide fallback
try:
    from utils import lex, normalize_en
except Exception:
    lex = None
    normalize_en = None

# ----------------- fallback simple tokenizer -----------------
_SPLIT_RE = re.compile(r"[^\w']+", flags=re.UNICODE)

def simple_tokenize(text: str) -> List[str]:
    """Lowercase + split on non-word characters; remove empty tokens."""
    if not text:
        return []
    toks = [t.lower() for t in _SPLIT_RE.split(text) if t and not t.isspace()]
    return toks

def make_query_tokens(query: str) -> List[str]:
    """
    Try to produce tokens using project's lex/normalize_en.
    If that produces nothing or is unavailable, fall back to simple_tokenize.
    """
    if not query:
        return []
    if normalize_en and lex:
        try:
            nq = normalize_en(query)
            toks = lex(nq)
            if toks:
                return [t for t in toks if t and not t.isspace()]
        except Exception:
            pass
    # fallback
    return simple_tokenize(query)

# ----------------- robust index loader -----------------

def _try_parse_json_maybe(s: Any):
    if isinstance(s, str):
        s_strip = s.strip()
        if (s_strip.startswith("{") and s_strip.endswith("}")) or (s_strip.startswith("[") and s_strip.endswith("]")):
            try:
                return json.loads(s_strip)
            except Exception:
                return s
    return s

def read_index(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Read index from JSON or JSONL and return:
      { doc_id: {"url":..., "title":..., "tfidf": {term: float}} }
    Robust to:
      - single-record JSON: {"id": "...", "url": "...", "tfidf": {...}}
      - compact mapping: {"id1": {...}, "id2": {...}}
      - JSONL: one JSON per line with {"id":..., "url":..., "tfidf": {...}}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = open(path, "r", encoding="utf-8").read()
    text = raw.strip()
    if not text:
        return {}

    # try parse whole file
    try:
        whole = json.loads(text)
    except Exception:
        whole = None

    out: Dict[str, Dict[str, Any]] = {}

    # whole parsed
    if isinstance(whole, dict):
        # single-record with keys id,url,tfidf
        if "id" in whole and ("tfidf" in whole or "url" in whole):
            doc_id = str(whole["id"])
            tfidf = whole.get("tfidf", {})
            tfidf = _try_parse_json_maybe(tfidf)
            if isinstance(tfidf, str):
                tfidf = _try_parse_json_maybe(tfidf)
            if not isinstance(tfidf, dict):
                tfidf = {}
            out[doc_id] = {"url": whole.get("url", ""), "title": whole.get("title", ""), "tfidf": _normalize_tfidf(tfidf)}
            return out
        # compact mapping id -> record
        if all(isinstance(v, dict) for v in whole.values()):
            for doc_id, rec in whole.items():
                rec = _try_parse_json_maybe(rec)
                if isinstance(rec, dict):
                    tfidf = rec.get("tfidf", {})
                    tfidf = _try_parse_json_maybe(tfidf)
                    if isinstance(tfidf, str):
                        tfidf = _try_parse_json_maybe(tfidf)
                    if not isinstance(tfidf, dict):
                        tfidf = {}
                    out[str(doc_id)] = {"url": rec.get("url", ""), "title": rec.get("title", ""), "tfidf": _normalize_tfidf(tfidf)}
                else:
                    out[str(doc_id)] = {"url": str(rec), "title": "", "tfidf": {}}
            return out
        # otherwise, fall through to line-by-line handling (maybe it's a single-record but weird)
    # else try JSONL or per-line records
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            if "id" in obj and ("tfidf" in obj or "url" in obj):
                doc_id = str(obj["id"])
                tfidf = obj.get("tfidf", {})
                tfidf = _try_parse_json_maybe(tfidf)
                if isinstance(tfidf, str):
                    tfidf = _try_parse_json_maybe(tfidf)
                if not isinstance(tfidf, dict):
                    tfidf = {}
                out[doc_id] = {"url": obj.get("url", "") or obj.get("path", ""), "title": obj.get("title", ""), "tfidf": _normalize_tfidf(tfidf)}
            else:
                # maybe this line is compact mapping of many docs
                if all(isinstance(v, dict) for v in obj.values()):
                    for doc_id, rec in obj.items():
                        rec = _try_parse_json_maybe(rec)
                        if isinstance(rec, dict):
                            tfidf = rec.get("tfidf", {})
                            tfidf = _try_parse_json_maybe(tfidf)
                            if isinstance(tfidf, str):
                                tfidf = _try_parse_json_maybe(tfidf)
                            if not isinstance(tfidf, dict):
                                tfidf = {}
                            out[str(doc_id)] = {"url": rec.get("url", "") or rec.get("path", ""), "title": rec.get("title", ""), "tfidf": _normalize_tfidf(tfidf)}
                        else:
                            out[str(doc_id)] = {"url": str(rec), "title": "", "tfidf": {}}
                    continue
                # skip unknown shape
                continue
    return out

def _normalize_tfidf(raw: Any) -> Dict[str, float]:
    """
    Ensure tfidf mapping is dict[str, float].
    If values are strings or other types, attempt to coerce to float.
    """
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        try:
            if isinstance(v, str):
                v2 = float(v)
            else:
                v2 = float(v)
            out[str(k)] = v2
        except Exception:
            # skip non-numeric
            continue
    return out

# ----------------- vector math -----------------

def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = 0.0
    for t, v in a.items():
        dot += v * b.get(t, 0.0)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

def query_vector(query: str) -> Dict[str, float]:
    toks = make_query_tokens(query)
    if not toks:
        return {}
    c = Counter(toks)
    total = float(sum(c.values())) or 1.0
    return {t: cnt / total for t, cnt in c.items()}

# ----------------- search + debug -----------------

def inspect_index(index: Dict[str, Dict[str, Any]], n_samples: int = 3) -> None:
    print("INDEX INSPECTION:")
    total_docs = len(index)
    print(f" total docs: {total_docs}")
    vocab = set()
    doc_sizes = []
    for doc_id, rec in index.items():
        if not isinstance(rec, dict):
            print(f" [warn] doc {doc_id} has unexpected type {type(rec)}; skipping")
            continue
        tf = rec.get("tfidf", {}) or {}
        if not isinstance(tf, dict):
            tf = {}
        vocab.update(tf.keys())
        doc_sizes.append((doc_id, len(tf)))
    print(f" vocab size (unique terms across docs): {len(vocab)}")
    doc_sizes.sort(key=lambda x: x[1], reverse=True)
    print(" top documents by number of indexed terms:")
    for doc_id, size in doc_sizes[:n_samples]:
        rec = index.get(doc_id, {})
        print(f"  - {doc_id} | terms: {size} | url: {rec.get('url')}")
        tf = rec.get("tfidf", {}) or {}
        if isinstance(tf, dict) and tf:
            top = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:10]
            print("    top terms:", ", ".join(f"{t}:{v:.6f}" for t, v in top))
        else:
            print("    top terms: (none)")

def search(index_path: str, query: str, top_k: int = 10, debug: bool = False) -> List[Tuple[float, str, str, str]]:
    index = read_index(index_path)
    if debug:
        inspect_index(index, n_samples=3)

    qv = query_vector(query)
    if debug:
        print("\nQuery tokens:", list(qv.keys()))
    if not qv:
        if debug:
            print("Empty or no tokens in query after normalization. (fallback tokenizer used if needed)")
        return []

    results = []
    for doc_id, rec in index.items():
        tfidf = rec.get("tfidf", {}) or {}
        if not isinstance(tfidf, dict):
            continue
        overlap = set(qv.keys()) & set(tfidf.keys())
        if debug and overlap:
            print(f"[dbg] doc {doc_id} overlap tokens: {sorted(overlap)} (count {len(overlap)})")
        if not overlap:
            continue
        score = cosine(qv, tfidf)
        if score > 0:
            results.append((score, doc_id, rec.get("url", ""), rec.get("title", "")))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Search tfidf index (robust and with fallback tokenization)")
    ap.add_argument("--index", required=True, help="Path to tfidf_index.json or jsonl")
    ap.add_argument("--query", required=True, help="Query string")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--debug", action="store_true", help="Print debug info")
    args = ap.parse_args()

    res = search(args.index, args.query, args.top_k, debug=args.debug)

    print("\n=== RESULTS ===\n")
    if not res:
        print("No matching documents found.")
        return

    for score, doc_id, url, title in res:
        print(f"[{score:.6f}] {title} ({doc_id})\n{url}\n")

if __name__ == "__main__":
    main()
