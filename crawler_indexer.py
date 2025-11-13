#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import json
import math
import uuid
import argparse
import logging
from typing import List, Dict, Set, Optional, Iterable
from urllib.parse import urlparse

# allow local imports from project root
sys.path.insert(0, os.path.dirname(__file__))

# project utils: tokenizer
try:
    from utils import lex
except Exception:
    # fallback simple tokenizer
    import re
    def lex(text: str):
        if not text:
            return []
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

# webutils expected to provide ROBOTS, fetch, extract_text_and_links, HostRateLimiter
import webutils  # type: ignore

try:
    import tldextract
except Exception:
    tldextract = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("crawler_indexer")

DEFAULT_MAX_DEPTH = 2
DEFAULT_MIN_INTERVAL = 1.0
DEFAULT_MAX_LINKS_PER_DEPTH = 200

def host_root_from_url(url: str) -> str:
    """Return language/host root used to keep crawl in same wiki subdomain."""
    if tldextract:
        ex = tldextract.extract(url)
        sub = ex.subdomain or ""
        root = getattr(ex, "top_domain_under_public_suffix", None) or getattr(ex, "registered_domain", None) or ex.domain
        return f"{sub}.{root}".strip(".")
    else:
        return urlparse(url).netloc

class SyncCrawler:
    def __init__(self, seeds: List[str], max_depth: int = 2, per_host_min_interval: float = DEFAULT_MIN_INTERVAL,
                 allowed_domains: Optional[Iterable[str]] = None, max_links_per_depth: int = DEFAULT_MAX_LINKS_PER_DEPTH,
                 ignore_robots: bool = False):
        self.seeds = seeds[:]
        self.max_depth = max_depth
        self.rate_limiter = webutils.HostRateLimiter(per_host_min_interval)
        # If allowed_domains is None -> no domain filtering (allow all).
        if allowed_domains is None:
            self.allowed_domains: Optional[Set[str]] = None
        else:
            self.allowed_domains = set(allowed_domains)
        self.max_links_per_depth = max_links_per_depth
        self.ignore_robots = bool(ignore_robots)
        self.visited: Set[str] = set()
        self.results: List[Dict[str, str]] = []
        self.seed_root = host_root_from_url(seeds[0]) if seeds else ""

    def _is_allowed_domain(self, url: str) -> bool:
        """If allowed_domains is None -> allow all; otherwise check suffix match."""
        if self.allowed_domains is None:
            return True
        host = urlparse(url).netloc.lower()
        for d in self.allowed_domains:
            if host == d or host.endswith("." + d):
                return True
        return False

    def _is_same_language_root(self, url: str) -> bool:
        try:
            return host_root_from_url(url) == self.seed_root
        except Exception:
            return True

    def _normalize(self, url: str) -> str:
        if "#" in url:
            url = url.split("#", 1)[0]
        return url

    def crawl(self) -> List[Dict[str, str]]:
        """Synchronous BFS crawl returning list of dicts: {'url','title','text'}"""
        current = [s for s in self.seeds if isinstance(s, str) and s.startswith("http")]
        filtered = []
        for u in current:
            if not self._is_allowed_domain(u):
                logger.info("Seed excluded by domain filter: %s", u)
                continue
            if not self.ignore_robots:
                try:
                    ok = webutils.can_fetch(u)
                except Exception as e:
                    logger.warning("robots check error for %s: %s (treating as blocked)", u, e)
                    ok = False
                if not ok:
                    logger.info("Seed excluded by robots.txt: %s", u)
                    continue
            filtered.append(u)
        current = filtered
        depth = 0
        logger.info("Starting sync crawl: seeds=%d, max_depth=%d, ignore_robots=%s", len(current), self.max_depth, self.ignore_robots)

        while current and depth <= self.max_depth:
            logger.info("Processing depth %d (queue=%d)", depth, len(current))
            next_links: List[str] = []
            processed = 0
            for url in current:
                url = self._normalize(url)
                if url in self.visited:
                    continue
                if not self._is_allowed_domain(url):
                    continue
                if not self._is_same_language_root(url):
                    continue
                if not self.ignore_robots:
                    try:
                        ok = webutils.can_fetch(url)
                    except Exception as e:
                        logger.warning("robots check error for %s: %s (treating as blocked)", url, e)
                        ok = False
                    if not ok:
                        logger.info("Blocked by robots: %s", url)
                        continue

                # honor crawl-delay (if set), otherwise per-host rate limiter
                cd = None
                try:
                    cd = webutils.ROBOTS.get_crawl_delay(url)
                except Exception:
                    cd = None
                host = urlparse(url).netloc
                if cd:
                    logger.info("Honoring Crawl-delay=%s for %s", cd, host)
                    time.sleep(cd)
                else:
                    try:
                        self.rate_limiter.wait(host)
                    except Exception:
                        pass

                try:
                    logger.info("Fetching: %s", url)
                    html = webutils.fetch(url, timeout=12.0)
                    if not html:
                        logger.warning("Empty or non-html: %s", url)
                        self.visited.add(url)
                        continue
                    text, links, title = webutils.extract_text_and_links(url, html)
                    self.results.append({"url": url, "title": title or "", "text": text or ""})
                    self.visited.add(url)
                    processed += 1

                    # collect next links
                    if depth < self.max_depth:
                        for href in links:
                            if not href.startswith("http"):
                                continue
                            hn = self._normalize(href)
                            if not self._is_allowed_domain(hn):
                                continue
                            if not self._is_same_language_root(hn):
                                continue
                            if hn in self.visited:
                                continue
                            if hn in next_links:
                                continue
                            next_links.append(hn)
                            if len(next_links) >= self.max_links_per_depth:
                                break
                except Exception as e:
                    logger.warning("Error fetching %s: %s", url, e)
                    self.visited.add(url)
                    continue

            logger.info("Depth %d processed: %d pages, next_links=%d", depth, processed, len(next_links))
            # unique next
            seen = set()
            uniq = []
            for l in next_links:
                if l not in seen:
                    seen.add(l)
                    uniq.append(l)
            current = uniq
            depth += 1

        logger.info("Crawl finished. Collected pages: %d", len(self.results))
        return self.results

# ---------------- Pure-Python TF-IDF builder (used if no Beam indexer present) ----------------
def build_index_from_docs_python(docs: List[Dict[str, str]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # create deterministic ids (UUID5 from URL)
    doc_records = []
    for d in docs:
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, d["url"]))
        doc_records.append({"id": doc_id, "url": d["url"], "title": d.get("title",""), "text": d.get("text","")})

    docs_path = os.path.join(out_dir, "docs.jsonl")
    with open(docs_path, "w", encoding="utf-8") as f:
        for rec in doc_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote docs.jsonl (n=%d)", len(doc_records))

    # tokenize & counts
    per_doc_counts: Dict[str, Dict[str,int]] = {}
    for rec in doc_records:
        toks = lex(rec["text"])
        c: Dict[str,int] = {}
        for t in toks:
            c[t] = c.get(t,0) + 1
        per_doc_counts[rec["id"]] = c

    # df
    df: Dict[str,int] = {}
    for counts in per_doc_counts.values():
        for term in counts.keys():
            df[term] = df.get(term,0) + 1

    N = max(1, len(doc_records))

    tfidf_records = []
    for rec in doc_records:
        docid = rec["id"]
        counts = per_doc_counts.get(docid, {})
        total = float(sum(counts.values())) or 1.0
        tfidf: Dict[str,float] = {}
        for t,cnt in counts.items():
            dfi = float(df.get(t,1))
            # smoothed idf
            idf = math.log(1.0 + (N/dfi))
            tfidf[t] = (cnt/total) * idf
        tfidf_records.append({"id": docid, "url": rec["url"], "title": rec.get("title",""), "tfidf": tfidf})

    # write jsonl
    idx_jsonl_path = os.path.join(out_dir, "tfidf_index.jsonl")
    with open(idx_jsonl_path, "w", encoding="utf-8") as f:
        for rec in tfidf_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote tfidf_index.jsonl (n=%d)", len(tfidf_records))

    # compact JSON map id->record (url + tfidf)
    compact = {rec["id"]: {"url": rec["url"], "title": rec.get("title",""), "tfidf": rec["tfidf"]} for rec in tfidf_records}
    compact_path = os.path.join(out_dir, "tfidf_index.json")
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False)
    logger.info("Wrote tfidf_index.json (n=%d)", len(compact))

# ---------------- Compatibility wrapper used in tests ----------------
def run_pipeline(seeds, out_dir: str, max_depth: int = 0, per_host_min_interval: float = 1.0, allowed_domains: Optional[Iterable[str]] = None):
    """
    Crawl seeds and build index. Signature kept compatible with tests:
      run_pipeline([seed], out_dir, max_depth=0, per_host_min_interval=0.01, allowed_domains=None)
    - seeds: list or single string
    - out_dir: output directory
    - returns: list of page dicts
    """
    if isinstance(seeds, str):
        seeds = [seeds]
    seeds = list(seeds or [])
    # allowed_domains: if None -> no filter. If empty iterable -> treat as no filter as well.
    allowed_dom_list: Optional[List[str]]
    if allowed_domains is None:
        allowed_dom_list = None
    else:
        # convert to list; if empty -> None (no filter)
        allowed_dom_list = list(allowed_domains) or None

    os.makedirs(out_dir, exist_ok=True)

    crawler = SyncCrawler(
        seeds=seeds,
        max_depth=int(max_depth),
        per_host_min_interval=float(per_host_min_interval),
        allowed_domains=allowed_dom_list,
        max_links_per_depth=DEFAULT_MAX_LINKS_PER_DEPTH,
        ignore_robots=False,
    )

    pages = crawler.crawl()

    # always write docs.jsonl (test expects this)
    docs_path = os.path.join(out_dir, "docs.jsonl")
    os.makedirs(os.path.dirname(docs_path) or ".", exist_ok=True)
    with open(docs_path, "w", encoding="utf-8") as f:
        for p in pages:
            rec = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, p["url"])), "url": p["url"], "title": p.get("title",""), "text": p.get("text","")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote docs.jsonl (n=%d) to %s", len(pages), docs_path)

    # choose Beam indexer if present, else fallback to python builder
    indexer_func = None
    # prefer a function named build_index_from_docs if defined at module level (tests may monkeypatch this)
    if "build_index_from_docs" in globals() and callable(globals().get("build_index_from_docs")):
        indexer_func = globals().get("build_index_from_docs")
    # try optional beam_indexer module
    if indexer_func is None:
        try:
            from beam_indexer import build_index_from_docs as beam_indexer_func  # type: ignore
            indexer_func = beam_indexer_func
        except Exception:
            indexer_func = None

    if indexer_func:
        try:
            indexer_func(pages, out_dir)
        except Exception as e:
            logger.exception("Indexer (build_index_from_docs) failed: %s. Falling back to python indexer.", e)
            build_index_from_docs_python(pages, out_dir)
    else:
        build_index_from_docs_python(pages, out_dir)

    return pages

# ---------------- CLI ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Sync crawler + TF-IDF indexer (with robots-mode)")
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=DEFAULT_MAX_DEPTH)
    ap.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--max-links-per-depth", type=int, default=DEFAULT_MAX_LINKS_PER_DEPTH)
    ap.add_argument("--no-robots", action="store_true", help="Ignore robots")
    ap.add_argument("--robots-mode", choices=["allow-on-error","block-on-error","strict"], default="allow-on-error", help="How to treat robots.txt errors")
    args = ap.parse_args(argv)

    if args.depth < 0 or args.depth > 5:
        raise SystemExit("--depth must be between 0 and 5")

    # set robots behavior if provided by webutils
    try:
        webutils.ROBOTS.set_mode(args.robots_mode)
    except Exception:
        pass

    os.makedirs(args.out, exist_ok=True)

    # If --domains not provided -> domains None => no domain filter
    allowed_domains = args.domains if args.domains is not None else None

    # Run the crawl+index pipeline
    run_pipeline(args.seeds, args.out, max_depth=args.depth, per_host_min_interval=args.min_interval, allowed_domains=allowed_domains)
    logger.info("Done. Outputs in: %s", args.out)

if __name__ == "__main__":
    main()
