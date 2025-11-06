from __future__ import annotations

# Preferuj lokalne moduły względem paczek z PyPI o tej samej nazwie
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import re
import json
import math
import uuid
import argparse
from typing import Dict, List, Set
from urllib.parse import urlparse

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from utils import lex
from webutils import HostRateLimiter, can_fetch, fetch, extract_text_and_links

WIKI_DOMAINS = {"wikipedia.org", "en.wikipedia.org", "pl.wikipedia.org"}


def is_allowed_domain(url: str, allowed_domains: Set[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in allowed_domains)


class FilterAndNormalizeUrls(beam.DoFn):
    def __init__(self, allowed_domains: Set[str]):
        self.allowed = set(allowed_domains)

    def process(self, url: str):
        if not url or not url.startswith("http"):
            return
        if is_allowed_domain(url, self.allowed):
            # usuń fragmenty (#...)
            yield re.sub(r"#.*$", "", url)


class RobotsFilter(beam.DoFn):
    def process(self, url: str):
        if can_fetch(url):
            yield url


class FetchPage(beam.DoFn):
    def __init__(self, min_interval_sec: float = 1.0):
        self.limiter = HostRateLimiter(min_interval_sec)

    def process(self, url: str):
        self.limiter.wait(urlparse(url).netloc)
        html = fetch(url)
        if html:
            text, links, title = extract_text_and_links(url, html)
            yield {"url": url, "title": title, "text": text, "links": links}


class ExtractLinks(beam.DoFn):
    def process(self, doc: Dict):
        for link in doc.get("links", []):
            yield link


class ToDocId(beam.DoFn):
    """Stabilne UUID5 na bazie URL (deterministyczne)."""
    def process(self, doc: Dict):
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc["url"]))
        yield {"id": doc_id, **doc}


def run_pipeline(
    seeds: List[str],
    out_dir: str,
    max_depth: int = 2,
    per_host_min_interval: float = 1.0,
    allowed_domains: List[str] = None,
):
    allowed = set(allowed_domains or list(WIKI_DOMAINS))

    with beam.Pipeline(options=PipelineOptions([])) as p:
        # Depth 0 (seedy)
        urls_d0 = (
            p
            | "Seeds" >> beam.Create(seeds)
            | "Filter0" >> beam.ParDo(FilterAndNormalizeUrls(allowed))
            | "Robots0" >> beam.ParDo(RobotsFilter())
            | "Distinct0" >> beam.Distinct()
        )

        docs_all = []
        current_urls = urls_d0

        # iteracyjne poziomy 0..max_depth
        for depth in range(max_depth + 1):
            docs = current_urls | f"Fetch D{depth}" >> beam.ParDo(
                FetchPage(min_interval_sec=per_host_min_interval)
            )
            docs_all.append(docs)

            if depth < max_depth:
                current_urls = (
                    docs
                    | f"Links D{depth}" >> beam.ParDo(ExtractLinks())
                    | f"Filter D{depth}" >> beam.ParDo(FilterAndNormalizeUrls(allowed))
                    | f"Robots D{depth}" >> beam.ParDo(RobotsFilter())
                    | f"Distinct D{depth}" >> beam.Distinct()
                )

        # scal dokumenty i nadaj ID
        all_docs = docs_all | "FlattenDocs" >> beam.Flatten() | "With IDs" >> beam.ParDo(ToDocId())

        # zapisz surowe dokumenty (jsonl)
        _ = (
            all_docs
            | "Dump docs jsonl" >> beam.Map(json.dumps)
            | "Write docs jsonl" >> beam.io.WriteToText(
                os.path.join(out_dir, "docs.jsonl"),
                shard_name_template=""
            )
        )

        # === Build TF-IDF index (ręcznie, bez MakeTfIdfFn) ===
        # 1) Tokenizacja
        docs_with_tokens = (
            all_docs
            | "Doc->Tokens" >> beam.Map(lambda d: (d["id"], d["url"], lex(d.get("text", ""))))
        )

        # 2) Liczność terminów w dokumencie: counts[term]
        term_stats = (
            docs_with_tokens
            | "Flat terms" >> beam.FlatMap(lambda tup: [((t, tup[0]), 1) for t in tup[2]])   # ((term, doc_id), 1)
            | "Count terms" >> beam.CombinePerKey(sum)                                        # ((term, doc_id), cnt)
            | "Key by doc" >> beam.Map(lambda kv: (kv[0][1], (kv[0][0], kv[1])))              # (doc_id, (term, cnt))
            | "Group per doc" >> beam.GroupByKey()                                            # (doc_id, [(term, cnt), ...])
            | "Make per-doc stats" >> beam.Map(lambda kv: {"id": kv[0], "counts": dict(kv[1])})
        )

        # 3) Document Frequency df[term]
        df = (
            term_stats
            | "Extract terms" >> beam.FlatMap(lambda d: [(t, 1) for t in d["counts"].keys()])
            | "Doc freq" >> beam.CombinePerKey(sum)
        )

        # 4) Liczba dokumentów N
        N = term_stats | "Count docs" >> beam.combiners.Count.Globally()

        # 5) Side-inputy: df jako map i N jako singleton
        df_map = df | "DF as map" >> beam.combiners.ToDict()

        def make_tfidf(d, df_map, Nval):
            counts = d["counts"]
            total = float(sum(counts.values())) or 1.0
            tfidf = {}
            for t, c in counts.items():
                dfi = float(df_map.get(t, 1))
                idf = math.log(1.0 + (Nval / dfi))
                tfidf[t] = (c / total) * idf
            return {"id": d["id"], "tfidf": tfidf}

        tfidf = (
            term_stats
            | "Compute TFIDF" >> beam.Map(
                make_tfidf,
                df_map=beam.pvalue.AsDict(df_map),
                Nval=beam.pvalue.AsSingleton(N),
            )
        )
        # === /Build TF-IDF ===

        # mapowanie id->url
        url_map_pcoll = all_docs | "URL map" >> beam.Map(lambda d: (d["id"], d["url"]))
        url_map = url_map_pcoll | "URL map ToDict" >> beam.combiners.ToDict()

        def attach_url(entry, url_map):
            doc_id = entry["id"]
            return {"id": doc_id, "url": url_map.get(doc_id, ""), "tfidf": entry["tfidf"]}

        index_records = tfidf | "Attach URLs" >> beam.Map(
            attach_url, url_map=beam.pvalue.AsDict(url_map)
        )

        # jsonl
        _ = (
            index_records
            | "Dump index jsonl" >> beam.Map(json.dumps)
            | "Write index jsonl" >> beam.io.WriteToText(
                os.path.join(out_dir, "tfidf_index.jsonl"),
                shard_name_template=""
            )
        )

        # kompakt: { id: {url, tfidf} }
        pairs = index_records | "Pairs" >> beam.Map(lambda rec: (rec["id"], {"url": rec["url"], "tfidf": rec["tfidf"]}))
        compact = pairs | "Pairs->Dict" >> beam.combiners.ToDict()
        _ = (
            compact
            | "Dump compact json" >> beam.Map(json.dumps)
            | "Write compact json" >> beam.io.WriteToText(
                os.path.join(out_dir, "tfidf_index.json"),
                shard_name_template=""
            )
        )


def main():
    ap = argparse.ArgumentParser(description="Wikipedia crawler + TF-IDF indexer (Apache Beam)")
    ap.add_argument("--seeds", nargs="+", required=True, help="Seed URLs (Wikipedia)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--depth", type=int, default=2, help="Max crawl depth (0..5)")
    ap.add_argument("--min-interval", type=float, default=1.0, help="Min seconds between requests per host")
    ap.add_argument("--domains", nargs="*", default=["wikipedia.org"], help="Allowed domains (suffix match)")
    args = ap.parse_args()

    if args.depth < 0 or args.depth > 5:
        raise SystemExit("--depth must be between 0 and 5")

    os.makedirs(args.out, exist_ok=True)
    run_pipeline(
        args.seeds,
        args.out,
        max_depth=args.depth,
        per_host_min_interval=args.min_interval,
        allowed_domains=args.domains,
    )


if __name__ == "__main__":
    main()
