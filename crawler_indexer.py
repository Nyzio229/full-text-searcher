from __future__ import annotations

import os
import re
import json
import math
import uuid
import argparse
from typing import Dict, Iterable, List, Tuple, Set
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
    def process(self, doc: Dict):
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc["url"]))
        yield {"id": doc_id, **doc}

def run_pipeline(seeds: List[str], out_dir: str, max_depth: int = 3, per_host_min_interval: float = 1.0,
                 allowed_domains: List[str] = None):
    allowed = set(allowed_domains or list(WIKI_DOMAINS))

    with beam.Pipeline(options=PipelineOptions([])) as p:
        urls_d0 = (
            p | "Seeds" >> beam.Create(seeds)
              | "Filter0" >> beam.ParDo(FilterAndNormalizeUrls(allowed))
              | "Robots0" >> beam.ParDo(RobotsFilter())
              | "Distinct0" >> beam.Distinct()
        )

        docs_all = []
        current_urls = urls_d0

        for depth in range(max_depth + 1):
            docs = current_urls | f"Fetch D{depth}" >> beam.ParDo(FetchPage(min_interval_sec=per_host_min_interval))
            docs_all.append(docs)

            if depth < max_depth:
                current_urls = (
                    docs
                    | f"Links D{depth}" >> beam.ParDo(ExtractLinks())
                    | f"Filter D{depth}" >> beam.ParDo(FilterAndNormalizeUrls(allowed))
                    | f"Robots D{depth}" >> beam.ParDo(RobotsFilter())
                    | f"Distinct D{depth}" >> beam.Distinct()
                )

        all_docs = docs_all | "FlattenDocs" >> beam.Flatten() | "With IDs" >> beam.ParDo(ToDocId())
        _ = (all_docs | "Write Docs" >> beam.Map(json.dumps) |
             beam.io.WriteToText(os.path.join(out_dir, "docs.jsonl"), shard_name_template=""))

        # Tokenize -> TFIDF
        def compute_tfidf(docs):
            from utils import MakeTfIdfFn, TermStatsFn
            return docs | "TFIDF" >> MakeTfIdfFn()

        tfidf = compute_tfidf(all_docs)

        url_map = (all_docs | beam.Map(lambda d: (d["id"], d["url"])) | beam.combiners.ToDict())

        def pack(entry, url_map):
            return {"id": entry["id"], "url": url_map.get(entry["id"], ""), "tfidf": entry["tfidf"]}

        index_records = tfidf | beam.Map(pack, url_map=beam.pvalue.AsDict(url_map))
        _ = (index_records | beam.Map(json.dumps) |
             beam.io.WriteToText(os.path.join(out_dir, "tfidf_index.jsonl"), shard_name_template=""))

        pairs = index_records | beam.Map(lambda rec: (rec["id"], {"url": rec["url"], "tfidf": rec["tfidf"]}))
        compact = pairs | beam.combiners.ToDict()
        _ = compact | beam.Map(json.dumps) | beam.io.WriteToText(os.path.join(out_dir, "tfidf_index.json"), shard_name_template="")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--min-interval", type=float, default=1.0)
    ap.add_argument("--domains", nargs="*", default=["wikipedia.org"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run_pipeline(args.seeds, args.out, args.depth, args.min_interval, args.domains)

if __name__ == "__main__":
    main()