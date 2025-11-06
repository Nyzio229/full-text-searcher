from __future__ import annotations

import os
import json
import glob
import uuid
import argparse
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from utils import LoadText, TermStatsFn, MakeTfIdfFn


def build_index(src_dir: str, out_path: str):
    # Zbieramy kandydatów (rekurencyjnie .txt)
    paths = sorted(glob.glob(os.path.join(src_dir, "**", "*.txt"), recursive=True))
    pairs = [(str(uuid.uuid4()), p) for p in paths]

    if not pairs:
        print(f"[indexer] No .txt files found under: {src_dir}")
        return

    opts = PipelineOptions()  # DirectRunner domyślnie
    with beam.Pipeline(options=opts) as p:
        docs = p | "CreateDocList" >> beam.Create(pairs)
        raw = docs | "ReadWithDoFn" >> beam.ParDo(LoadText())

        branches = raw | "TF_and_DF" >> beam.ParDo(TermStatsFn()).with_outputs("tf", "df")
        tf_rows = branches.tf
        df_pairs = branches.df

        # (token, doc) → unique → (token,1) → sum → dict
        df_dict = (
            df_pairs
            | "KeyToken" >> beam.Map(lambda td: (td[0], td[1]))   # (token, doc_id)
            | "UniqueTokenDoc" >> beam.Distinct()
            | "ToOnes" >> beam.Map(lambda td: (td[0], 1))
            | "CountDF" >> beam.CombinePerKey(sum)
            | "DFasDict" >> beam.combiners.ToDict()
        )

        N_pc = docs | "CountDocs" >> beam.combiners.Count.Globally()

        tfidf = tf_rows | "MakeTfIdf" >> beam.ParDo(
            MakeTfIdfFn(),
            df_map=beam.pvalue.AsSingleton(df_dict),
            N=beam.pvalue.AsSingleton(N_pc),
        )

        # zapis jeden raz (ToList → Map)
        _ = (
            tfidf
            | "CollectAll" >> beam.combiners.ToList()
            | "WriteJSON" >> beam.Map(lambda items: json.dump(items, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2))
        )

    print(f"[indexer] Wrote index → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Build TF-IDF JSON index (EN-only).")
    ap.add_argument("--src", required=True, help="Source folder with .txt files (searched recursively)")
    ap.add_argument("--out", default="tfidf_index.json", help="Output JSON file path")
    args = ap.parse_args()

    build_index(args.src, args.out)


if __name__ == "__main__":
    main()