"""Prep MLDR eval data for fr / de / ja / zh.

Downloads:
  - Shitao/MLDR [lang] test split → queries + positive_passages → qrels
  - Shitao/MLDR corpus-<lang> → 10k-doc corpus per lang

Writes:
  mldr_eval/<lang>/queries.jsonl   {qid, query}
  mldr_eval/<lang>/qrels.jsonl     {qid, rels: {docid: 1}}
  mldr_eval/<lang>/corpus.jsonl    {docid, text}     (~10k per lang)
"""
import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset


def prep_lang(lang: str, out_root: str):
    print(f"\n=== {lang} ===", flush=True)
    out_dir = os.path.join(out_root, lang)
    os.makedirs(out_dir, exist_ok=True)

    # Queries + qrels from test split
    ds = load_dataset("Shitao/MLDR", lang, trust_remote_code=True, split="test")
    queries: dict[str, str] = {}
    qrels = defaultdict(dict)
    for r in ds:
        queries[r["query_id"]] = r["query"]
        for p in r["positive_passages"]:
            qrels[r["query_id"]][p["docid"]] = 1

    with open(os.path.join(out_dir, "queries.jsonl"), "w") as f:
        for qid, q in queries.items():
            f.write(json.dumps({"qid": qid, "query": q}, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "qrels.jsonl"), "w") as f:
        for qid, rels in qrels.items():
            f.write(json.dumps({"qid": qid, "rels": rels}) + "\n")

    n_rel = sum(1 for rels in qrels.values() for v in rels.values() if v > 0)
    print(f"  queries: {len(queries)}  positive qrels: {n_rel}", flush=True)

    # Corpus (canonical 10k subset shipped as corpus-<lang>)
    ds_c = load_dataset("Shitao/MLDR", f"corpus-{lang}", trust_remote_code=True, split="corpus")
    with open(os.path.join(out_dir, "corpus.jsonl"), "w") as f:
        for r in ds_c:
            f.write(json.dumps({"docid": r["docid"], "text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"  corpus: {len(ds_c)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["fr", "de", "ja", "zh"])
    p.add_argument("--out", default="mldr_eval")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for lang in args.langs:
        prep_lang(lang, args.out)
    print("\nDONE")


if __name__ == "__main__":
    main()
