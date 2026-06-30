"""Prep MIRACL dev eval data (fr / ja / de) at 10k corpus per language.

For each lang:
  - Sample 100 queries from the dev split (fixed seed 42)
  - Take all positive + hard-negative passages referenced by those queries
  - Top up with random distractor docs from miracl-corpus (streaming) until 10k

Writes:
  miracl_eval/<lang>/queries.jsonl   {qid, query}
  miracl_eval/<lang>/qrels.jsonl     {qid, rels: {docid: 0|1}}
  miracl_eval/<lang>/corpus.jsonl    {docid, text}    (~10k per lang)
"""
import argparse
import json
import os
import random
from collections import defaultdict

from datasets import load_dataset


def prep_lang(lang: str, out_root: str, n_queries: int, corpus_size: int, seed: int):
    print(f"\n=== {lang} ===", flush=True)
    out_dir = os.path.join(out_root, lang)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    # 1. Sample queries + collect qrel ids from dev split
    ds = load_dataset("miracl/miracl", lang, split="dev", trust_remote_code=True)
    idxs = rng.sample(range(len(ds)), min(n_queries, len(ds)))
    print(f"  sampling {len(idxs)} / {len(ds)} dev queries", flush=True)

    queries = []
    qrels = defaultdict(dict)
    seeded_corpus = {}
    for i in idxs:
        row = ds[i]
        qid = row["query_id"]
        queries.append({"qid": qid, "query": row["query"]})
        for p in row.get("positive_passages", []):
            seeded_corpus[p["docid"]] = p["text"]
            qrels[qid][p["docid"]] = 1
        for p in row.get("negative_passages", []):
            seeded_corpus[p["docid"]] = p["text"]
            qrels[qid].setdefault(p["docid"], 0)

    # 2. Top up with distractors from miracl-corpus until corpus_size
    needed = corpus_size - len(seeded_corpus)
    print(f"  seeded corpus: {len(seeded_corpus)} docs; need {needed} distractors", flush=True)
    distractors: list[tuple[str, str]] = []
    if needed > 0:
        corpus_ds = load_dataset("miracl/miracl-corpus", lang, split="train",
                                 streaming=True, trust_remote_code=True)
        seen = 0
        for row in corpus_ds:
            did = row["docid"]
            if did in seeded_corpus:
                continue
            seen += 1
            if len(distractors) < needed:
                distractors.append((did, row["text"]))
            else:
                # reservoir replacement
                j = rng.randint(0, seen - 1)
                if j < needed:
                    distractors[j] = (did, row["text"])
            if seen >= max(needed * 20, 500_000):
                break
            if seen % 200_000 == 0:
                print(f"    scanned {seen} candidate docs", flush=True)

    # 3. Write
    with open(os.path.join(out_dir, "queries.jsonl"), "w") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "qrels.jsonl"), "w") as f:
        for qid, rels in qrels.items():
            f.write(json.dumps({"qid": qid, "rels": rels}, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "corpus.jsonl"), "w") as f:
        for did, text in seeded_corpus.items():
            f.write(json.dumps({"docid": did, "text": text}, ensure_ascii=False) + "\n")
        for did, text in distractors:
            f.write(json.dumps({"docid": did, "text": text}, ensure_ascii=False) + "\n")

    total = len(seeded_corpus) + len(distractors)
    n_rel = sum(1 for rels in qrels.values() for v in rels.values() if v > 0)
    print(f"  wrote {len(queries)} queries, {total} docs, {n_rel} positive judgments", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["fr", "ja", "de"])
    p.add_argument("--n-queries", type=int, default=100)
    p.add_argument("--corpus-size", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="miracl_eval")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for lang in args.langs:
        prep_lang(lang, args.out, args.n_queries, args.corpus_size, args.seed)
    print("\nDONE")


if __name__ == "__main__":
    main()
