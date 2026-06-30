"""MIRACL dev evaluation (fr / ja / de).

In-distribution for both SPLARE and mlv1. Loads data previously prepped by
evaluation/prep_miracl.py, indexes each language corpus through SPLARE and
mlv1 via native sparse_encoding ingest pipelines (prune_type=top_k, prune_ratio=400),
scores, and writes results JSON.

Usage:
  python evaluation/eval_miracl.py \\
    --splare-model-id <SPLARE OS model id> \\
    --mlv1-model-id <mlv1 OS model id>

Final-report MIRACL results for SPLARE v3 at 10k corpus:
  fr  nDCG@10 0.739 vs mlv1 0.722
  ja  nDCG@10 0.709 vs mlv1 0.622
  de  nDCG@10 0.795 vs mlv1 0.676
"""
import argparse
import json
import os

from _common import (DEFAULT_OS_URL, DEFAULT_PRUNE_RATIO, bulk_index, make_index,
                     make_pipeline, score_queries)


def run_lang(lang: str, data_dir: str, splare_mid: str, mlv1_mid: str | None,
             os_url: str, prune_ratio: int,
             splare_query_mid: str | None = None) -> dict:
    corpus = [json.loads(l) for l in open(os.path.join(data_dir, lang, "corpus.jsonl"))]
    queries = [(json.loads(l)["qid"], json.loads(l)["query"])
               for l in open(os.path.join(data_dir, lang, "queries.jsonl"))]
    qrels = {json.loads(l)["qid"]: json.loads(l)["rels"]
             for l in open(os.path.join(data_dir, lang, "qrels.jsonl"))}

    print(f"\n=== MIRACL {lang}: {len(corpus)} docs, {len(queries)} queries ===", flush=True)

    pairs = [("splare", splare_mid, splare_query_mid or splare_mid)]
    if mlv1_mid:
        pairs.append(("mlv1", mlv1_mid, mlv1_mid))

    results: dict[str, dict] = {}
    for tag, ingest_mid, query_mid in pairs:
        pipe = f"pipe-miracl-{lang}-{tag}"
        idx = f"miracl-{lang}-{tag}"
        make_pipeline(pipe, ingest_mid, prune_ratio=prune_ratio, os_url=os_url)
        make_index(idx, pipe, os_url=os_url)
        print(f"  [{tag}] indexing via {ingest_mid}…", flush=True)
        n, errs, t = bulk_index(idx, corpus, batch=32 if tag == "splare" else 100, os_url=os_url)
        print(f"  [{tag}] indexed {n} in {t:.0f}s ({errs} batch errors)", flush=True)
        print(f"  [{tag}] scoring via {query_mid}…", flush=True)
        res = score_queries(idx, query_mid, queries, qrels, os_url=os_url)
        print(f"  [{tag}] {res}", flush=True)
        results[tag] = res

    return {"lang": lang, "n_corpus": len(corpus), **results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["fr", "ja", "de"])
    p.add_argument("--data-dir", default="miracl_eval")
    p.add_argument("--splare-model-id", required=True)
    p.add_argument("--splare-query-model-id", default=None,
                   help="Optional separate OS model id for queries (e.g. with top_k=40)")
    p.add_argument("--mlv1-model-id", default=None,
                   help="Optional — omit for splare-only eval")
    p.add_argument("--os-url", default=DEFAULT_OS_URL)
    p.add_argument("--prune-ratio", type=int, default=DEFAULT_PRUNE_RATIO)
    p.add_argument("--out", default="results/miracl_scores.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_results = []
    for lang in args.langs:
        all_results.append(run_lang(lang, args.data_dir, args.splare_model_id,
                                    args.mlv1_model_id, args.os_url, args.prune_ratio,
                                    splare_query_mid=args.splare_query_model_id))
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    if args.mlv1_model_id:
        print(f"{'lang':<4}  {'SPLARE':>8}  {'mlv1':>8}  {'Δ':>8}")
        for r in all_results:
            s = r["splare"]["ndcg@10"]; m = r["mlv1"]["ndcg@10"]
            print(f"{r['lang']:<4}  {s:>8.4f}  {m:>8.4f}  {s-m:>+8.4f}")
    else:
        print(f"{'lang':<4}  {'SPLARE':>8}")
        for r in all_results:
            s = r["splare"]["ndcg@10"]
            print(f"{r['lang']:<4}  {s:>8.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
