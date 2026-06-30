"""MLDR evaluation (fr / de / ja / zh).

OOD for both SPLARE and mlv1 (long Wikipedia articles, truncated at MAX_LEN=512
on both sides). Uses native sparse_encoding pipeline with asymmetric Top-K pruning
(prune_ratio=400 at ingest, natural sparsity at query).

Data pre-downloaded by evaluation/prep_mldr.py.

Usage:
  python evaluation/eval_mldr.py \\
    --splare-model-id <SPLARE OS model id> \\
    --mlv1-model-id <mlv1 OS model id>

Final-report MLDR results for SPLARE v3 at 10k corpus, MAX_LEN=512:
  fr  nDCG@10 0.546 vs mlv1 0.420
  de  nDCG@10 0.261 vs mlv1 0.187
  ja  nDCG@10 0.294 vs mlv1 0.233
  zh  nDCG@10 0.258 vs mlv1 0.212
"""
import argparse
import json
import os

from _common import (DEFAULT_OS_URL, DEFAULT_PRUNE_RATIO, bulk_index, make_index,
                     make_pipeline, score_queries)


def run_lang(lang: str, data_dir: str, splare_mid: str, mlv1_mid: str | None,
             os_url: str, prune_ratio: int,
             splare_query_mid: str | None = None) -> dict:
    corpus_path = os.path.join(data_dir, lang, "corpus.jsonl")
    queries = [(json.loads(l)["qid"], json.loads(l)["query"])
               for l in open(os.path.join(data_dir, lang, "queries.jsonl"))]
    qrels = {json.loads(l)["qid"]: json.loads(l)["rels"]
             for l in open(os.path.join(data_dir, lang, "qrels.jsonl"))}
    docs = [json.loads(l) for l in open(corpus_path)]

    print(f"\n=== MLDR {lang}: {len(docs)} docs, {len(queries)} queries ===", flush=True)

    pairs = [("splare", splare_mid, splare_query_mid or splare_mid)]
    if mlv1_mid:
        pairs.append(("mlv1", mlv1_mid, mlv1_mid))

    results: dict[str, dict] = {}
    for tag, ingest_mid, query_mid in pairs:
        pipe = f"pipe-mldr-{lang}-{tag}"
        idx = f"mldr-{lang}-{tag}"
        make_pipeline(pipe, ingest_mid, prune_ratio=prune_ratio, os_url=os_url)
        make_index(idx, pipe, os_url=os_url)
        print(f"  [{tag}] indexing via {ingest_mid}…", flush=True)
        n, errs, t = bulk_index(idx, docs, batch=32 if tag == "splare" else 100, os_url=os_url)
        print(f"  [{tag}] indexed {n} in {t:.0f}s ({errs} batch errors)", flush=True)
        print(f"  [{tag}] scoring via {query_mid}…", flush=True)
        res = score_queries(idx, query_mid, queries, qrels, os_url=os_url)
        print(f"  [{tag}] {res}", flush=True)
        results[tag] = res

    return {"lang": lang, "n_corpus": len(docs), **results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["fr", "de", "ja", "zh"])
    p.add_argument("--data-dir", default="mldr_eval")
    p.add_argument("--splare-model-id", required=True)
    p.add_argument("--splare-query-model-id", default=None,
                   help="Optional separate OS model id for queries (e.g. with top_k=40)")
    p.add_argument("--mlv1-model-id", default=None,
                   help="Optional — omit for splare-only eval")
    p.add_argument("--os-url", default=DEFAULT_OS_URL)
    p.add_argument("--prune-ratio", type=int, default=DEFAULT_PRUNE_RATIO)
    p.add_argument("--out", default="results/mldr_scores.json")
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
