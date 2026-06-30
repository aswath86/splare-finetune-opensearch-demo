"""XPQA evaluation (6 configs).

OOD for both SPLARE and mlv1 (e-commerce product QA). Downloads corpus + queries
+ qrels from mteb/XPQARetrieval on the fly (no separate prep step needed).
Uses native sparse_encoding pipeline with asymmetric Top-K pruning.

Usage:
  python evaluation/eval_xpqa.py \\
    --splare-model-id <SPLARE OS model id> \\
    --mlv1-model-id <mlv1 OS model id>

Final-report XPQA results for SPLARE v3:
  fra-fra  SPLARE 0.657 vs mlv1 0.561  (+0.096)   French mono
  jpn-jpn  SPLARE 0.710 vs mlv1 0.591  (+0.119)   Japanese mono
  spa-spa  SPLARE 0.563 vs mlv1 0.478  (+0.085)   Spanish mono
  fra-eng  SPLARE 0.618 vs mlv1 0.324  (+0.294)   English → French
  jpn-eng  SPLARE 0.672 vs mlv1 0.193  (+0.480)   English → Japanese
  deu-eng  SPLARE 0.678 vs mlv1 0.327  (+0.351)   English → German
"""
import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset

from _common import (DEFAULT_OS_URL, DEFAULT_PRUNE_RATIO, bulk_index, make_index,
                     make_pipeline, score_queries)

CONFIGS = [
    ("fra-fra", "French mono"),
    ("jpn-jpn", "Japanese mono"),
    ("spa-spa", "Spanish mono"),
    ("fra-eng", "English → French"),
    ("jpn-eng", "English → Japanese"),
    ("deu-eng", "English → German"),
]


def download(cfg: str):
    q_ds = load_dataset("mteb/XPQARetrieval", f"{cfg}-queries", trust_remote_code=True)["test"]
    c_ds = load_dataset("mteb/XPQARetrieval", f"{cfg}-corpus", trust_remote_code=True)["test"]
    qr_ds = load_dataset("mteb/XPQARetrieval", f"{cfg}-qrels", trust_remote_code=True)["test"]

    queries = [(r["id"], r["text"]) for r in q_ds]
    docs = [{"docid": r["id"], "text": r["text"]} for r in c_ds]
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in qr_ds:
        qrels[r["query-id"]][r["corpus-id"]] = int(r["score"])
    return queries, docs, dict(qrels)


def run_config(cfg: str, label: str, splare_mid: str, mlv1_mid: str | None,
               os_url: str, prune_ratio: int,
               splare_query_mid: str | None = None) -> dict:
    queries, docs, qrels = download(cfg)
    print(f"\n=== XPQA {cfg} ({label}): {len(docs)} docs, {len(queries)} queries ===", flush=True)

    pairs = [("splare", splare_mid, splare_query_mid or splare_mid)]
    if mlv1_mid:
        pairs.append(("mlv1", mlv1_mid, mlv1_mid))

    results: dict[str, dict] = {}
    for tag, ingest_mid, query_mid in pairs:
        pipe = f"pipe-xpqa-{cfg}-{tag}"
        idx = f"xpqa-{cfg}-{tag}"
        make_pipeline(pipe, ingest_mid, prune_ratio=prune_ratio, os_url=os_url)
        make_index(idx, pipe, os_url=os_url)
        print(f"  [{tag}] indexing via {ingest_mid}…", flush=True)
        n, errs, t = bulk_index(idx, docs, batch=32 if tag == "splare" else 100,
                                os_url=os_url, log_every=0)
        print(f"  [{tag}] indexed {n} in {t:.0f}s ({errs} batch errors)", flush=True)
        print(f"  [{tag}] scoring via {query_mid}…", flush=True)
        res = score_queries(idx, query_mid, queries, qrels, os_url=os_url)
        print(f"  [{tag}] {res}", flush=True)
        results[tag] = res

    return {"config": cfg, "label": label, "n_corpus": len(docs), **results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+",
                   default=[c for c, _ in CONFIGS],
                   help="XPQA configs to evaluate")
    p.add_argument("--splare-model-id", required=True,
                   help="SPLARE OS model id used for ingest (and queries, unless --splare-query-model-id is set)")
    p.add_argument("--splare-query-model-id", default=None,
                   help="Optional separate OS model id for queries (e.g. same endpoint but with top_k=40). Defaults to --splare-model-id.")
    p.add_argument("--mlv1-model-id", default=None,
                   help="Optional — pass to compare against mlv1, omit for splare-only")
    p.add_argument("--os-url", default=DEFAULT_OS_URL)
    p.add_argument("--prune-ratio", type=int, default=DEFAULT_PRUNE_RATIO)
    p.add_argument("--out", default="results/xpqa_scores.json")
    args = p.parse_args()

    labels = dict(CONFIGS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    all_results = []
    for cfg in args.configs:
        all_results.append(run_config(cfg, labels.get(cfg, cfg),
                                      args.splare_model_id, args.mlv1_model_id,
                                      args.os_url, args.prune_ratio,
                                      splare_query_mid=args.splare_query_model_id))
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    if args.mlv1_model_id:
        print(f"{'config':<10}  {'label':<22}  {'SPLARE':>8}  {'mlv1':>8}  {'Δ':>8}")
        for r in all_results:
            s = r["splare"]["ndcg@10"]; m = r["mlv1"]["ndcg@10"]
            print(f"{r['config']:<10}  {r['label']:<22}  {s:>8.4f}  {m:>8.4f}  {s-m:>+8.4f}")
    else:
        print(f"{'config':<10}  {'label':<22}  {'SPLARE':>8}")
        for r in all_results:
            s = r["splare"]["ndcg@10"]
            print(f"{r['config']:<10}  {r['label']:<22}  {s:>8.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
