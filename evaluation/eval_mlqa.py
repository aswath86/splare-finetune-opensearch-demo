"""MLQA evaluation — 49 configs (7 langs × 7 langs).

Each (corpus_lang, query_lang) config has its own corpus with config-specific
doc IDs, so we create 49 × 2 = 98 indices. Downloads corpus/queries/qrels on
the fly from mteb/MLQARetrieval. Uses native sparse_encoding pipeline with
asymmetric Top-K pruning (prune_ratio=400).

Writes incremental results to the output JSON after every config so interrupted
runs can be resumed. Re-running skips already-scored configs.

Usage:
  python evaluation/eval_mlqa.py \\
    --splare-model-id <SPLARE OS model id> \\
    --mlv1-model-id <mlv1 OS model id>

Final-report MLQA averages for SPLARE v3:
  Monolingual (7 configs): SPLARE 0.648  vs mlv1 0.472  (+0.176)
  Cross-lingual (42):      SPLARE 0.627  vs mlv1 0.225  (+0.402)
  Overall (49):            SPLARE 0.630  vs mlv1 0.261  (+0.369)
"""
import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset

from _common import (DEFAULT_OS_URL, DEFAULT_PRUNE_RATIO, bulk_index, make_index,
                     make_pipeline, neural_sparse_search, ndcg_at_k, recall_at_k)

LANGS = ["ara", "deu", "eng", "hin", "spa", "vie", "zho"]


def _score_chunk(idx, mid, chunk, qrels, os_url):
    m: dict[str, list[float]] = defaultdict(list)
    z = 0
    for qid, qt in chunk:
        retr = neural_sparse_search(idx, mid, qt, k=10, os_url=os_url)
        if not retr:
            z += 1
            continue
        qrel = qrels.get(qid, {})
        if not qrel:
            continue
        m["ndcg@10"].append(ndcg_at_k(retr, qrel, 10))
        m["recall@10"].append(recall_at_k(retr, qrel, 10))
    return m, z


def score_parallel(idx: str, mid: str, queries: list[tuple[str, str]],
                   qrels: dict[str, dict[str, int]], workers: int,
                   os_url: str) -> dict:
    chunk_size = max(1, len(queries) // workers)
    chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]
    all_m: dict[str, list[float]] = defaultdict(list)
    all_z = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_score_chunk, idx, mid, c, qrels, os_url) for c in chunks]
        for fut in as_completed(futs):
            m, z = fut.result()
            for k, v in m.items():
                all_m[k].extend(v)
            all_z += z
    return {
        **{k: round(sum(v) / len(v), 4) if v else 0 for k, v in all_m.items()},
        "n": len(queries), "zero": all_z, "scored": len(all_m["ndcg@10"]),
        "score_time_s": round(time.time() - t0, 1),
    }


def process_config(c_lang: str, q_lang: str, splare_mid: str, mlv1_mid: str,
                   splare_pipe: str, mlv1_pipe: str, workers: int,
                   os_url: str) -> dict:
    cfg = f"{c_lang}-{q_lang}"
    c_ds = load_dataset("mteb/MLQARetrieval", f"{cfg}-corpus", trust_remote_code=True)["test"]
    q_ds = load_dataset("mteb/MLQARetrieval", f"{cfg}-queries", trust_remote_code=True)["test"]
    qr_ds = load_dataset("mteb/MLQARetrieval", f"{cfg}-qrels", trust_remote_code=True)["test"]

    docs = [{"docid": r["id"], "text": r["text"]} for r in c_ds]
    queries = [(r["id"], r["text"]) for r in q_ds]
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for r in qr_ds:
        qrels[r["query-id"]][r["corpus-id"]] = int(r["score"])
    qrels = dict(qrels)

    result = {"n_corpus": len(docs), "n_queries": len(queries)}
    t0 = time.time()

    for tag, mid, pipe in [("splare", splare_mid, splare_pipe),
                           ("mlv1", mlv1_mid, mlv1_pipe)]:
        idx = f"mlqa-{cfg}-{tag}"
        make_index(idx, pipe, os_url=os_url)
        ti = time.time()
        _, errs, _ = bulk_index(idx, docs, batch=32, os_url=os_url, log_every=0)
        idx_time = time.time() - ti
        r = score_parallel(idx, mid, queries, qrels, workers, os_url)
        r["idx_time_s"] = round(idx_time, 1)
        r["idx_errs"] = errs
        result[tag] = r

    result["total_time_s"] = round(time.time() - t0, 1)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=LANGS)
    p.add_argument("--splare-model-id", required=True)
    p.add_argument("--mlv1-model-id", required=True)
    p.add_argument("--os-url", default=DEFAULT_OS_URL)
    p.add_argument("--prune-ratio", type=int, default=DEFAULT_PRUNE_RATIO)
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel query workers per config")
    p.add_argument("--out", default="results/mlqa_scores.json")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip configs already scored in the output file (default on)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    splare_pipe = "pipe-mlqa-splare-400"
    mlv1_pipe = "pipe-mlqa-mlv1-400"
    make_pipeline(splare_pipe, args.splare_model_id, prune_ratio=args.prune_ratio,
                  os_url=args.os_url)
    make_pipeline(mlv1_pipe, args.mlv1_model_id, prune_ratio=args.prune_ratio,
                  os_url=args.os_url)

    # Resume support
    results: dict[str, dict] = {}
    if args.skip_existing and os.path.exists(args.out):
        try:
            results = json.load(open(args.out))
        except json.JSONDecodeError:
            results = {}

    print(f"{'cfg':<10} {'SPLARE':>8} {'mlv1':>8} {'Δ':>8} {'q':>5} {'c':>5} {'t':>6}",
          flush=True)
    for cl in args.langs:
        for ql in args.langs:
            cfg = f"{cl}-{ql}"
            if args.skip_existing and results.get(cfg, {}).get("splare", {}).get("ndcg@10", 0):
                existing = results[cfg]
                s = existing["splare"]["ndcg@10"]
                m = existing["mlv1"]["ndcg@10"]
                print(f"{cfg:<10} {s:>8.4f} {m:>8.4f} {s-m:>+8.4f} "
                      f"{existing['n_queries']:>5} {existing['n_corpus']:>5} "
                      f"(cached)", flush=True)
                continue

            res = process_config(cl, ql, args.splare_model_id, args.mlv1_model_id,
                                 splare_pipe, mlv1_pipe, args.workers, args.os_url)
            results[cfg] = res
            s = res["splare"].get("ndcg@10", 0)
            m = res["mlv1"].get("ndcg@10", 0)
            print(f"{cfg:<10} {s:>8.4f} {m:>8.4f} {s-m:>+8.4f} "
                  f"{res['n_queries']:>5} {res['n_corpus']:>5} {res['total_time_s']:>6.0f}",
                  flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary: mono vs cross
    mono = [r for cfg, r in results.items() if cfg.split("-")[0] == cfg.split("-")[1]]
    cross = [r for cfg, r in results.items() if cfg.split("-")[0] != cfg.split("-")[1]]

    def avg(rs, tag):
        vals = [r[tag].get("ndcg@10", 0) for r in rs if r.get(tag, {}).get("ndcg@10", 0) > 0]
        return sum(vals) / len(vals) if vals else 0

    print("\n=== SUMMARY ===")
    print(f"Monolingual ({len(mono)}):   SPLARE {avg(mono,'splare'):.4f}  "
          f"mlv1 {avg(mono,'mlv1'):.4f}  Δ {avg(mono,'splare')-avg(mono,'mlv1'):+.4f}")
    print(f"Cross-lingual ({len(cross)}): SPLARE {avg(cross,'splare'):.4f}  "
          f"mlv1 {avg(cross,'mlv1'):.4f}  Δ {avg(cross,'splare')-avg(cross,'mlv1'):+.4f}")
    allr = list(results.values())
    print(f"All ({len(allr)}):            SPLARE {avg(allr,'splare'):.4f}  "
          f"mlv1 {avg(allr,'mlv1'):.4f}  Δ {avg(allr,'splare')-avg(allr,'mlv1'):+.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
