"""Shared retrieval helpers for evaluation scripts.

OpenSearch helpers: create sparse_encoding ingest pipeline with asymmetric Top-K
pruning (matches production deployment), create index, bulk-index docs, neural_sparse
search, IR metrics (nDCG, MAP, recall, precision, MRR).
"""
import json
import math
import os
import time
from collections import defaultdict
from typing import Iterable

import requests

DEFAULT_OS_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9202")
DEFAULT_PRUNE_RATIO = 400


def make_pipeline(pipeline_name: str, model_id: str,
                  prune_ratio: int = DEFAULT_PRUNE_RATIO,
                  os_url: str = DEFAULT_OS_URL) -> None:
    """Create/overwrite ingest pipeline that runs sparse_encoding with Top-K prune."""
    requests.delete(f"{os_url}/_ingest/pipeline/{pipeline_name}")
    body = {
        "processors": [{
            "sparse_encoding": {
                "model_id": model_id,
                "field_map": {"text": "text_sparse"},
                "prune_type": "top_k",
                "prune_ratio": prune_ratio,
            }
        }]
    }
    r = requests.put(f"{os_url}/_ingest/pipeline/{pipeline_name}", json=body)
    r.raise_for_status()


def make_index(index: str, pipeline: str, os_url: str = DEFAULT_OS_URL) -> None:
    """Create/overwrite index with sparse rank_features field and default pipeline."""
    requests.delete(f"{os_url}/{index}")
    body = {
        "settings": {"default_pipeline": pipeline, "number_of_shards": 1,
                     "refresh_interval": "-1"},
        "mappings": {"properties": {
            "text": {"type": "text"},
            "docid": {"type": "keyword"},
            "text_sparse": {"type": "rank_features"},
        }},
    }
    r = requests.put(f"{os_url}/{index}", json=body)
    r.raise_for_status()


def bulk_index(index: str, docs: Iterable[dict], batch: int = 32,
               os_url: str = DEFAULT_OS_URL, log_every: int = 20) -> tuple[int, int, float]:
    """Index docs (each must have 'docid' and 'text'). Retries batches up to 3×.

    Returns (n_indexed, n_batch_errors, elapsed_s).
    """
    docs = list(docs)
    t0 = time.time()
    errs = 0
    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        lines = []
        for d in chunk:
            lines.append(json.dumps({"index": {"_index": index, "_id": d["docid"]}}))
            lines.append(json.dumps({"text": d["text"], "docid": d["docid"]},
                                    ensure_ascii=False))
        payload = ("\n".join(lines) + "\n").encode()
        ok = False
        for _ in range(3):
            r = requests.post(f"{os_url}/_bulk", data=payload,
                              headers={"Content-Type": "application/x-ndjson"},
                              timeout=600)
            if r.status_code == 200 and not r.json().get("errors"):
                ok = True
                break
            time.sleep(3)
        if not ok:
            errs += 1
        if log_every and (i // batch) % log_every == 0:
            print(f"    {min(i + batch, len(docs))}/{len(docs)}  [{time.time()-t0:.0f}s]",
                  flush=True)
    requests.post(f"{os_url}/{index}/_refresh")
    return len(docs), errs, time.time() - t0


def neural_sparse_search(index: str, model_id: str, query_text: str,
                         k: int = 10, os_url: str = DEFAULT_OS_URL) -> list[str]:
    body = {
        "size": k, "_source": ["docid"], "timeout": "60s",
        "query": {"neural_sparse": {"text_sparse": {
            "query_text": query_text, "model_id": model_id,
        }}},
    }
    try:
        r = requests.post(f"{os_url}/{index}/_search", json=body, timeout=120)
        if r.status_code != 200:
            return []
        return [h["_id"] for h in r.json().get("hits", {}).get("hits", [])]
    except requests.RequestException:
        return []


# ---------- IR metrics ----------

def _dcg(rels: list[int], k: int) -> float:
    return sum((2 ** rels[i] - 1) / math.log2(i + 2) for i in range(min(len(rels), k)))


def ndcg_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    rels = [qrel.get(d, 0) for d in retrieved[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal, k)
    return _dcg(rels, k) / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    return sum(1 for d in retrieved[:k] if qrel.get(d, 0) > 0) / k


def recall_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    total = sum(1 for v in qrel.values() if v > 0)
    if total == 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if qrel.get(d, 0) > 0) / total


def map_at_k(retrieved: list[str], qrel: dict[str, int], k: int) -> float:
    total = min(sum(1 for v in qrel.values() if v > 0), k)
    if total == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, d in enumerate(retrieved[:k]):
        if qrel.get(d, 0) > 0:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / total


def mrr(retrieved: list[str], qrel: dict[str, int]) -> float:
    for i, d in enumerate(retrieved):
        if qrel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def score_queries(index: str, model_id: str, queries: list[tuple[str, str]],
                  qrels: dict[str, dict[str, int]], k: int = 10,
                  os_url: str = DEFAULT_OS_URL) -> dict:
    """Run each query against index and compute mean metrics over qrels-covered queries.

    queries: list of (qid, query_text)
    qrels:   {qid: {docid: grade}}
    """
    m: dict[str, list[float]] = defaultdict(list)
    zero_hits = 0
    for qid, qtext in queries:
        retr = neural_sparse_search(index, model_id, qtext, k=k, os_url=os_url)
        if not retr:
            zero_hits += 1
            continue
        qrel = qrels.get(qid, {})
        if not qrel:
            continue
        m["ndcg@10"].append(ndcg_at_k(retr, qrel, k))
        m["map@10"].append(map_at_k(retr, qrel, k))
        m["recall@10"].append(recall_at_k(retr, qrel, k))
        m["precision@10"].append(precision_at_k(retr, qrel, k))
        m["mrr"].append(mrr(retr, qrel))
    result = {k2: round(sum(v) / len(v), 4) if v else 0 for k2, v in m.items()}
    result["n_queries"] = len(queries)
    result["scored"] = len(m["ndcg@10"])
    result["zero_hits"] = zero_hits
    return result
