"""
Step 3: prepare multilingual training data.

Loads MIRACL + Mr.TyDi subsets of bge-multilingual-gemma2-data, keeps only
the fields we need (query, pos, neg, teacher scores), subsamples hard negatives
to NEGATIVES_PER_Q, writes one JSONL with one training example per line.

Output schema (one JSON per line):
  { "query": str,
    "pos": str,
    "pos_score": float,
    "negs": [str, ...],
    "neg_scores": [float, ...] }
"""
import argparse, json, os, random
from datasets import load_dataset, get_dataset_split_names
import config as C


def normalize_row(row, n_negs: int):
    pos_list = row["pos"]
    pos_scores = row["pos_scores"]
    if not pos_list or not pos_scores:
        return None
    pos = pos_list[0]
    pos_score = float(pos_scores[0])

    neg_list = row["neg"] or []
    neg_scores = row["neg_scores"] or []
    if len(neg_list) < n_negs or len(neg_scores) < n_negs:
        return None
    # deterministic: take top-k by score-distance from pos isn't needed;
    # paper uses random sample from the hard-neg pool
    idx = random.sample(range(min(len(neg_list), len(neg_scores))), n_negs)
    negs = [neg_list[i] for i in idx]
    neg_scs = [float(neg_scores[i]) for i in idx]
    return {"query": row["query"], "pos": pos, "pos_score": pos_score,
            "negs": negs, "neg_scores": neg_scs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/train_multilingual.jsonl")
    p.add_argument("--max-per-config", type=int, default=None,
                   help="Cap rows per config (for quick iteration)")
    p.add_argument("--seed", type=int, default=37)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total_written = total_skipped = 0
    with open(args.output, "w") as f:
        for cfg in C.DATA_CONFIGS:
            splits = get_dataset_split_names(C.DATA_REPO, cfg)
            print(f"Loading {C.DATA_REPO} / {cfg}  (splits: {splits})")
            for split in splits:
                ds = load_dataset(C.DATA_REPO, cfg, split=split)
                rows = ds if args.max_per_config is None else ds.select(range(min(args.max_per_config, len(ds))))
                written = skipped = 0
                for row in rows:
                    sample = normalize_row(row, C.NEGATIVES_PER_Q)
                    if sample is None:
                        skipped += 1
                        continue
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
                print(f"  {cfg}/{split}: wrote {written}, skipped {skipped}")
                total_written += written
                total_skipped += skipped

    print(f"Done. Total: wrote {total_written}, skipped {total_skipped}  ->  {args.output}")


if __name__ == "__main__":
    main()
