#!/usr/bin/env python3
"""
Build leakage-free training pairs from legal_raw.tsv.

For every document (split by title into train/val/test before pairing):
- 1 positive: (title, its own passage), plus any extra queries given with --queries;
- n hard negatives: passages BM25 ranks highest for the query (another document);
- n random negatives: passages from another jurisdiction when possible.

Writes data/contrastive_{train,val,test}.jsonl.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoprune.training.data import build_pairs, read_legal_tsv, write_jsonl  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", default="data/legal_raw.tsv")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--hard", type=int, default=1, help="BM25 hard negatives per query")
    parser.add_argument("--random", type=int, default=1, help="random negatives per query")
    parser.add_argument("--hard-pool", type=int, default=10, help="BM25 ranks sampled from for hard negatives")
    parser.add_argument("--queries", default=None,
                        help="optional JSONL {\"doc_id\": ..., \"queries\": [...]} with real search queries")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    records = read_legal_tsv(args.tsv)
    print(f"Loaded {len(records)} unique records from {args.tsv}")
    extra = None
    if args.queries:
        with open(args.queries, "r", encoding="utf-8") as handle:
            extra = {row["doc_id"]: row["queries"] for row in map(json.loads, handle) if row.strip()}
        print(f"Extra queries for {len(extra)} documents")

    splits = build_pairs(records, n_hard=args.hard, n_random=args.random, hard_pool=args.hard_pool,
                         seed=args.seed, extra_queries=extra)
    for split, rows in splits.items():
        path = Path(args.out_dir) / f"contrastive_{split}.jsonl"
        write_jsonl(rows, path)
        types = Counter(r["type"] for r in rows)
        print(f"{split:5s}: {len(rows):6d} pairs -> {path} {dict(types)}")

    titles = {split: {r["query"] for r in rows} for split, rows in splits.items()}
    overlap = (titles["train"] & titles["val"]) | (titles["train"] & titles["test"])
    print(f"Queries shared between train and val/test: {len(overlap)} (expected 0)")


if __name__ == "__main__":
    main()
