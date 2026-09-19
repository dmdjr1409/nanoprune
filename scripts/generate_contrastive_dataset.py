#!/usr/bin/env python3
"""
Generate High-Contrast Hard Negative Training Dataset from legal_raw.tsv.
Pairs:
- 2,000 High-Confidence Positives (label = 1.0)
- 1,500 Hard Negatives (lexical trap overlap, label = 0.0)
- 1,000 Cross-Domain Negatives (label = 0.0)
Total: 4,500 pairs with hard contrastive targets [0.0, 1.0].
"""
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any

SEED = 42
random.seed(SEED)

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_contrastive_pairs(
    tsv_path: str = "data/legal_raw.tsv",
    output_jsonl: str = "data/contrastive_pairs.jsonl",
    max_pairs: int = 4500
):
    print("=" * 60)
    print("🎯 Building High-Contrast Contrastive Dataset")
    print("=" * 60)

    records = []
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t") if "\t" in line else line.strip().split(r"\t")
            if len(parts) >= 3:
                jurisdiction = clean_text(parts[0])
                title = clean_text(parts[1])
                content = clean_text(parts[2])
                if len(title) > 5 and len(content) > 50:
                    records.append({
                        "jurisdiction": jurisdiction,
                        "title": title,
                        "content": content[:500]
                    })

    print(f"Loaded {len(records)} clean records from {tsv_path}.")

    pairs = []

    # Common legal anchor keywords to construct hard negative traps
    traps = ["indemnité", "tribunal", "recours", "délai", "responsabilité", "procédure", "article", "ministériel"]

    # 1. Positives
    n_pos = min(len(records), 2000)
    for i in range(n_pos):
        rec = records[i]
        query = f"{rec['title']}"
        if len(query) > 90:
            query = query[:90]

        pairs.append({
            "query": query,
            "content": rec["content"],
            "label": 1.0,
            "type": "positive",
            "category": 0 if "droits" in rec["jurisdiction"].lower() else (1 if "commerce" in rec["content"].lower() else 2)
        })

    # 2. Hard Negatives (lexical overlap trap)
    for i in range(1500):
        rec_a = records[i % len(records)]
        # Find another record sharing at least one anchor word
        query_words = set(rec_a["title"].lower().split())
        candidate_neg = None
        for offset in range(1, 50):
            neg_cand = records[(i + offset * 17) % len(records)]
            neg_words = set(neg_cand["content"].lower().split())
            overlap = query_words.intersection(neg_words)
            if overlap and neg_cand["title"] != rec_a["title"]:
                candidate_neg = neg_cand
                break

        if candidate_neg is None:
            candidate_neg = records[(i + 311) % len(records)]

        query = f"{rec_a['title']}"
        if len(query) > 90:
            query = query[:90]

        pairs.append({
            "query": query,
            "content": candidate_neg["content"],
            "label": 0.0,
            "type": "hard_negative",
            "category": 0
        })

    # 3. Cross-Domain Negatives
    for i in range(1000):
        rec_a = records[i % len(records)]
        rec_b = records[(i + 777) % len(records)]
        query = f"{rec_a['title']}"
        if len(query) > 90:
            query = query[:90]

        pairs.append({
            "query": query,
            "content": rec_b["content"],
            "label": 0.0,
            "type": "cross_negative",
            "category": 3
        })

    random.shuffle(pairs)
    pairs = pairs[:max_pairs]

    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(pairs)} contrastive pairs to {out_file}")
    pos_cnt = sum(1 for p in pairs if p["label"] == 1.0)
    print(f"   Positives: {pos_cnt} ({pos_cnt/len(pairs)*100:.1f}%) | Negatives: {len(pairs)-pos_cnt}")

if __name__ == "__main__":
    build_contrastive_pairs()
