#!/usr/bin/env python3
"""
NanoPrune v0.3.0 Knowledge Distillation Pipeline.
Uses Laya (ModernBERT-large 421M System 1 model) as Teacher to generate
continuous calibrated soft labels and multi-primitive annotations on legal data.
"""
import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure laya is importable
import laya

SEED = 42
random.seed(SEED)

def generate_distillation_data(
    input_tsv: str = "data/legal_raw.tsv",
    output_jsonl: str = "data/distilled_pairs.jsonl",
    max_pairs: int = 2000,
):
    print("=" * 60)
    print("🎓 NanoPrune v0.3 Knowledge Distillation (Teacher: Laya 421M)")
    print("=" * 60)

    tsv_path = Path(input_tsv)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Input file not found: {tsv_path}")

    print(f"📖 Lecture des extraits juridiques depuis {input_tsv}...")
    records = []
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if r"\t" in line:
                parts = line.strip().split(r"\t")
            else:
                parts = line.strip().split("\t")
            if len(parts) >= 3:
                jurisdiction, title, content = parts[0].strip(), parts[1].strip(), parts[2].strip()
                if len(title) > 5 and len(content) > 50:
                    records.append({
                        "jurisdiction": jurisdiction,
                        "title": title,
                        "content": content[:600],
                    })

    print(f"   {len(records)} textes juridiques valides chargés.")
    if len(records) < 100:
        raise ValueError("Trop peu d'enregistrements pour la distillation.")

    # Construction des paires (requête, extrait)
    candidate_pairs = []
    n_pos = min(len(records), max_pairs // 2)

    for i in range(n_pos):
        rec = records[i]
        # Requête positive naturelle basée sur le titre / juridiction
        query_pos = f"{rec['title'][:80]}"
        candidate_pairs.append({
            "query": query_pos,
            "content": rec["content"],
            "expected_type": "positive",
            "jurisdiction": rec["jurisdiction"]
        })

        # Hard negative : la même requête confrontée à un texte d'une autre juridiction
        neg_rec = records[(i + 413) % len(records)]
        candidate_pairs.append({
            "query": query_pos,
            "content": neg_rec["content"],
            "expected_type": "negative",
            "jurisdiction": neg_rec["jurisdiction"]
        })

    random.shuffle(candidate_pairs)
    candidate_pairs = candidate_pairs[:max_pairs]
    print(f"   {len(candidate_pairs)} paires prêtes pour l'évaluation par Laya.")

    # Chargement du Teacher Laya
    print("\n🤖 Chargement du Teacher Laya (Apple Silicon MPS)...")
    t_start_load = time.time()
    agent = laya.load("convaiinnovations/laya")
    print(f"   Teacher prêt en {time.time() - t_start_load:.2f}s sur {agent.device} !")

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n⚡ Inférence & distillation continue sur {len(candidate_pairs)} paires...")
    t_distill_start = time.time()

    distilled_count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, item in enumerate(candidate_pairs):
            query = item["query"]
            content = item["content"]

            state = f"Document: {content}"
            questions = {
                "relevance": {
                    "type": "noul",
                    "instructions": f"Ce document juridique apporte-t-il une réponse pertinente à la recherche : '{query}' ?",
                },
                "category": {
                    "type": "choice",
                    "instructions": "Quelle est la matière juridique principale de cet extrait ?",
                    "criteria": {
                        "human_rights": "Droits fondamentaux, libertés, CEDH ou constitutionnel",
                        "business_tax": "Droit commercial, fiscalité, contrats d'affaires, finances",
                        "public_admin": "Règlements administratifs, nominations, arrêtés communaux",
                        "civil_family": "Droit civil, responsabilité, personnes ou succession",
                    }
                },
                "complexity": {
                    "type": "score",
                    "instructions": "Niveau de complexité / portée juridique de la clause",
                    "criteria": [
                        "Niveau 0 : Simple arrêté ou formalité",
                        "Niveau 1 : Règle standard",
                        "Niveau 2 : Portée importante ou contentieuse",
                        "Niveau 3 : Arrêt de principe ou question constitutionnelle"
                    ]
                }
            }

            try:
                res = agent.system_one(state, questions)
                answers = res["answers"]

                # Récupération des probabilités continues (soft targets)
                soft_rel = float(answers["relevance"]["noul"])
                conf_rel = float(answers["relevance"]["confidence"])

                choice_ans = answers["category"]["choice"]
                cat_probs = answers["category"]["probabilities"]
                categories = ["human_rights", "business_tax", "public_admin", "civil_family"]
                cat_idx = categories.index(choice_ans) if choice_ans in categories else 0

                score_val = float(answers["complexity"]["score"])

                distilled_entry = {
                    "query": query,
                    "content": content,
                    "soft_relevance": round(soft_rel, 4),
                    "confidence": round(conf_rel, 4),
                    "category": choice_ans,
                    "category_idx": cat_idx,
                    "category_probs": [round(cat_probs.get(c, 0.0), 4) for c in categories],
                    "complexity_score": round(score_val, 3),
                }

                out_f.write(json.dumps(distilled_entry, ensure_ascii=False) + "\n")
                distilled_count += 1

            except Exception as e:
                # Fallback graceful en cas d'erreur de parsing ponctuelle
                continue

            if (idx + 1) % 100 == 0 or (idx + 1) == len(candidate_pairs):
                elapsed = time.time() - t_distill_start
                rate = (idx + 1) / elapsed
                rem = (len(candidate_pairs) - (idx + 1)) / max(1e-3, rate)
                print(f"   [{idx + 1}/{len(candidate_pairs)}] {rate:.1f} paires/s | Écoulé: {elapsed:.1f}s | Restant: {rem:.1f}s")

    t_total = time.time() - t_distill_start
    print(f"\n🎉 Distillation terminée avec succès !")
    print(f"   Total généré : {distilled_count} paires distillées enregistrées dans {output_path}")
    print(f"   Temps total : {t_total:.1f}s (~{t_total/distilled_count*1000:.1f} ms/paire)")

if __name__ == "__main__":
    tsv_in = sys.argv[1] if len(sys.argv) > 1 else "data/legal_raw.tsv"
    jsonl_out = sys.argv[2] if len(sys.argv) > 2 else "data/distilled_pairs.jsonl"
    n_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    generate_distillation_data(tsv_in, jsonl_out, n_pairs)
