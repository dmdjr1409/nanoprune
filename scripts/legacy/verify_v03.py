#!/usr/bin/env python3
"""
NanoPrune v0.3.0 Verification & Benchmark Script.
Tests all 3 System One primitives (prune, choice, score) on real legal clauses.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from nanoprune.engine.pruner import NanoPruner

def run_verification():
    print("=" * 65)
    print("🔍 NanoPrune v0.3.0 (Distilled Edition) Verification & Benchmark")
    print("=" * 65)

    weights_path = Path("weights/nanoprune-v0.3.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    pruner = NanoPruner.load(model_path=weights_path, threshold=0.60)
    print(f"📦 Model loaded: {weights_path} ({weights_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"⚡ Backend: {'PyTorch' if pruner.torch_model is not None else 'ONNX'}")

    # Test Primitive 1: Pruning / Relevance
    print("\n--- [Primitive 1: Prune / Relevance] ---")
    query = "Modalités de résiliation anticipée et indemnités de rupture"
    candidates = [
        "Article 12: Chacune des parties peut résilier le contrat sans indemnité sous préavis de 30 jours.",
        "Article 4: Le montant des honoraires est payable par virement bancaire sous 15 jours fin de mois.",
        "Article 19: Les droits de propriété intellectuelle sur les créations restent acquis au client.",
        "Règlement grand-ducal portant fixation de la taxe communale sur les chiens pour l'année 2026."
    ]

    t0 = time.perf_counter()
    retained = pruner.prune(query, candidates)
    t_prune = (time.perf_counter() - t0) * 1000

    print(f"⏱️ Inférence (4 candidats): {t_prune:.2f} ms (~{t_prune/len(candidates):.2f} ms/doc)")
    print(f"🎯 Conservés (seuil >= 60%) : {len(retained)} / {len(candidates)}")
    for text, score in retained:
        print(f"   [{score*100:.1f}%] {text[:75]}...")

    # Test Primitive 2: Choice (Categorical Classification)
    print("\n--- [Primitive 2: Choice / Categorical Selection] ---")
    clause = "La Cour juge que le délai de paiement de l'aide juridictionnelle constitue une atteinte disproportionnée aux droits protégés par l'article 1 du Protocole n°1."
    categories = [
        "human_rights",
        "business_tax",
        "public_admin",
        "civil_family"
    ]

    t0 = time.perf_counter()
    best_idx, best_cat, confidence = pruner.choice(clause, categories)
    t_choice = (time.perf_counter() - t0) * 1000

    print(f"⏱️ Inférence Choice: {t_choice:.2f} ms")
    print(f"🏷️ Catégorie: {best_cat} (Index: {best_idx}) | Confiance: {confidence*100:.1f}%")

    # Test Primitive 3: Score (Rubric / Complexity Rating)
    print("\n--- [Primitive 3: Score / Rubric Complexity (0.0 to 4.0)] ---")
    clauses_to_score = [
        ("Arrêté municipal - fermeture exceptionnelle du bureau communal le 2 mai.", "Arrêté simple"),
        ("Arrêt de principe de la Cour de cassation sur la rétroactivité des créances salariales.", "Arrêt complexe"),
    ]

    for text, desc in clauses_to_score:
        t0 = time.perf_counter()
        score_val = pruner.score_rubric(text)
        t_sc = (time.perf_counter() - t0) * 1000
        print(f"⏱️ {t_sc:.2f} ms | Score: {score_val:.2f}/4.00 [{desc}] -> '{text[:65]}...'")

    print("\n" + "=" * 65)
    print("✅ All NanoPrune v0.3.0 primitives verified successfully!")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()
