#!/usr/bin/env python3
"""
Training and Distillation script for NanoPrune System One decision model.
Produces calibrated ~2.8MB ONNX weights.
"""
import os
import sys
import random
import time
from pathlib import Path
from typing import List, Tuple

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nanoprune.core.tokenizer import NanoTokenizer
from nanoprune.core.model import NanoPruneModel
from nanoprune.core.calibration import compute_ece, TemperatureScaler
from nanoprune.core.export import export_to_onnx

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class RAGDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, float]], tokenizer: NanoTokenizer, max_len: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, context, label = self.pairs[idx]
        input_ids, mask = self.tokenizer.encode_pair(query, context, max_length=self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }

def generate_multi_domain_rag_pairs() -> List[Tuple[str, str, float]]:
    """
    Generates realistic, calibrated RAG query-context pairs across
    medical, legal, technical, business, and negation scenarios.
    """
    data: List[Tuple[str, str, float]] = []

    # 1. Medical Domain
    med_queries = [
        ("Allergie pénicilline", "Allergie sévère documentée aux pénicillines et amoxicilline (choc anaphylactique).", 1.0),
        ("Allergie pénicilline", "Aucune allergie connue aux pénicillines ou aux antibiotiques.", 0.15),
        ("Allergie pénicilline", "Traitement par ramipril 5mg et aspirine 75mg le matin.", 0.02),
        ("Contre-indication AINS", "Contre-indication formelle aux anti-inflammatoires non stéroïdiens (AINS) pour antécédent d'ulcère.", 1.0),
        ("Contre-indication AINS", "Le patient tolère parfaitement l'ibuprofène 400mg ponctuellement.", 0.10),
        ("Posologie paracétamol", "Prescription de paracétamol 1g : 1 comprimé toutes les 6 heures, maximum 3g par jour.", 1.0),
        ("Posologie paracétamol", "Prescription d'amoxicilline 1g matin et soir pendant 6 jours consécutifs.", 0.05),
        ("Bilan cardiaque stent", "Pose d'un stent coronarien actif en 2022 avec suivi cardiologique semestriel normal.", 1.0),
        ("Bilan cardiaque stent", "Radiographie pulmonaire sans anomalie pleuro-parenchymateuse décelable.", 0.05),
    ]

    # 2. Legal / Contract Domain
    legal_queries = [
        ("Clause non-concurrence durée", "La présente clause de non-concurrence est fixée pour une durée stricte de 12 mois renouvelable.", 1.0),
        ("Clause non-concurrence durée", "Le contrat de bail commercial est conclu pour une durée de 9 années entières.", 0.05),
        ("Délai de préavis démission", "En cas de démission du salarié, la durée du préavis est fixée à 3 mois calendaires.", 1.0),
        ("Délai de préavis démission", "Les congés payés doivent être posés avec un accord préalable de la direction.", 0.05),
        ("Indemnité de rupture", "L'indemnité forfaitaire de rupture conventionnelle s'élève à 15 000 euros net.", 1.0),
        ("Indemnité de rupture", "Le montant des frais de déplacement est remboursé sur présentation des justificatifs.", 0.02),
    ]

    # 3. Technical & Agent RAG Domain
    tech_queries = [
        ("Port PostgreSQL base de données", "La base de données PostgreSQL écoute sur le port standard 5432 avec SSL activé.", 1.0),
        ("Port PostgreSQL base de données", "Le serveur Nginx écoute sur le port 80 pour HTTP et 443 pour HTTPS.", 0.05),
        ("Code retrait préventif purée tomates", "Le code interne officiel de retrait préventif du lot de purée de tomates est RET-8841-BIO.", 1.0),
        ("Code retrait préventif purée tomates", "La procédure d'inventaire hebdomadaire du rayon épicerie a été validée hier.", 0.02),
        ("Code congélateur Liège", "L'identifiant d'accès sécurisé au congélateur froid négatif de Liège est FRZ-04-LIEGE.", 1.0),
        ("Code congélateur Liège", "La livraison de surgelés à Namur s'effectue tous les mardis à 6h00.", 0.05),
    ]

    base_samples = med_queries + legal_queries + tech_queries

    # Expand with variations and hard negatives (synonyms, permutations)
    for q, c, label in base_samples:
        data.append((q, c, label))
        # Hard negative by swapping unrelated context
        for other_q, other_c, _ in base_samples:
            if other_q != q:
                data.append((q, other_c, 0.02))

    # Synthetic scaling: 2000 pairs with varied noise
    expanded_data = []
    for _ in range(15):
        for q, c, label in data:
            # Inject slight context variations
            noise_prefix = random.choice([
                "Rappel dossier archivé : ",
                "Note interne : ",
                "Historique des fiches : ",
                "Extrait conforme : ",
                "",
            ])
            expanded_data.append((q, f"{noise_prefix}{c}", label))

    random.shuffle(expanded_data)
    return expanded_data

def train_nanoprune(
    output_dir: str = "models",
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("⚡ Initialisation du Tokenizer...")
    tokenizer = NanoTokenizer()
    print(f"   Vocabulaire : {tokenizer.vocab_size} tokens")

    print("⚡ Génération du corpus RAG calibré...")
    all_pairs = generate_multi_domain_rag_pairs()
    split_idx = int(len(all_pairs) * 0.85)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    print(f"   Train : {len(train_pairs)} paires | Val : {len(val_pairs)} paires")

    train_ds = RAGDataset(train_pairs, tokenizer)
    val_ds = RAGDataset(val_pairs, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device d'entraînement : {device}")

    model = NanoPruneModel(
        vocab_size=tokenizer.vocab_size + 200,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=2,
        max_seq_len=256,
    ).to(device)

    param_count = model.count_parameters()
    print(f"⚡ Paramètres du modèle : {param_count:,} (~{param_count * 4 / 1024 / 1024:.2f} MB en FP32)")

    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n🚀 Démarrage de l'entraînement ({epochs} époques)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["label"].unsqueeze(-1).to(device)

            optimizer.zero_grad()
            logits, probs = model(ids, mask)

            # Combined Loss: BCE + Brier Score
            loss_bce = bce_loss_fn(logits, targets)
            loss_brier = torch.mean((probs - targets) ** 2)
            total_loss = 0.7 * loss_bce + 0.3 * loss_brier

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss.item() * len(ids)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                targets = batch["label"]
                _, probs = model(ids, mask)
                val_probs.extend(probs.squeeze(-1).cpu().numpy())
                val_labels.extend(targets.numpy())

        cal_metrics = compute_ece(np.array(val_probs), np.array(val_labels))
        print(
            f"Époque {epoch:02d}/{epochs:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Brier: {cal_metrics['brier_score']:.4f} | "
            f"ECE: {cal_metrics['ece']*100:.2f}%"
        )

    elapsed = time.time() - start_time
    print(f"\n✅ Entraînement terminé en {elapsed:.1f} secondes.")

    # Save PyTorch weights
    pt_path = out_path / "nanoprune-v0.1.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"📦 Poids PyTorch sauvegardés : {pt_path} ({pt_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Export to ONNX
    onnx_path = out_path / "nanoprune-v0.1.onnx"
    print("⚡ Export vers le format ONNX...")
    model.cpu()
    exported = export_to_onnx(model, onnx_path, seq_len=256, quantize=True)
    final_file = exported if exported.exists() else onnx_path
    print(f"🏆 Modèle officiel exporté : {final_file} ({final_file.stat().st_size / 1024 / 1024:.2f} MB)")

    return final_file

if __name__ == "__main__":
    train_nanoprune()
