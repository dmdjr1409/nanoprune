#!/usr/bin/env python3
"""
NanoPrune Legal Edition (v0.2) Training Script.
Distills real-world legal codes, court decisions, and contracts into a calibrated 2.8MB model.
"""
import os
import sys
import csv
import random
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nanoprune.core.tokenizer import NanoTokenizer
from nanoprune.core.model import NanoPruneModel
from nanoprune.core.calibration import compute_ece
from nanoprune.core.export import export_to_onnx

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class LegalRAGDataset(Dataset):
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

def load_legal_pairs(tsv_path: str) -> List[Tuple[str, str, float]]:
    print(f"📖 Chargement des extraits juridiques depuis {tsv_path}...")
    rows = []
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if len(line) >= 2:
                title = line[0].strip()
                content = line[1].strip()
                if len(title) > 5 and len(content) > 50:
                    rows.append((title, content))

    print(f"   {len(rows)} articles de loi et décisions chargés.")

    pairs: List[Tuple[str, str, float]] = []
    all_contents = [r[1] for r in rows]

    # Génération des paires positives et hard negatives
    for idx, (title, content) in enumerate(rows):
        # Paire positive
        pairs.append((title[:90], content, 1.0))

        # Paire négative : le même titre juridique avec un texte de loi complètement différent
        neg_content = all_contents[(idx + 777) % len(all_contents)]
        pairs.append((title[:90], neg_content, 0.01))

    random.shuffle(pairs)
    print(f"   {len(pairs)} paires d'entraînement équilibrées créées !")
    return pairs

def train_legal_edition(
    tsv_path: str = "/tmp/legal_sample.tsv",
    output_dir: str = "models",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = NanoTokenizer.char_level()  # v0.1-v0.3 used the char vocabulary
    pairs = load_legal_pairs(tsv_path)

    split = int(len(pairs) * 0.85)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]

    train_ds = LegalRAGDataset(train_pairs, tokenizer)
    val_ds = LegalRAGDataset(val_pairs, tokenizer)
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
        head_hidden=64,  # v0.1-v0.3 heads had 64 hidden units
    ).to(device)

    param_count = model.count_parameters()
    print(f"⚡ Paramètres du modèle : {param_count:,} (~{param_count * 4 / 1024 / 1024:.2f} MB en FP32)")

    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n⚖️ Démarrage de l'entraînement NanoPrune Legal Edition ({epochs} époques)...")
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["label"].unsqueeze(-1).to(device)

            optimizer.zero_grad()
            logits, probs = model(ids, mask)

            loss = 0.7 * bce_loss_fn(logits, targets) + 0.3 * torch.mean((probs - targets) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(ids)

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

        metrics = compute_ece(np.array(val_probs), np.array(val_labels))
        print(f"Époque {epoch:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Brier: {metrics['brier_score']:.4f} | ECE: {metrics['ece']*100:.2f}%")

    elapsed = time.time() - start
    print(f"\n✅ Entraînement juridique terminé en {elapsed:.1f} secondes !")

    # Sauvegarde PT
    pt_path = out_path / "nanoprune-legal-v0.2.pt"
    torch.save(model.state_dict(), pt_path)
    print(f"📦 Poids PyTorch sauvegardés : {pt_path}")

    # Export ONNX
    onnx_path = out_path / "nanoprune-legal-v0.2.onnx"
    model.cpu()
    exported = export_to_onnx(model, onnx_path, seq_len=256)
    final_file = exported if exported.exists() else onnx_path
    print(f"🏆 Modèle officiel juridique exporté : {final_file} ({final_file.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    tsv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/legal_sample.tsv"
    train_legal_edition(tsv_path=tsv)
