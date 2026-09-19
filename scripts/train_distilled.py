#!/usr/bin/env python3
"""
NanoPrune v0.3.0 Training Script with Multi-Task Knowledge Distillation.
Trains NanoPrune (960k Student) from Laya (421M Teacher) soft-calibrated targets.
"""
import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nanoprune.core.tokenizer import NanoTokenizer
from nanoprune.core.model import NanoPruneModel
from nanoprune.core.calibration import compute_ece
from nanoprune.core.export import export_to_onnx

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class DistilledDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], tokenizer: NanoTokenizer, max_len: int = 256):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        query = item["query"]
        content = item["content"]

        input_ids, mask = self.tokenizer.encode_pair(query, content, max_length=self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "soft_relevance": torch.tensor(item["soft_relevance"], dtype=torch.float32),
            "category_idx": torch.tensor(item["category_idx"], dtype=torch.long),
            "category_probs": torch.tensor(item["category_probs"], dtype=torch.float32),
            "complexity_score": torch.tensor(item["complexity_score"], dtype=torch.float32),
        }

def load_distilled_data(jsonl_path: str) -> List[Dict[str, Any]]:
    print(f"📖 Chargement des données distillées depuis {jsonl_path}...")
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"   {len(items)} paires de distillation chargées.")
    return items

def train_distilled_v03(
    data_path: str = "data/distilled_pairs.jsonl",
    output_dir: str = "weights",
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 4e-4,
    device_name: str = "mps" if torch.backends.mps.is_available() else "cpu",
):
    print("=" * 60)
    print("🚀 Entraînement NanoPrune v0.3.0 (Distilled Multi-Primitive)")
    print("=" * 60)

    device = torch.device(device_name)
    print(f"🖥️ Périphérique d'entraînement : {device}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = NanoTokenizer()
    items = load_distilled_data(data_path)

    random.shuffle(items)
    split = int(len(items) * 0.85)
    train_items = items[:split]
    val_items = items[split:]

    train_ds = DistilledDataset(train_items, tokenizer)
    val_ds = DistilledDataset(val_items, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    vocab_size = len(tokenizer.vocab)
    print(f"📦 Vocabulaire tokenizer : {vocab_size} tokens")

    model = NanoPruneModel(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=2,
        max_seq_len=256,
        dropout=0.1,
        temperature=1.0,
        num_choices=4,
    ).to(device)

    param_count = model.count_parameters()
    print(f"🧠 Paramètres du modèle NanoPrune : {param_count:,} ({param_count * 4 / 1024 / 1024:.2f} MB en FP32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_pt_path = out_path / "nanoprune-v0.3.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_rel = batch["soft_relevance"].to(device).unsqueeze(-1)
            target_cat_probs = batch["category_probs"].to(device)
            target_score = batch["complexity_score"].to(device).unsqueeze(-1)

            optimizer.zero_grad()

            preds = model.forward_all(input_ids, attention_mask)

            # Perte 1 : Relevance soft BCE
            loss_rel = F.binary_cross_entropy(preds["relevance_probs"], soft_rel)

            # Perte 2 : Choice KL divergence (distillation)
            log_choice = F.log_softmax(preds["choice_logits"], dim=-1)
            loss_choice = F.kl_div(log_choice, target_cat_probs, reduction="batchmean")

            # Perte 3 : Score regression
            loss_score = F.mse_loss(preds["score"], target_score)

            # Perte combinée multi-tâches
            loss = loss_rel + 0.5 * loss_choice + 0.25 * loss_score

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(input_ids)

        scheduler.step()
        train_loss /= len(train_items)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_rel = []
        val_targets_rel = []
        val_cat_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                soft_rel = batch["soft_relevance"].to(device).unsqueeze(-1)
                target_cat_idx = batch["category_idx"].to(device)
                target_cat_probs = batch["category_probs"].to(device)
                target_score = batch["complexity_score"].to(device).unsqueeze(-1)

                preds = model.forward_all(input_ids, attention_mask)

                l_rel = F.binary_cross_entropy(preds["relevance_probs"], soft_rel)
                log_c = F.log_softmax(preds["choice_logits"], dim=-1)
                l_choice = F.kl_div(log_c, target_cat_probs, reduction="batchmean")
                l_score = F.mse_loss(preds["score"], target_score)

                l_batch = l_rel + 0.5 * l_choice + 0.25 * l_score
                val_loss += l_batch.item() * len(input_ids)

                val_preds_rel.extend(preds["relevance_probs"].squeeze(-1).cpu().numpy().tolist())
                val_targets_rel.extend(soft_rel.squeeze(-1).cpu().numpy().tolist())

                pred_cat = preds["choice_logits"].argmax(dim=-1)
                val_cat_correct += (pred_cat == target_cat_idx).sum().item()
                val_total += len(input_ids)

        val_loss /= len(val_items)
        cat_acc = val_cat_correct / max(1, val_total)
        brier = np.mean((np.array(val_preds_rel) - np.array(val_targets_rel)) ** 2)

        # Binaires pour ECE (seuil 0.5)
        bin_targets = np.array(val_targets_rel) >= 0.5
        ece_res = compute_ece(val_preds_rel, bin_targets.astype(int).tolist())
        ece_val = ece_res["ece"]

        epoch_time = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} ({epoch_time:.1f}s) | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Brier: {brier:.4f} | ECE: {ece_val*100:.2f}% | Cat Acc: {cat_acc*100:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_pt_path)

    print(f"\n💾 Meilleur modèle PyTorch sauvegardé dans {best_pt_path} ({best_pt_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # Export ONNX
    onnx_path = out_path / "nanoprune-v0.3.onnx"
    print(f"\n📦 Export ONNX vers {onnx_path}...")
    model.load_state_dict(torch.load(best_pt_path, map_location="cpu"))
    model.eval()

    export_to_onnx(model, str(onnx_path), seq_len=256)
    print(f"✅ Modèle ONNX généré : {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/distilled_pairs.jsonl"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "weights"
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    train_distilled_v03(data_file, out_dir, epochs=ep)
