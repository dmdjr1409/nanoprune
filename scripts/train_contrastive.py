#!/usr/bin/env python3
"""
NanoPrune v0.4.0 Contrastive Margin Training Pipeline.
Trains the 5.45M-parameter Transformer with BCE relevance loss,
contrastive margin regularization, and categorical choice loss.
"""
import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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

class ContrastiveDataset(Dataset):
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

        label = float(item["label"])
        cat = int(item.get("category", 0))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
            "category": torch.tensor(cat, dtype=torch.long),
        }

def load_data(jsonl_path: str) -> List[Dict[str, Any]]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def train():
    print("=" * 70)
    print("⚡ NanoPrune v0.4 Contrastive Margin Training (5.4M params)")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = NanoTokenizer(json_path="data/tokenizer_legal.json")
    print(f"Tokenizer loaded (vocab: {tokenizer.vocab_size})")

    # Load dataset
    data_path = "data/contrastive_pairs.jsonl"
    all_items = load_data(data_path)
    random.shuffle(all_items)

    split = int(len(all_items) * 0.85)
    train_items = all_items[:split]
    val_items = all_items[split:]
    print(f"Train samples: {len(train_items)} | Val samples: {len(val_items)}")

    train_ds = ContrastiveDataset(train_items, tokenizer)
    val_ds = ContrastiveDataset(val_items, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Initialize 5.4M model
    model = NanoPruneModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        n_layers=4,
        max_seq_len=256,
        dropout=0.1,
        num_choices=4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params * 4 / 1024 / 1024:.2f} MB FP32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights_path = Path("weights/nanoprune-v0.4.pt")
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    t0_train = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            cats = batch["category"].to(device)

            optimizer.zero_grad()

            outputs = model.forward_all(input_ids, mask)
            rel_logits = outputs["relevance_logits"].squeeze(-1)
            choice_logits = outputs["choice_logits"]

            # Main relevance loss: BCEWithLogits
            loss_rel = bce_loss_fn(rel_logits, labels)

            # Margin Contrastive Penalty: push positive logits > +2.0 and negative logits < -2.0
            pos_mask = labels == 1.0
            neg_mask = labels == 0.0
            loss_margin = 0.0
            if pos_mask.any() and neg_mask.any():
                pos_logits = rel_logits[pos_mask]
                neg_logits = rel_logits[neg_mask]
                # We want pos_logits - neg_logits >= 2.0
                margin_diff = F.relu(1.5 - (pos_logits.mean() - neg_logits.mean()))
                loss_margin = margin_diff

            # Multi-primitive category loss
            loss_cat = ce_loss_fn(choice_logits, cats)

            total_loss = loss_rel + 0.5 * loss_margin + 0.2 * loss_cat
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                _, probs = model(input_ids, mask)
                val_preds.extend(probs.squeeze(-1).cpu().numpy().tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

        val_preds_arr = np.array(val_preds)
        val_targets_arr = np.array(val_targets)

        binary_preds = (val_preds_arr >= 0.50).astype(int)
        val_acc = np.mean(binary_preds == val_targets_arr) * 100.0
        cal_res = compute_ece(val_preds_arr, val_targets_arr.astype(int))
        ece = cal_res["ece"]
        brier = cal_res["brier_score"]

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.1f}% | ECE: {ece*100:.2f}% | Brier: {brier:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_weights_path))

    elapsed = time.time() - t0_train
    print(f"\n🎉 Training complete in {elapsed:.1f}s! Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"💾 Checkpoint saved to {best_weights_path}")

    # Export to ONNX
    onnx_path = Path("weights/nanoprune-v0.4.onnx")
    print(f"\n📦 Exporting to ONNX: {onnx_path}...")
    try:
        model.load_state_dict(torch.load(str(best_weights_path), map_location="cpu"))
        model.to("cpu").eval()
        export_to_onnx(model, output_path=onnx_path, max_seq_len=256)
        print(f"✅ ONNX export successful ({onnx_path.stat().st_size / 1024:.1f} KB)!")
    except Exception as e:
        print(f"⚠️ ONNX export warning: {e}")

if __name__ == "__main__":
    train()
