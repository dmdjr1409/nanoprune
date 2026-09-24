"""Training loop for NanoPruneModel, with calibration, export and a model manifest.

Model selection only ever looks at the validation split. The test split and
the bundled evaluation suites are reported at the end, never used to choose
anything.
"""
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.calibration import TemperatureScaler, compute_ece
from ..core.tokenizer import NanoTokenizer
from .data import CATEGORY_LABELS, IGNORE_CATEGORY, read_jsonl


@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "weights"
    name: str = "nanoprune-v0.5"
    tokenizer_path: Optional[str] = "data/tokenizer_legal.json"
    train_file: str = "contrastive_train.jsonl"
    val_file: str = "contrastive_val.jsonl"
    test_file: str = "contrastive_test.jsonl"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_len: int = 256
    seed: int = 42
    device: str = "auto"
    # Relevance target = (1 - w) * label + w * teacher probability (when present).
    teacher_weight: float = 0.0
    margin: float = 1.5
    margin_weight: float = 0.5
    choice_weight: float = 0.2
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 4
    head_hidden: int = 128
    dropout: float = 0.1
    export_onnx: bool = True
    eval_suites: Sequence[str] = field(default_factory=lambda: ("dev", "heldout"))


def _device(name: str):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _targets(rows: List[dict], teacher_weight: float) -> List[float]:
    targets = []
    for row in rows:
        label = float(row["label"])
        if teacher_weight > 0.0 and row.get("teacher") is not None:
            label = (1.0 - teacher_weight) * label + teacher_weight * float(row["teacher"])
        targets.append(label)
    return targets


def _batches(rows: List[dict], targets: List[float], batch_size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield [rows[i] for i in idx], [targets[i] for i in idx]


def _encode(tokenizer: NanoTokenizer, rows: List[dict], max_len: int, device):
    import torch
    ids, masks = tokenizer.encode_pairs([r["query"] for r in rows], [r["content"] for r in rows], max_length=max_len)
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(masks, dtype=torch.long, device=device)


def predict_logits(model, tokenizer: NanoTokenizer, rows: List[dict], max_len: int, device, batch_size: int = 64) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            ids, masks = _encode(tokenizer, rows[start:start + batch_size], max_len, device)
            logits, _ = model(ids, masks)
            out.extend(logits.squeeze(-1).float().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def _metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> Dict[str, float]:
    from ..evaluation import roc_auc
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30, 30)))
    hard = (labels >= 0.5).astype(int)
    eps = 1e-7
    nll = float(-np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)))
    return {
        "loss": nll,
        "auc": roc_auc(hard.tolist(), probs.tolist()) or 0.0,
        "accuracy": float(np.mean((probs >= 0.5).astype(int) == hard)),
        "ece": compute_ece(probs, hard)["ece"],
    }


def train(config: TrainConfig, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Train, calibrate, export and describe a model. Returns the metrics."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from ..core.export import export_to_onnx
    from ..core.model import NanoPruneModel
    from ..core.weights import write_manifest

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    device = _device(config.device)

    data_dir, out_dir = Path(config.data_dir), Path(config.out_dir)
    tokenizer_file = Path(config.tokenizer_path) if config.tokenizer_path else None
    tokenizer = NanoTokenizer.from_file(tokenizer_file) if tokenizer_file else NanoTokenizer.char_level()

    train_rows = read_jsonl(data_dir / config.train_file)
    val_rows = read_jsonl(data_dir / config.val_file)
    test_path = data_dir / config.test_file
    test_rows = read_jsonl(test_path) if test_path.exists() else []
    if not train_rows or not val_rows:
        raise ValueError("Empty training or validation split")
    log(f"Device: {device} | tokenizer: {tokenizer.kind} ({tokenizer.vocab_size} tokens)")
    log(f"Pairs: train {len(train_rows)} | val {len(val_rows)} | test {len(test_rows)}")

    model = NanoPruneModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_layers=config.n_layers,
        max_seq_len=config.max_len,
        dropout=config.dropout,
        num_choices=len(CATEGORY_LABELS),
        head_hidden=config.head_hidden,
    ).to(device)
    log(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_CATEGORY)

    train_targets = _targets(train_rows, config.teacher_weight)
    val_labels = np.asarray([float(r["label"]) for r in val_rows])
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{config.name}.pt"
    best = {"auc": -1.0, "loss": float("inf"), "epoch": 0}
    history = []
    t0 = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for rows, targets in _batches(train_rows, train_targets, config.batch_size, True, rng):
            ids, masks = _encode(tokenizer, rows, config.max_len, device)
            target = torch.tensor(targets, dtype=torch.float32, device=device)
            cats = torch.tensor([int(r.get("category", IGNORE_CATEGORY)) for r in rows], dtype=torch.long, device=device)

            out = model.forward_all(ids, masks)
            logits = out["relevance_logits"].squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            pos, neg = target >= 0.5, target < 0.5
            if config.margin_weight > 0 and pos.any() and neg.any():
                # Keep the mean positive logit at least `margin` above the mean negative logit.
                loss = loss + config.margin_weight * F.relu(config.margin - (logits[pos].mean() - logits[neg].mean()))
            if config.choice_weight > 0 and (cats != IGNORE_CATEGORY).any():
                loss = loss + config.choice_weight * ce_loss(out["choice_logits"], cats)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        scheduler.step()

        val = _metrics(predict_logits(model, tokenizer, val_rows, config.max_len, device), val_labels)
        history.append({"epoch": epoch, "train_loss": total / max(1, batches), **val})
        log(f"Epoch {epoch:02d}/{config.epochs:02d} | train loss {total / max(1, batches):.4f} | "
            f"val loss {val['loss']:.4f} | val AUC {val['auc']:.3f} | val acc {val['accuracy']:.3f} | ECE {val['ece']:.3f}")
        if val["auc"] > best["auc"] or (val["auc"] == best["auc"] and val["loss"] < best["loss"]):
            best = {"auc": val["auc"], "loss": val["loss"], "epoch": epoch}
            torch.save(model.state_dict(), pt_path)

    log(f"Training done in {time.time() - t0:.1f}s; best epoch {best['epoch']} (val AUC {best['auc']:.3f})")
    model.load_state_dict(torch.load(str(pt_path), map_location=device, weights_only=True))

    # Temperature scaling on the validation split.
    val_logits = predict_logits(model, tokenizer, val_rows, config.max_len, device)
    before = _metrics(val_logits, val_labels)
    temperature = TemperatureScaler().fit(val_logits, (val_labels >= 0.5).astype(float))
    after = _metrics(val_logits, val_labels, temperature)
    model.temperature.data.fill_(temperature)
    log(f"Temperature {temperature:.3f}: val ECE {before['ece']:.3f} -> {after['ece']:.3f}")
    model.to("cpu").eval()
    torch.save(model.state_dict(), pt_path)

    metrics: Dict[str, Any] = {"val": after, "best_epoch": best["epoch"], "history": history}
    if test_rows:
        test_labels = np.asarray([float(r["label"]) for r in test_rows])
        metrics["test"] = _metrics(predict_logits(model, tokenizer, test_rows, config.max_len, "cpu"), test_labels, 1.0)
        log(f"Test split: AUC {metrics['test']['auc']:.3f} | acc {metrics['test']['accuracy']:.3f} | ECE {metrics['test']['ece']:.3f}")

    files = {"pt": pt_path.name}
    if config.export_onnx:
        onnx_path = out_dir / f"{config.name}.onnx"
        export_to_onnx(model, onnx_path, seq_len=config.max_len, heads="all")
        files["onnx"] = onnx_path.name
        log(f"ONNX export: {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KiB)")

    shipped_tokenizer = None
    if tokenizer_file is not None:
        shipped_tokenizer = out_dir / f"{config.name}-tokenizer.json"
        if shipped_tokenizer.resolve() != tokenizer_file.resolve():
            shutil.copyfile(tokenizer_file, shipped_tokenizer)

    # Suites are reported for information; they never influence training.
    if config.eval_suites:
        from ..engine.pruner import NanoPruner
        from ..evaluation import evaluate_scorer, load_suite
        pruner = NanoPruner.from_torch_model(model, tokenizer)
        metrics["suites"] = {}
        for suite in config.eval_suites:
            res = evaluate_scorer(load_suite(suite), pruner.score, n_bootstrap=0)
            metrics["suites"][suite] = {k: res.get(k) for k in ("accuracy", "auc", "precision", "recall", "ece")}
            log(f"Suite {suite}: accuracy {res['accuracy']:.3f} | AUC {res['auc']:.3f}")

    manifest_path = out_dir / f"{config.name}.json"
    write_manifest(
        manifest_path,
        name=config.name,
        architecture=model.config(),
        files=files,
        tokenizer_file=shipped_tokenizer,
        temperature=temperature,
        extra={
            "choice_labels": CATEGORY_LABELS,
            "heads": {"relevance": True, "choice": config.choice_weight > 0, "score": False},
            "metrics": {k: v for k, v in metrics.items() if k != "history"},
            "training": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()},
        },
    )
    log(f"Manifest: {manifest_path}")
    with open(out_dir / f"{config.name}.history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return metrics
