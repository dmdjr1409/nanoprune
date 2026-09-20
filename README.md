# NanoPrune ✂️

> **NanoPrune v0.4 — a 5.45M-parameter local System One relevance model and RAG pruner.**
> Runs query/document relevance scoring locally, with a PyTorch checkpoint of ~20.7 MiB and an ONNX bundle of ~20.8 MiB.

[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](#model-facts)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Parameters](https://img.shields.io/badge/parameters-5.45M-brightgreen.svg)](#model-facts)

NanoPrune is a **non-generative** Transformer used to score, rank, and prune retrieved context before it reaches a larger model. It is designed for local inference and can also act as the first tier of a hybrid cascade with a larger System One model such as Laya.

The current v0.4 release replaces the older v0.3 model. Older README figures such as **495k parameters**, **1.9 MB**, **228 KB ONNX**, and **2.8 MB** described previous artifacts and do **not** describe v0.4.

---

## What v0.4 does

- **Relevance scoring** — score a `(query, document)` pair in `[0, 1]`.
- **RAG pruning/ranking** — discard low-scoring chunks before an LLM call.
- **Local document search** — index and search a directory without cloud inference.
- **Hybrid cascade** — use NanoPrune as the fast first tier and send ambiguous cases to Laya.
- **Categorical head (`choice`)** — trained against the fixed categories used by the v0.4 training data; it should not be treated as arbitrary zero-shot classification.
- **Rubric head (`score`)** — present in the architecture, but **experimental in v0.4**. The v0.4 contrastive training pipeline does not apply a dedicated score-head loss, so it is not part of the validated v0.4 benchmark claim.

NanoPrune does not generate prose, so “hallucination rate” is not an appropriate metric for the model itself. Its relevant failure modes are ranking/classification errors such as false positives and false negatives.

---

## Model facts

The committed v0.4 checkpoint contains **5,454,728 parameters/tensors elements** in its state dict.

| Item | v0.4 |
| --- | ---: |
| Transformer layers | 4 |
| Hidden size | 256 |
| Attention heads | 8 |
| Feed-forward size | 1024 |
| Maximum sequence length | 256 |
| WordPiece vocabulary | 8192 |
| PyTorch checkpoint | 21,710,865 bytes (~20.70 MiB) |
| ONNX graph | 422,723 bytes (~0.40 MiB) |
| ONNX external data | 21,430,272 bytes (~20.44 MiB) |
| ONNX bundle total | 21,852,995 bytes (~20.84 MiB) |

The ONNX export currently exposes the **relevance** path (`logits`, `probabilities`). The multi-primitive heads are available through the PyTorch model.

---

## Development benchmark

`data/benchmark_results.json` contains the recorded 50-case development benchmark used while building the v0.4 cascade.

| Metric | NanoPrune v0.4 | Laya 421M | Hybrid cascade |
| --- | ---: | ---: | ---: |
| Accuracy vs. the 50 labels | 60% | 92% | **94%** |
| Recorded latency / document | 4.41 ms | 77.02 ms | 49.16 ms |
| Hard-negative traps fooled | 6 / 10 | 0 / 10 | 1 / 10 |

The cascade sent **24/50** cases to Laya and therefore avoided **52%** of Laya calls in that run. Its recorded speedup versus running Laya on every case was **1.57×**.

### Benchmark limitation

The cascade thresholds and rescue logic were iterated while inspecting this same 50-case set. Therefore, **94% is a development score, not an independent held-out/generalization result**. A frozen external test set is required before using that number as a production-quality accuracy claim.

Latency numbers are measurements from the recorded local run and should not be treated as universal cross-hardware results.

To reproduce the comparison in an environment where Laya is installed:

```bash
python3 scripts/benchmark_deep_comparison.py
```

---

## Quickstart

### 1. Clone and create an environment

```bash
git clone https://github.com/dmdjr1409/nanoprune.git
cd nanoprune

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install for local ONNX relevance inference

Use an editable install so the committed tokenizer and weight artifacts remain available from the repository:

```bash
pip install -e ".[onnx]"
```

For PyTorch/model-development workflows:

```bash
pip install -e ".[train]"
```

Verify the package version:

```bash
nanoprune --version
# nanoprune 0.4.0
```

---

## Python usage

```python
from nanoprune import NanoPruner

pruner = NanoPruner.load()

query = "Severe penicillin allergy"
candidates = [
    "Patient exhibits documented severe anaphylaxis to penicillin and amoxicillin.",
    "Patient underwent routine knee surgery in 2019 without complications.",
    "Prescribed paracetamol 1g twice daily for mild fever.",
]

retained = pruner.prune(
    query=query,
    candidates=candidates,
    threshold=0.75,
)

for text, confidence in retained:
    print(f"[{confidence * 100:.1f}%] {text}")
```

---

## CLI

Search a local folder:

```bash
nanoprune search "Contre-indication AINS" --dir sample_data/medical
```

Prune raw candidates:

```bash
nanoprune prune \
  "Refund policy" \
  "30-day money back guarantee" \
  "Opening hours: 9am-5pm"
```

Launch the local search UI:

```bash
nanoprune app --dir sample_data/medical
```

---

## Architecture

v0.4 uses:

- **Tokenizer:** 8192-token WordPiece vocabulary (`data/tokenizer_legal.json`).
- **Encoder:** 4-layer bidirectional Transformer (`d_model=256`, `heads=8`, `ffn=1024`, `seq_len=256`).
- **Relevance head:** binary relevance score used by `score`, `rank`, and `prune`.
- **Choice head:** four-way categorical head trained by the v0.4 category objective.
- **Score head:** continuous `[0, 4]` head retained in the architecture but not independently trained by the v0.4 contrastive pipeline.
- **Hybrid cascade:** conservative fast-drop logic plus optional Laya arbitration for ambiguous cases.

---

## Running tests

Install the training/test dependencies, then run the suite from the repository root:

```bash
pip install -e ".[train]"
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py" -v
```

The current suite contains **10 unit tests**.

---

## Repository artifacts

- `weights/nanoprune-v0.4.pt` — PyTorch v0.4 checkpoint.
- `weights/nanoprune-v0.4.onnx` + `.onnx.data` — ONNX relevance model and external weights.
- `data/tokenizer_legal.json` — tokenizer used for v0.4 training/inference.
- `data/benchmark_results.json` — recorded development benchmark.
- `scripts/train_contrastive.py` — v0.4 training pipeline.
- `scripts/benchmark_deep_comparison.py` — NanoPrune/Laya/cascade comparison.

---

## License

MIT © [Junior Diomande](https://github.com/dmdjr1409).
