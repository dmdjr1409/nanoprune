# NanoPrune ✂️

> **Local relevance scoring and RAG context pruning with a small non-generative Transformer, plus a private local search app.**

[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

NanoPrune scores how relevant a passage is to a query (a number in `[0, 1]`) and drops the passages below a threshold. It does not generate text. You put it in front of a larger model, so that only useful context reaches it (RAG), or you use it to search your own documents without sending them anywhere.

It borrows the "System One" idea of [Jev](https://www.firecrawl.dev/blog/what-is-jev) (TypeSafe AI): typed decisions such as `choice`, `score` and `noul` (a yes/no probability) instead of prose. [Laya](https://huggingface.co/convaiinnovations/laya) (Convai, 421M parameters, ModernBERT-large) is an open, Jev-compatible model. NanoPrune is a much smaller model for the relevance case, and it can serve as the fast first tier of a cascade with Laya.

## Status: read this first

| | |
| --- | --- |
| Local search app, CLI, evaluation, training pipeline | ✅ working and tested (86 unit tests) |
| Model **v0.4** (5.45M parameters, WordPiece) | ⚠️ weights are **not published**: they exist only on the author's machine; recorded accuracy on the dev suite was 60 % |
| Published model **v0.3** (release assets) | ❌ does not separate relevant from irrelevant passages: 46 % on the dev suite (AUC 0.52), 43 % on the held-out suite; its `.onnx` asset lacks its `.onnx.data` file |
| No weights installed | NanoPrune runs a **keyword heuristic** and says so everywhere (CLI warning, `backend: heuristic`, banner in the app) |

On the held-out suite, keyword baselines reach 55–65 % accuracy. A NanoPrune model has to beat that before its neural score is worth using (see [Evaluation](#evaluation)).

---

## Install

```bash
git clone https://github.com/dmdjr1409/nanoprune.git
cd nanoprune
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[onnx]"     # ONNX inference (onnxruntime + tokenizers)
pip install -e ".[train]"    # PyTorch inference, training, ONNX export
pip install -e ".[docs]"     # PDF indexing in the search app (.docx needs nothing extra)
```

`pip install -e .` alone (numpy only) gives the search app, the keyword heuristic, BM25 and the evaluation tools.

## Getting the weights

`NanoPruner.load()` looks for weights in this order and uses the first directory that has any (newest model first, ONNX before PyTorch):

1. `$NANOPRUNE_MODEL` (a weights file) or `$NANOPRUNE_HOME/weights` and `$NANOPRUNE_HOME`;
2. `weights/` in this repository (editable install);
3. `~/.cache/nanoprune/weights` (or `$XDG_CACHE_HOME/nanoprune/weights`), where `nanoprune download` puts files.

```bash
nanoprune download --tag v0.4.0   # once published; checks each file's SHA-256
nanoprune info                    # what is loaded, from where, with which tokenizer
```

A model trained with the WordPiece tokenizer must be shipped with it: NanoPrune refuses to run a model with the wrong vocabulary instead of returning meaningless scores. New models come with a manifest (`nanoprune-<version>.json`) that ties the weights to their tokenizer (with its SHA-256) and architecture. Legacy files (`nanoprune-v0.4.pt`, `-v0.3`, …) are recognised by name; for v0.4 the tokenizer `tokenizer_legal.json` is looked up next to the weights and in `data/`.

**Publishing weights (maintainers).** `scripts/package_release.py` turns a checkpoint and its tokenizer into release files (`.pt`, self-contained `.onnx` with all heads, tokenizer, manifest), checks that ONNX and PyTorch agree, and prints the `gh release create` command:

```bash
python scripts/package_release.py --pt weights/nanoprune-v0.4.pt \
    --tokenizer data/tokenizer_legal.json --name nanoprune-v0.4 --tag v0.4.0
```

---

## Python usage

```python
from nanoprune import NanoPruner

pruner = NanoPruner.load()          # add strict=True to fail when no model is found
print(pruner.describe())            # backend: onnx / torch / heuristic

retained = pruner.prune(
    query="Severe penicillin allergy",
    candidates=[
        "Patient exhibits documented severe anaphylaxis to penicillin and amoxicillin.",
        "Patient underwent routine knee surgery in 2019 without complications.",
    ],
    threshold=0.75,
)
for text, score in retained:
    print(f"[{score:.2f}] {text}")
```

`score`, `score_pair`, `prune` and `rank` work with every backend. `choice` and `score_rubric` need a trained head (PyTorch checkpoint, or ONNX exported with all heads) and raise `HeadUnavailableError` otherwise, instead of returning made-up values. In v0.4, `choice` only knows its four training categories (options are mapped to them by position), and the `score` head is not trained.

## CLI

```bash
nanoprune search "Contre-indication AINS" --dir sample_data/medical
nanoprune prune "Refund policy" "30-day money back guarantee" "Opening hours: 9am-5pm"
nanoprune app --dir sample_data/medical        # local search app on http://127.0.0.1:7860
nanoprune info                                 # loaded model and search paths
nanoprune eval                                 # baselines (+ model, + Laya with --laya)
nanoprune download --tag <tag>                 # install published weights
```

Every model-using command accepts `--model PATH`, `--tokenizer PATH` and `--strict`, and prints which backend produced the scores.

## Local search app

`nanoprune app` indexes a folder (`.txt .md .json .csv .tsv .log .rst .docx`, plus `.pdf` with `pypdf`), then retrieves passages with BM25 and re-ranks them with the model, or with the heuristic when no model is loaded. Results show the file, line numbers, the best matching sentence and the score. When a model is loaded and the index is small (up to 256 passages), every passage is scored, so passages that share no keyword with the query can still be found.

Files are read and indexed by the local server only. The server:

- listens on `127.0.0.1` only and rejects requests whose `Host` is not that address (blocks DNS rebinding);
- rejects cross-site requests (`Origin` / `Sec-Fetch-Site`), requires `application/json` bodies, and never sends CORS headers, so other websites open in your browser cannot read your files through it;
- serves the page with a strict Content-Security-Policy and renders document text as text, so a crafted document cannot run script in the page;
- loads no third-party resources (no web fonts or CDN).

## Hybrid cascade with Laya

`HybridCascadePruner` drops candidates that share no word with the query and have a low tier-1 score, keeps very confident matches, and sends the rest to Laya. A "rescue" rule keeps candidates Laya hesitates on (0.25–0.50) when they share words with the query and the tier-1 score is at least 0.70. The Laya checkpoint is configurable (`laya_model=` or `NANOPRUNE_LAYA_MODEL`). The default `convaiinnovations/laya` is the English ModernBERT checkpoint; Convai also publishes a multilingual variant, which is worth testing on French text.

---

## Evaluation

```bash
nanoprune eval                     # dev + held-out suites, lexical baselines, discovered model
nanoprune eval --laya              # + Laya, the cascade, and two ablations of the cascade
```

Two suites ship with the package ([details](src/nanoprune/benchmarks/README.md)):

- **dev**: the 50 cases the v0.4 cascade thresholds were tuned on. Lexically easy: BM25 ranks it with AUC 0.97.
- **heldout**: 50 queries × 4 documents, never used for tuning: direct positives, paraphrased positives (≤ 1 shared word), lexical traps (≥ 2 shared words, wrong question) and unrelated documents.

Results at threshold 0.5, from `nanoprune eval --no-model` and the published v0.3 checkpoint (95 % bootstrap intervals in the command output):

| Scorer | dev accuracy | dev AUC | held-out accuracy | held-out AUC | held-out traps accepted | held-out paraphrases found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always "relevant" | 50.0 % | 0.50 | 50.0 % | 0.50 | 50/50 | 50/50 |
| Keyword heuristic (no-model fallback) | 90.0 % | 0.87 | 55.0 % | 0.65 | 43/50 | 3/50 |
| Lexical overlap (cascade gate) | 84.0 % | 0.98 | 57.5 % | 0.64 | 35/50 | 0/50 |
| IDF-weighted coverage | 72.0 % | 0.97 | 64.5 % | 0.63 | 21/50 | 0/50 |
| BM25 (ranking only) | — | 0.97 | — | 0.62 | — | — |
| NanoPrune v0.3 (published) | 46.0 % | 0.52 | 43.0 % | 0.41 | 18/50 | 7/50 |
| NanoPrune v0.4 (recorded, not published) | 60 % | — | not measured | — | — | — |

The v0.4 cascade was recorded at 94 % on the dev suite (Laya alone: 92 %), but its thresholds were tuned on that suite, and the lexical-overlap score it uses as a gate already reaches 84 % there on its own. `nanoprune eval --laya` reruns the cascade with the NanoPrune score replaced by a constant (0.5, then 0.72). If the real cascade does not beat these ablations, the neural score adds nothing to it.

## Training

The pipeline (v0.5) fixes the problems of the v0.4 scripts: the train/validation split leaked titles and passages, "hard" negatives were random (a shared "de" was enough), the ONNX export call was broken, and there was no calibration. Every step is tested end to end on a synthetic corpus.

```bash
# 1. WordPiece tokenizer, trained on text normalised exactly like at inference
python scripts/train_tokenizer.py --tsv data/legal_raw.tsv
# 2. pairs, split by document before pairing; BM25-mined hard negatives
python scripts/generate_contrastive_dataset.py --tsv data/legal_raw.tsv [--queries real_queries.jsonl]
# 3. optional: Laya soft labels for distillation (resumable)
python scripts/distill_teacher.py
# 4. train, select on validation AUC, calibrate the temperature, export ONNX, write the manifest
python scripts/train_contrastive.py --name nanoprune-v0.5 [--suffix .teacher --teacher-weight 0.5]
```

The main limitation remains the data: titles are used as queries unless you supply real search queries (`--queries`). A model trained from scratch on a few thousand pairs learns little language; distilling Laya into a small pretrained encoder is the most promising next step. The v0.1–v0.3 scripts are kept in [`scripts/legacy/`](scripts/legacy/README.md).

## Model facts (v0.4)

| Item | v0.4 |
| --- | ---: |
| Parameters | 5,454,728 |
| Transformer layers / hidden size / heads / FFN | 4 / 256 / 8 / 1024 |
| Maximum sequence length | 256 |
| WordPiece vocabulary | 8192 (`data/tokenizer_legal.json`) |
| PyTorch checkpoint | ~20.7 MiB |
| ONNX (graph + external data) | ~20.8 MiB |

## Running tests

```bash
pip install -e ".[all]"   # or just `pip install -e .`: tests needing torch/onnx/pypdf are skipped
python -m unittest discover -s tests -p "test_*.py" -v
```

## License

MIT © [Junior Diomande](https://github.com/dmdjr1409). See [CHANGELOG.md](CHANGELOG.md) for the history of changes.
