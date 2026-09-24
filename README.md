# NanoPrune ✂️

> **Local relevance scoring, semantic search and RAG context pruning, 100 % offline, plus a private local search app.**

[![Version](https://img.shields.io/badge/version-0.7.0-blue.svg)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

NanoPrune scores how relevant a passage is to a query (a probability in `[0, 1]`) and drops the passages below a threshold. It does not generate text. You put it in front of a larger model, so that only useful context reaches it (RAG), or you use it to search your own documents without sending them anywhere.

Two scorers are available: a **semantic model** (a pretrained multilingual sentence-embedding model run with ONNX, recommended) and the small **NanoPrune** Transformer. Without either, a keyword heuristic is used and flagged as such.

![The NanoPrune search app: a question worded differently from the document finds the prescription passage](docs/app-screenshot.png)

It borrows the "System One" idea of [Jev](https://www.firecrawl.dev/blog/what-is-jev) (TypeSafe AI): typed decisions such as `choice`, `score` and `noul` (a yes/no probability) instead of prose. [Laya](https://huggingface.co/convaiinnovations/laya) (Convai, 421M parameters, ModernBERT-large) is an open, Jev-compatible model. NanoPrune is a much smaller model for the relevance case, and it can serve as the fast first tier of a cascade with Laya.

## Status: read this first

| | |
| --- | --- |
| **Semantic search** (multilingual-e5-large, int8, 552 MB) | ✅ **85 % accuracy on the held-out suite** (AUC 0.93), finds 45/50 paraphrased passages; recommended mode |
| Local search app, CLI, evaluation, training pipeline | ✅ working and tested (119 unit tests) |
| Model **v0.4** (5.45M parameters, WordPiece) | ⚠️ weights are **not published**: they exist only on the author's machine; recorded accuracy on the dev suite was 60 % |
| Published model **v0.3** (release assets) | ❌ does not separate relevant from irrelevant passages: 46 % on the dev suite (AUC 0.52), 43 % on the held-out suite; its `.onnx` asset lacks its `.onnx.data` file |
| No weights installed | NanoPrune runs a **keyword heuristic** and says so everywhere (CLI warning, `backend: heuristic`, banner in the app) |

On the held-out suite, keyword baselines reach 55–65 % accuracy and the semantic model 85 %. A NanoPrune model has to beat the keywords before its score is worth using, and the semantic model before it is worth its smaller size (see [Evaluation](#evaluation)).

---

## Install

```bash
git clone https://github.com/dmdjr1409/nanoprune.git
cd nanoprune
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[semantic]" # semantic search (onnxruntime, tokenizers, onnx)
pip install -e ".[onnx]"     # NanoPrune ONNX inference (onnxruntime + tokenizers)
pip install -e ".[train]"    # PyTorch inference, training, ONNX export
pip install -e ".[docs]"     # PDF indexing in the search app (.docx needs nothing extra)
```

`pip install -e .` alone (numpy only) gives the search app, the keyword heuristic, BM25 and the evaluation tools.

### Quick start: semantic search

```bash
pip install -e ".[semantic]"
nanoprune download --dense multilingual-e5-large   # 1.3 GB download, int8-quantised to 552 MB, calibrated
nanoprune app --dir ~/Documents/Clients            # opens the browser; uses the semantic model automatically
```

The archive is the ONNX export of [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) (MIT licence) mirrored by the fastembed project; its SHA-256 is pinned in `nanoprune/engine/dense.py`. Once installed, `search`, `prune`, `app` and `eval` use it by default; `--no-dense` or `--model` switch back to NanoPrune.

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

**Publishing a version (maintainers).** Set the version in `pyproject.toml` and `nanoprune/__init__.py`, add its section to `CHANGELOG.md`, merge, then push a `vX.Y.Z` tag (or run the *release* workflow from the Actions tab with the version number). The workflow checks that the three agree, builds and tests the wheel and the sdist, and publishes the GitHub release with the changelog section as notes.

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

from nanoprune import SemanticScorer

scorer = SemanticScorer.load()      # installed semantic model; same score/prune/rank interface
print(scorer.prune("Refund policy", ["30-day money back guarantee", "Opening hours: 9am-5pm"]))
```

Searching a folder from Python:

```python
from nanoprune import LocalDocumentIndexer, LocalSearchEngine, SemanticScorer

indexer = LocalDocumentIndexer()
indexer.index_directory("sample_data/medical", progress=lambda done, total: print(done, "/", total))
engine = LocalSearchEngine(indexer, SemanticScorer.load(disk_cache=True))
engine.prepare()                    # embeds every passage once (cached on disk)
response = engine.search("Que prendre pour une douleur au genou ?", top_k=5, threshold=0.5)
for r in response["results"]:       # also: near_misses, matches_total, more_available
    print(f"{r['score']:.0%} {r['rel_path']}:{r['line_start']} {r['highlight']}")
```

`score`, `score_pair`, `prune` and `rank` work with every backend. `choice` and `score_rubric` need a trained head (PyTorch checkpoint, or ONNX exported with all heads) and raise `HeadUnavailableError` otherwise, instead of returning made-up values. In v0.4, `choice` only knows its four training categories (options are mapped to them by position), and the `score` head is not trained.

## CLI

```bash
nanoprune search "Contre-indication AINS" --dir sample_data/medical   # add --json for the full response
nanoprune prune "Refund policy" "30-day money back guarantee" "Opening hours: 9am-5pm"
nanoprune app --dir sample_data/medical        # local search app on http://127.0.0.1:7860 (--no-browser)
nanoprune info                                 # loaded model and search paths
nanoprune eval                                 # baselines (+ model, + Laya with --laya)
nanoprune download --tag <tag>                 # install published NanoPrune weights
nanoprune download --dense multilingual-e5-large  # install the semantic model
nanoprune calibrate --dense /path/to/model     # calibrate any ONNX sentence-embedding model on the dev suite
```

Every model-using command accepts `--dense PATH|auto`, `--no-dense`, `--model PATH`, `--tokenizer PATH` and `--strict`, and prints which backend produced the scores.

## Local search app

`nanoprune app` starts a server on `127.0.0.1`, opens the page in your browser (`--no-browser` to skip) and indexes the folder given with `--dir` (or the bundled sample) in the background. Passages are scored with the semantic model when one is installed, otherwise with the NanoPrune model or the keyword heuristic (BM25 then pre-selects candidates).

- **Import**: drop a folder or files anywhere on the page, pick them, or type a folder path; with a path, results can open the original files. Formats: `.txt .md .json .csv .tsv .log .rst .docx`, plus `.pdf` with `pypdf`. Imports run in the background with a progress bar, the time left and a *Cancel* button; the previous folder stays searchable until the new one is ready, and skipped files are listed with the reason.
- **Results**: one card per file with its best passage, the sentence that answers best (query words highlighted), the line numbers and the score; other passages of the same file are folded under it. Each result can be opened with its default application or copied as a citation (« passage » — file, lines). When nothing passes the filter, the closest passages are shown anyway, below it.
- **Filter**: *Large*, *Normal*, *Strict* keep passages scoring at least 30, 50 or 70 %.
- **Keyboard**: `/` or `Ctrl+K` search, `↓` `↑` move through the results, `Enter` shows the full passage, `O` opens the file, `C` copies the citation, `Esc` clears, `?` help.
- Light and dark themes (following the system, with a toggle), and a layout that works in a phone-sized window.

In semantic mode every passage is embedded once and each query scores all of them, so passages that share no keyword with the query are found (25–50 ms per query for a few hundred passages). Embedding takes about 40 ms per passage on 4 CPU threads. The vectors are saved in `~/.cache/nanoprune/embeddings/` (SQLite): a folder already analysed is ready in seconds at the next start, an interrupted import resumes where it stopped, and only new or modified passages are computed. `NANOPRUNE_DISK_CACHE=0` keeps nothing on disk; deleting that folder clears it (`nanoprune info` shows its size).

The page talks to a small same-origin JSON API: `GET /api/status` (index, model, running import), `POST /api/search`, `POST /api/load_folder` and `POST /api/index_direct` (they answer `202` with an import job; add `"wait": true` to get the final report), `POST /api/cancel` and `POST /api/open`.

Files are read and indexed by the local server only. The server:

- listens on `127.0.0.1` only and rejects requests whose `Host` is not that address (blocks DNS rebinding);
- rejects cross-site requests (`Origin` / `Sec-Fetch-Site`), requires `application/json` bodies, and never sends CORS headers, so other websites open in your browser cannot read your files through it;
- serves the page with a strict Content-Security-Policy and renders document text as text, so a crafted document cannot run script in the page;
- opens only files that are part of the current index, with the system's default application and without a shell;
- loads no third-party resources (no web fonts or CDN).

## Semantic search

`SemanticScorer` embeds the query and the passages with a sentence-embedding model (mean pooling, L2-normalised) and turns their cosine similarity into a probability with Platt scaling fitted on the **dev** suite only; the held-out suite is never used for tuning. Any ONNX export with a Hugging Face `tokenizer.json` works (for example `multilingual-e5-small` exported with `optimum-cli`), after `nanoprune calibrate --dense DIR`.

What it does well, measured on the held-out suite: 50/50 direct answers, 45/50 paraphrases (keywords find 0–3), 50/50 unrelated passages rejected, calibration error (ECE) 6 %. What it does not do well:

- **lexical traps**: 25/50 passages about another question with the same vocabulary are still accepted;
- **negation**: "ne supporte pas les antibiotiques" ranks "aucune allergie aux antibiotiques" high; embeddings capture the topic more than the polarity;
- **one-topic folders**: in a folder of medical records, every record is topically close to a medical query; the ranking stays right, but scores are high, so use the *Strict* filter for precision.

A cross-encoder re-ranker (or Laya) on the top results is the natural next step for traps and negation. Combining the semantic score with keyword scores was tested and *reduced* held-out accuracy (to 60–68 %), because the dev suite rewards shared words.

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

Results at threshold 0.5, from `nanoprune eval` with the semantic model and the published v0.3 checkpoint (95 % bootstrap intervals in the command output):

| Scorer | dev accuracy | dev AUC | held-out accuracy | held-out AUC | held-out traps accepted | held-out paraphrases found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always "relevant" | 50.0 % | 0.50 | 50.0 % | 0.50 | 50/50 | 50/50 |
| Keyword heuristic (no-model fallback) | 90.0 % | 0.87 | 55.0 % | 0.65 | 43/50 | 3/50 |
| Lexical overlap (cascade gate) | 84.0 % | 0.98 | 57.5 % | 0.64 | 35/50 | 0/50 |
| IDF-weighted coverage | 72.0 % | 0.97 | 64.5 % | 0.63 | 21/50 | 0/50 |
| BM25 (ranking only) | — | 0.97 | — | 0.62 | — | — |
| NanoPrune v0.3 (published) | 46.0 % | 0.52 | 43.0 % | 0.41 | 18/50 | 7/50 |
| NanoPrune v0.4 (recorded, not published) | 60 % | — | not measured | — | — | — |
| **Semantic: multilingual-e5-large int8** | 100 %\* | 1.00\* | **85.0 %** [80–90] | **0.93** | 25/50 | **45/50** |

\* calibrated on the dev suite, so its dev numbers are not a test result.

The v0.4 cascade was recorded at 94 % on the dev suite (Laya alone: 92 %), but its thresholds were tuned on that suite, and the lexical-overlap score it uses as a gate already reaches 84 % there on its own. `nanoprune eval --laya` reruns the cascade with the NanoPrune score replaced by a constant (0.5, then 0.72). If the real cascade does not beat these ablations, the neural score adds nothing to it.

## Training

The pipeline (v0.5) fixes the problems of the v0.4 scripts: the train/validation split leaked titles and passages, "hard" negatives were random (a shared "de" was enough), the ONNX export call was broken, and there was no calibration. Every step is tested end to end on a synthetic corpus.

```bash
# 1. WordPiece tokenizer, trained on text normalised exactly like at inference
python scripts/train_tokenizer.py --tsv data/legal_raw.tsv
# 2. pairs, split by document before pairing; BM25-mined hard negatives
python scripts/generate_contrastive_dataset.py --tsv data/legal_raw.tsv [--queries real_queries.jsonl]
# 3. optional: teacher labels for distillation (resumable): Laya, or the semantic model
python scripts/distill_teacher.py [--teacher dense]
# 4. train, select on validation AUC, calibrate the temperature, export ONNX, write the manifest
python scripts/train_contrastive.py --name nanoprune-v0.5 [--suffix .teacher --teacher-weight 0.5]
```

The main limitation remains the data: titles are used as queries unless you supply real search queries (`--queries`). A model trained from scratch on a few thousand pairs learns little language; distilling the semantic model (MIT licence) or Laya (Apache 2.0) into a small encoder is the most promising way to get a model much smaller than 552 MB. The v0.1–v0.3 scripts are kept in [`scripts/legacy/`](scripts/legacy/README.md).

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
pip install -e ".[all]"   # or just `pip install -e .`: tests needing torch/onnx/tokenizers/pypdf are skipped
python -m unittest discover -s tests -p "test_*.py" -v
```

## License

MIT © [Junior Diomande](https://github.com/dmdjr1409). See [CHANGELOG.md](CHANGELOG.md) for the history of changes.
