# Changelog

## 0.7.0 — 2026-09-24

The local search app, reworked for everyday use.

### Added

- Background imports: a progress bar (files read, passages analysed, time
  left) and a Cancel button; the previous folder stays searchable until the new
  index is ready. `POST /api/load_folder` and `/api/index_direct` answer `202`
  with an import job (`"wait": true` gives the final report as before),
  `POST /api/cancel` stops it and `GET /api/status` reports it.
- Embedding cache on disk (`EmbeddingStore`, SQLite under
  `~/.cache/nanoprune/embeddings`): a folder analysed once is ready in seconds
  at the next start, an interrupted import resumes where it stopped, and only
  new or modified passages are computed. `NANOPRUNE_DISK_CACHE=0` disables it.
- Results grouped by file, with the query words highlighted
  (`highlight_terms`), the path relative to the indexed folder (`rel_path`),
  "open the file" (`POST /api/open`: files of the current index only, default
  application, no shell) and "copy the citation" actions, the closest passages
  below the filter (`near_misses`), `matches_total` / `more_available` and a
  "show more" button.
- Page: onboarding when no folder is loaded, drop files or folders anywhere,
  Large / Normal / Strict filter, recent searches, example questions on the
  sample folder, keyboard shortcuts (`/`, `Ctrl+K`, arrows, `Enter`, `O`, `C`,
  `Esc`, `?`), help panel, light and dark themes, phone-sized layout, and a
  report of skipped files (unsupported, too large, no readable text).
- `nanoprune app` answers immediately, indexes in the background and opens the
  browser on desktop sessions (`--no-browser`). `nanoprune search` shows
  progress on a terminal, prints paths relative to the folder, explains empty
  results and has `--json`. `nanoprune --help` starts with a quick start, and
  `nanoprune info` shows the embedding cache.
- API: `progress` / `should_stop` callbacks on
  `LocalDocumentIndexer.index_directory`, `DenseEncoder.encode` and
  `LocalSearchEngine.prepare`, `OperationCancelled`,
  `SemanticScorer.embed` / `score_embeddings`, `SemanticScorer.load(disk_cache=True)`.
- Release workflow: pushing a `vX.Y.Z` tag, or running the workflow with the
  version, checks that the version matches the package and this changelog,
  builds and checks the wheel and the sdist, and publishes the GitHub release
  with this changelog section as notes (`scripts/release_notes.py`).
- 22 new tests (119 in total): background jobs, cancellation, file opening,
  grouping and quote selection, the disk store, the CLI options, the release
  notes.

### Changed

- The sentence quoted for a result weighs query words by rarity and ignores
  words found in no document; when no sentence carries enough of the query
  (typical of semantic matches), the whole passage is quoted instead of a
  sentence that only shares a common word.
- A passage that mostly repeats a better-ranked neighbour (chunk overlap) is
  not listed twice.
- The search engine keeps the embedding matrix of the index and scores it with
  one matrix product; `SemanticScorer` is thread-safe, caches query vectors,
  sorts batches by length (less padding) and keeps 20,000 passages in memory
  (the disk store holds the rest).
- Uploaded files without readable text are reported as skipped.

## 0.6.0 — 2026-09-24

Semantic search: the first scorer that clearly beats keyword matching on the
held-out suite (85.0 % accuracy vs 55–64.5 %, AUC 0.93 vs 0.62–0.65).

### Added

- `SemanticScorer` (`nanoprune.engine.dense`): a pretrained sentence-embedding
  model run with onnxruntime and tokenizers (no PyTorch), mean/CLS pooling,
  query/passage prefixes, passage-embedding cache, and Platt calibration
  fitted on the dev suite (`fit_platt`). Same interface as `NanoPruner`.
- `nanoprune download --dense multilingual-e5-large`: SHA-256-pinned download
  of the ONNX export of intfloat/multilingual-e5-large (MIT), safe archive
  extraction, int8 quantisation (552 MB instead of 2.2 GB, about 2x faster,
  same accuracy) and calibration.
- `nanoprune calibrate --dense DIR` for any other ONNX embedding model;
  `--dense PATH|auto` / `--no-dense` on every model-using command. An installed
  semantic model is used by default by `search`, `prune`, `app` and `eval`.
- Search engine: semantic scorers score every passage (keyword-free matches are
  found), and passage embeddings are computed once at import time.
- `nanoprune eval` compares several models side by side and marks results on
  the suite a model was calibrated on.
- Distillation: `scripts/distill_teacher.py --teacher dense` labels training
  pairs with the semantic model instead of Laya.
- `semantic` extra; 11 new tests (97 in total).

### Measured (held-out suite, threshold 0.5)

- Semantic (multilingual-e5-large int8): 85.0 % [80–90], AUC 0.93, 45/50
  paraphrases found, 25/50 lexical traps still accepted.
- Mixing the semantic score with keyword features, fitted on dev, lowered
  held-out accuracy to 60–68 %: not shipped.

## 0.5.0 — 2026-09-24

Package release focused on security, honesty about what is running, and
measurement. The model weights are unchanged (v0.4).

### Security

- Local app: a web page open in the browser could make the server index any
  folder and read the files back (`Access-Control-Allow-Origin: *` and JSON
  parsed from `text/plain` "simple" requests). The server now rejects foreign
  `Host` headers (DNS rebinding), cross-site `Origin`/`Sec-Fetch-Site`,
  non-JSON bodies and oversized requests, and never sends CORS headers.
- Local app: document contents were inserted with `innerHTML` (stored XSS
  through indexed files). The UI now builds the DOM from text nodes, lives in
  `app.js`/`app.css` under a strict Content-Security-Policy with no inline
  script, and sets `X-Frame-Options`, `nosniff` and `no-referrer`.
- The page no longer loads Google Fonts: nothing leaves the machine.
- PyTorch checkpoints are loaded with `weights_only=True`; `.docx` parsing
  refuses oversized archives.

### Fixed

- Drag-and-drop import emptied the index and indexed nothing
  (`_chunk_content` results were never stored).
- `tokenizers` was missing from the dependencies: with v0.4 weights the
  tokenizer silently fell back to a 321-token vocabulary and produced
  meaningless scores. It is now declared, and a model/tokenizer vocabulary
  mismatch raises `TokenizerMismatchError`.
- Without weights, NanoPrune silently used the keyword heuristic while the app
  announced "NanoPrune v0.4 Transformer (5.45M parameters)" and "% certifié".
  The backend is now reported everywhere and a `NanoPruneWarning` is emitted;
  `strict=True` / `--strict` fail instead.
- `choice()` returned the first option with "50 % confidence" and
  `score_rubric()` returned text length / 200 when no model was loaded; they
  now raise `HeadUnavailableError`.
- The published v0.3 checkpoint could not be loaded (head size 64 vs 128):
  the architecture is now read from the weights.
- ONNX export: `train_contrastive.py` passed a wrong argument (no ONNX was
  produced); recent PyTorch versions wrote an external `.onnx.data` file that
  was not shipped; exporting with all heads left the model in training mode.
  Exports are now single-file and verified against PyTorch.
- A broken discovered ONNX file (missing `.onnx.data`) no longer prevents
  loading: the next candidate (e.g. the `.pt`) is used.
- Cascade statistics counted every tier-1 drop as a "fast drop".
- Temperature scaling used a fixed-step gradient descent that could stop far
  from the optimum; it now uses a golden-section search.

### Added

- BM25 retrieval (`nanoprune.engine.lexical`) with French/English stopwords,
  accent folding and light stemming, used to pre-select passages.
- Indexer: real chunk overlap, line numbers, `.docx` (standard library) and
  `.pdf` (`pypdf`, extra `docs`) support, skipped hidden/dependency folders,
  file size and count limits, `index_text()`.
- UI: recursive folder drag-and-drop, file picker, threshold slider, line
  numbers, import report, heuristic-mode banner.
- `nanoprune.evaluation` and `nanoprune eval`: accuracy, precision, recall,
  F1, AUC, per-query AUC, ECE, bootstrap confidence intervals, per-category
  results, trap counts, lexical baselines, Laya and cascade ablations.
- Evaluation suites: the 50-case dev set as JSONL and a new 200-pair held-out
  set with paraphrases and lexical traps.
- `nanoprune download` (GitHub release assets, SHA-256 verified),
  `nanoprune info`, `--model`, `--tokenizer`, `--strict`.
- Model manifests tying weights to their tokenizer and architecture;
  `scripts/package_release.py` to publish weights.
- Training pipeline in `nanoprune.training`: document-level train/val/test
  split (no leakage), BM25-mined hard negatives, consistent category labels,
  resumable Laya labelling, validation-based model selection, temperature
  calibration, manifest writing.
- 86 unit tests (server attacks, imports, loading, export parity, evaluation,
  cascade decisions, download, CLI, end-to-end training) and a GitHub Actions
  workflow.

### Changed

- Search results expose `score`/`score_pct` (`confidence` kept as an alias),
  `line_start`/`line_end`, `bm25`, and `text` is the passage without the
  `[Title]` prefix.
- The cascade takes `laya_model` / `laya_agent` and any tier-1 object with a
  `score()` method.
- v0.1–v0.3 training scripts moved to `scripts/legacy/`.
