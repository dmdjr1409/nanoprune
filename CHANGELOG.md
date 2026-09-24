# Changelog

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
