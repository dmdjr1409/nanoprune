# Legacy training scripts (v0.1–v0.3)

Kept for history; they are not maintained and are superseded by the v0.5
pipeline (`scripts/generate_contrastive_dataset.py`, `scripts/distill_teacher.py`,
`scripts/train_contrastive.py`).

| Script | Produced | Known problems |
| --- | --- | --- |
| `train_v01.py` | `nanoprune-v0.1` (2 layers, char tokenizer) | 21 hand-written pairs repeated 15 times with prefixes; the random validation split contains copies of training pairs |
| `train_legal_v02.py` | `nanoprune-legal-v0.2` | title → own-content positives only, random negatives, random split (title/content leakage) |
| `train_distilled_v03.py` | `nanoprune-v0.3` (495k parameters) | trained on Laya soft labels, but the published checkpoint scores 46 % on the dev suite (AUC 0.52): it does not separate relevant from irrelevant passages |
| `verify_v03.py` | — | smoke test of the three primitives on four examples |

They were adjusted only so that they still run from this folder: `src/` path,
`head_hidden=64` (the v0.1–v0.3 head size) and the char-level tokenizer they
were trained with.
