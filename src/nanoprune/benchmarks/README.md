# NanoPrune evaluation suites

JSON Lines files, one labelled `(query, doc)` pair per line:

```json
{"id": "ho-01a", "category": "direct_positive", "domain": "travail", "query": "...", "doc": "...", "label": 1}
```

Run them with `nanoprune eval` (or `nanoprune.evaluation.run_standard_evaluation`).

## `dev_legal_fr_50.jsonl` — development set (50 pairs)

The 50 cases of the original `scripts/benchmark_deep_comparison.py`, unchanged:
15 direct positives, 15 obvious negatives, 10 hard negatives ("traps") and 10
subtle positives, French legal texts.

- The v0.4 cascade thresholds and rescue rule were **tuned on this set**: its
  scores are development scores, not generalisation results.
- It is lexically easy: every obvious negative shares no word with its query,
  and plain BM25 ranks the set with an AUC above 0.95.

## `heldout_fr_v1.jsonl` — held-out set (200 pairs)

50 French queries (labour, housing, family, consumer, data protection,
criminal, administrative, tax and company law, plus medical-record notes),
each paired with four documents:

| Category | Label | Construction rule (checked automatically) |
| --- | :---: | --- |
| `direct_positive` | 1 | answers the query and shares at least 2 of its content words |
| `paraphrase_positive` | 1 | answers the query with different wording: at most 1 shared content word |
| `lexical_trap_negative` | 0 | shares at least 2 content words but concerns another question |
| `unrelated_negative` | 0 | another topic, no shared content word |

It is adversarial to keyword matching by construction: lexical baselines stay
near 55–65 % accuracy because they miss the paraphrases and accept the traps.
That is the gap a neural re-ranker has to close.

Rules:

- **Never tune anything on this set** (thresholds, rules, checkpoints,
  prompts). Use `dev` or your own validation data for that; report `heldout`
  numbers only once a configuration is frozen.
- The pairs were written in September 2026 by an AI assistant (Claude) from
  general knowledge of French law and clinical notes. They are synthetic, not
  legal advice, and a few labels may be debatable: report or fix doubtful items
  and bump the file version (`heldout_fr_v2.jsonl`) rather than editing v1 in
  place, so that published numbers stay comparable.
