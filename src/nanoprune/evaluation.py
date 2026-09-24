"""Evaluate relevance scorers on labelled (query, document) suites.

A scorer is any callable ``scorer(query, documents) -> scores`` with scores in
[0, 1] (``NanoPruner.score`` has this signature). Suites are JSON Lines files
with one case per line::

    {"id": "...", "category": "...", "query": "...", "doc": "...", "label": 0 or 1}

Two suites ship with the package (``nanoprune/benchmarks``):

- ``dev``: the 50-case development set the v0.4 cascade was tuned on;
- ``heldout``: 50 queries x 4 documents that were never used for tuning,
  including paraphrased positives and lexical traps (see the README there).
"""
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .core.calibration import compute_ece

SUITES_DIR = Path(__file__).parent / "benchmarks"
BUNDLED_SUITES = {
    "dev": SUITES_DIR / "dev_legal_fr_50.jsonl",
    "heldout": SUITES_DIR / "heldout_fr_v1.jsonl",
}

Scorer = Callable[[str, List[str]], List[float]]


@dataclass
class Case:
    id: str
    category: str
    query: str
    doc: str
    label: int
    note: Optional[str] = None

    @property
    def is_trap(self) -> bool:
        return "trap" in self.category


def load_suite(path_or_name: Union[str, Path]) -> List[Case]:
    """Load a bundled suite by name ("dev", "heldout") or any JSONL file."""
    path = BUNDLED_SUITES.get(str(path_or_name), Path(path_or_name))
    cases = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"{path}:{number}: label must be 0 or 1")
            cases.append(Case(
                id=str(row.get("id", number)),
                category=str(row.get("category", "uncategorized")),
                query=row["query"],
                doc=row["doc"],
                label=label,
                note=row.get("note"),
            ))
    return cases


def score_cases(cases: Sequence[Case], scorer: Scorer) -> List[float]:
    """Score every case, batching the documents that share a query."""
    by_query: Dict[str, List[int]] = {}
    for i, case in enumerate(cases):
        by_query.setdefault(case.query, []).append(i)
    scores = [0.0] * len(cases)
    for query, indices in by_query.items():
        for i, score in zip(indices, scorer(query, [cases[i].doc for i in indices])):
            scores[i] = float(score)
    return scores


# ---------------------------------------------------------------------- metrics

def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """Probability that a random positive outscores a random negative (ties count 1/2)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    # Rank-based Mann-Whitney U with average ranks for ties.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    tp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y == 0)
    tn = sum(1 for s, y in zip(scores, labels) if s < threshold and y == 0)
    fn = sum(1 for s, y in zip(scores, labels) if s < threshold and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    metric: Callable[[Sequence[int], Sequence[float]], Optional[float]],
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """Percentile bootstrap confidence interval of ``metric``."""
    if n_resamples <= 0 or not labels:
        return None
    rng = random.Random(seed)
    n = len(labels)
    values = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        value = metric([labels[i] for i in idx], [scores[i] for i in idx])
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    lo = values[int(math.floor(alpha / 2 * (len(values) - 1)))]
    hi = values[int(math.ceil((1 - alpha / 2) * (len(values) - 1)))]
    return lo, hi


def query_auc(cases: Sequence[Case], scores: Sequence[float]) -> Optional[float]:
    """Mean within-query AUC: how well each query's positives outrank its negatives."""
    groups: Dict[str, List[int]] = {}
    for i, case in enumerate(cases):
        groups.setdefault(case.query, []).append(i)
    values = []
    for indices in groups.values():
        value = roc_auc([cases[i].label for i in indices], [scores[i] for i in indices])
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def evaluate_scores(
    cases: Sequence[Case],
    scores: Sequence[float],
    threshold: Optional[float] = 0.5,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """All metrics for one scorer. ``threshold=None`` skips threshold metrics (e.g. raw BM25)."""
    labels = [c.label for c in cases]
    result: Dict[str, Any] = {"n": len(cases), "auc": roc_auc(labels, scores), "query_auc": query_auc(cases, scores)}
    result["auc_ci"] = bootstrap_ci(labels, scores, roc_auc, n_bootstrap)
    if threshold is not None:
        result["threshold"] = threshold
        result.update(binary_metrics(labels, scores, threshold))
        result["accuracy_ci"] = bootstrap_ci(
            labels, scores, lambda y, s: binary_metrics(y, s, threshold)["accuracy"], n_bootstrap
        )
        in_unit = all(0.0 <= s <= 1.0 for s in scores)
        if in_unit:
            result.update({k: v for k, v in compute_ece(scores, labels).items() if k in ("ece", "brier_score")})
        per_category: Dict[str, List[int]] = {}
        for case, score in zip(cases, scores):
            per_category.setdefault(case.category, []).append(int((score >= threshold) == bool(case.label)))
        result["per_category"] = {k: (sum(v), len(v)) for k, v in sorted(per_category.items())}
        traps = [score >= threshold for case, score in zip(cases, scores) if case.is_trap]
        if traps:
            result["traps_fooled"] = (sum(traps), len(traps))
    return result


def evaluate_scorer(cases: Sequence[Case], scorer: Scorer, **kwargs) -> Dict[str, Any]:
    return evaluate_scores(cases, score_cases(cases, scorer), **kwargs)


# ---------------------------------------------------------------------- baselines

def constant_scorer(value: float) -> Scorer:
    return lambda query, docs: [value] * len(docs)


def keyword_heuristic_scorer() -> Scorer:
    """The fallback scorer NanoPruner uses when no weights are loaded."""
    from .engine.pruner import NanoPruner
    return NanoPruner(model_path=None).score


def lexical_overlap_scorer() -> Scorer:
    """Share of query stems found in the document (the cascade's tier-1 gate)."""
    from .engine.cascade import HybridCascadePruner
    return lambda query, docs: [HybridCascadePruner._lexical_overlap(query, d) for d in docs]


def idf_coverage_scorer(corpus: Iterable[str]) -> Scorer:
    """IDF-weighted share of the query terms present in the document."""
    from .engine.lexical import idf_table, idf_weighted_coverage
    idf = idf_table(corpus)
    return lambda query, docs: [idf_weighted_coverage(query, d, idf) for d in docs]


def bm25_scorer(corpus: Sequence[str]) -> Scorer:
    """Raw BM25 against the whole suite's documents (unbounded: AUC only)."""
    from .engine.lexical import BM25Index
    corpus = list(dict.fromkeys(corpus))
    index = BM25Index(corpus)
    position = {doc: i for i, doc in enumerate(corpus)}

    def score(query: str, docs: List[str]) -> List[float]:
        all_scores = index.scores(query)
        return [all_scores[position[d]] if d in position else 0.0 for d in docs]

    return score


def cascade_scorer(cascade) -> Tuple[Scorer, Dict[str, int]]:
    """Wrap ``HybridCascadePruner.prune_cascade``; also counts Laya calls."""
    stats = {"calls": 0, "laya_calls": 0}

    def score(query: str, docs: List[str]) -> List[float]:
        out = []
        for doc in docs:
            res = cascade.prune_cascade(query, [doc])
            stats["calls"] += 1
            stats["laya_calls"] += res["stats"].get("laya_calls", 0)
            out.append(res["retained"][0][1] if res["retained"] else 0.0)
        return out

    return score, stats


# ---------------------------------------------------------------------- reporting

def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.1f}"


def _ci(ci: Optional[Tuple[float, float]]) -> str:
    return "" if not ci else f" [{ci[0] * 100:.0f}–{ci[1] * 100:.0f}]"


def format_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Markdown table of the results of several scorers on one suite."""
    lines = [
        "| Scorer | Accuracy % [95% CI] | Precision % | Recall % | F1 % | AUC % [95% CI] | Query AUC % | Traps fooled | ECE % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in results.items():
        traps = r.get("traps_fooled")
        if "laya_call_rate" in r:
            name = f"{name} — Laya on {r['laya_call_rate'] * 100:.0f}% of docs"
        lines.append(
            f"| {name} | "
            f"{_pct(r.get('accuracy'))}{_ci(r.get('accuracy_ci'))} | "
            f"{_pct(r.get('precision')) if 'precision' in r else '—'} | "
            f"{_pct(r.get('recall')) if 'recall' in r else '—'} | "
            f"{_pct(r.get('f1')) if 'f1' in r else '—'} | "
            f"{_pct(r.get('auc'))}{_ci(r.get('auc_ci'))} | "
            f"{_pct(r.get('query_auc'))} | "
            f"{f'{traps[0]}/{traps[1]}' if traps else '—'} | "
            f"{_pct(r.get('ece')) if 'ece' in r else '—'} |"
        )
    return "\n".join(lines)


def format_categories(results: Dict[str, Dict[str, Any]]) -> str:
    """Markdown table of per-category accuracy."""
    categories: List[str] = []
    for r in results.values():
        for cat in r.get("per_category", {}):
            if cat not in categories:
                categories.append(cat)
    if not categories:
        return ""
    lines = ["| Scorer | " + " | ".join(categories) + " |", "| --- |" + " ---: |" * len(categories)]
    for name, r in results.items():
        per = r.get("per_category", {})
        cells = [f"{per[c][0]}/{per[c][1]}" if c in per else "—" for c in categories]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _model_label(model, suite_name: str) -> str:
    info = model.describe()
    label = f"{info.get('backend')} model ({info.get('model_name')})"
    fitted_on = (info.get("calibration") or {}).get("fitted_on")
    if fitted_on == suite_name:
        label += " [calibrated on this suite]"
    return label


def run_standard_evaluation(
    suites: Sequence[str] = ("dev", "heldout"),
    pruner=None,
    laya_agent=None,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    models: Optional[Sequence[Any]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Evaluate the baselines, the given models, Laya and the cascade (if given).

    ``pruner`` / ``models`` are relevance scorers with a ``score`` method
    (NanoPruner, SemanticScorer...); scorers without a model are skipped.
    The cascade ablation replaces the tier-1 score by a constant: 0.5 (only the
    lexical gate and Laya remain) and 0.72 (the lexical "rescue" rule also fires).
    If the real cascade does not beat both, the tier-1 model adds nothing.
    """
    candidates = list(models or []) + ([pruner] if pruner is not None else [])
    loaded = [m for m in candidates if getattr(m, "has_model", False)]
    from .engine.cascade import HybridCascadePruner

    report: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for suite_name in suites:
        cases = load_suite(suite_name)
        corpus = [c.doc for c in cases]
        cascade_stats: Dict[str, Dict[str, int]] = {}
        scorers: Dict[str, Tuple[Scorer, Optional[float]]] = {
            "always relevant": (constant_scorer(1.0), threshold),
            "keyword heuristic (no model)": (keyword_heuristic_scorer(), threshold),
            "lexical overlap (cascade gate)": (lexical_overlap_scorer(), threshold),
            "IDF-weighted coverage": (idf_coverage_scorer(corpus), threshold),
            "BM25 (ranking only)": (bm25_scorer(corpus), None),
        }
        for model in loaded:
            scorers[_model_label(model, suite_name)] = (model.score, threshold)
        if laya_agent is not None:
            scorers["Laya"] = (laya_scorer(laya_agent), threshold)
            tier1_variants = [(f"cascade (tier-1: {m.describe().get('model_name')})", m) for m in loaded]
            tier1_variants += [
                ("cascade, tier-1 = 0.5 (ablation)", _ConstantPruner(0.5)),
                ("cascade, tier-1 = 0.72 (ablation)", _ConstantPruner(0.72)),
            ]
            for name, tier1 in tier1_variants:
                cascade = HybridCascadePruner(nanoprune_model=tier1, enable_laya=False, laya_agent=laya_agent)
                scorer, stats = cascade_scorer(cascade)
                scorers[name] = (scorer, threshold)
                cascade_stats[name] = stats

        report[suite_name] = {}
        for name, (scorer, th) in scorers.items():
            report[suite_name][name] = evaluate_scorer(cases, scorer, threshold=th, n_bootstrap=n_bootstrap)
            if name in cascade_stats:
                calls = cascade_stats[name]
                report[suite_name][name]["laya_call_rate"] = calls["laya_calls"] / max(1, calls["calls"])
    return report


class _ConstantPruner:
    """Stand-in tier-1 scorer for cascade ablations."""

    def __init__(self, value: float):
        self.value = value

    def score(self, query: str, candidates: List[str]) -> List[float]:
        return [self.value] * len(candidates)


LAYA_QUESTION = "Ce document apporte-t-il une réponse pertinente à la recherche : '{query}' ?"


def laya_scorer(laya_agent) -> Scorer:
    """Relevance probability from a Laya ``system_one`` agent (``noul`` question)."""
    def score(query: str, docs: List[str]) -> List[float]:
        out = []
        for doc in docs:
            res = laya_agent.system_one(
                f"Document: {doc}",
                {"relevance": {"type": "noul", "instructions": LAYA_QUESTION.format(query=query)}},
            )
            out.append(float(res["answers"]["relevance"]["noul"]))
        return out
    return score
