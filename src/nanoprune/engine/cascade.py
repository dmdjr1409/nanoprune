import os
import re
import time
import warnings
from typing import List, Tuple, Dict, Any, Optional

from ..errors import NanoPruneWarning
from .pruner import NanoPruner

DEFAULT_LAYA_MODEL = "convaiinnovations/laya"
# Question sent to Laya for each ambiguous candidate. The cascade thresholds were
# tuned with this exact wording (the "pure Laya" benchmark used a slightly
# different one, see nanoprune.evaluation.LAYA_QUESTION).
CASCADE_LAYA_QUESTION = "Ce document apporte-t-il une réponse pertinente à : '{query}' ?"


class HybridCascadePruner:
    """
    Two-tier System One cascade:
    - Tier 1: a fast scorer (NanoPrune) plus a lexical-overlap gate drops obvious
      negatives and keeps very confident matches.
    - Tier 2: Laya arbitrates the ambiguous candidates.

    Latency and dispatch rates depend on hardware and data; measure them with
    ``nanoprune eval --laya`` rather than relying on hard-coded claims. Run the
    same command's ablations to check how much the NanoPrune score contributes.
    """
    def __init__(
        self,
        nanoprune_model: Optional[Any] = None,
        drop_threshold: float = 0.75,
        keep_threshold: float = 0.95,
        enable_laya: bool = True,
        laya_model: Optional[str] = None,
        laya_agent: Optional[Any] = None,
    ):
        """
        Args:
            nanoprune_model: tier-1 scorer, any object with ``score(query, candidates)``
                (default: ``NanoPruner.load()``).
            laya_model: Laya checkpoint to load (default: $NANOPRUNE_LAYA_MODEL or
                "convaiinnovations/laya", the English ModernBERT checkpoint; Convai
                also publishes a multilingual variant worth testing on French).
            laya_agent: an already loaded Laya agent (skips loading).
        """
        self.drop_threshold = drop_threshold
        self.keep_threshold = keep_threshold
        self.nanoprune = nanoprune_model or NanoPruner.load()
        self.laya_model = laya_model or os.environ.get("NANOPRUNE_LAYA_MODEL", DEFAULT_LAYA_MODEL)

        self.laya_agent = laya_agent
        if self.laya_agent is None and enable_laya:
            try:
                import laya
                self.laya_agent = laya.load(self.laya_model)
            except Exception as e:
                warnings.warn(
                    f"Laya not loaded ({e}); ambiguous candidates are decided by the tier-1 score alone.",
                    NanoPruneWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _lexical_overlap(query: str, doc: str) -> float:
        stop = {
            "le", "la", "les", "un", "une", "des", "du", "de", "d", "en", "pour",
            "et", "à", "au", "aux", "par", "dans", "sur", "ce", "cette", "ces"
        }
        q_words = set(re.findall(r"\w+", query.lower())) - stop
        d_words = set(re.findall(r"\w+", doc.lower())) - stop
        if not q_words:
            return 0.0
        q_stems = {w[:5] for w in q_words}
        d_stems = {w[:5] for w in d_words}
        return len(q_stems.intersection(d_stems)) / len(q_stems)

    def prune_cascade(
        self,
        query: str,
        candidates: List[str],
    ) -> Dict[str, Any]:
        """
        Executes the two-tier cascade over candidates.
        """
        t_start = time.perf_counter()
        if not candidates:
            return {
                "retained": [],
                "dropped_count": 0,
                "stats": {"total_candidates": 0, "tier1_fast_dropped": 0, "tier1_fast_kept": 0, "tier2_arbitrated": 0},
                "latency_ms": 0.0,
            }

        # --- Tier 1: fast pass over all candidates ---
        t1_start = time.perf_counter()
        tier1_scores = self.nanoprune.score(query, candidates)
        t1_elapsed = (time.perf_counter() - t1_start) * 1000.0

        tier1_dropped: List[Tuple[str, float, str]] = []
        tier1_kept: List[Tuple[str, float, str]] = []
        ambiguous_indices: List[int] = []

        for idx, (cand, score) in enumerate(zip(candidates, tier1_scores)):
            lex_ov = self._lexical_overlap(query, cand)
            # Safe fast drop: zero lexical overlap AND tier-1 score below threshold
            if lex_ov == 0.0 and score < self.drop_threshold:
                tier1_dropped.append((cand, score, "tier1_fast_dropped"))
            elif score >= self.keep_threshold:
                tier1_kept.append((cand, score, "tier1_match"))
            else:
                ambiguous_indices.append(idx)

        # --- Tier 2: asymmetric arbitration (Laya + NanoPrune rescue) ---
        t2_start = time.perf_counter()
        tier2_results: List[Tuple[str, float, str]] = []
        laya_calls = 0

        if ambiguous_indices and self.laya_agent is not None:
            for idx in ambiguous_indices:
                cand = candidates[idx]
                np_score = tier1_scores[idx]
                state = f"Document: {cand}"
                questions = {
                    "relevance": {
                        "type": "noul",
                        "instructions": CASCADE_LAYA_QUESTION.format(query=query),
                    }
                }
                try:
                    laya_calls += 1
                    res = self.laya_agent.system_one(state, questions)
                    laya_prob = float(res["answers"]["relevance"]["noul"])
                    lex_ov = self._lexical_overlap(query, cand)

                    # 1. Laya considers the candidate relevant
                    if laya_prob >= 0.50:
                        tier2_results.append((cand, laya_prob, "tier2_laya_kept"))
                    # 2. Asymmetric rescue: Laya hesitates (0.25-0.50) but the candidate is
                    #    lexically anchored (lex_ov >= 0.20) and tier 1 is confident (>= 0.70)
                    elif 0.25 <= laya_prob < 0.50 and lex_ov >= 0.20 and np_score >= 0.70:
                        rescued_score = round(0.50 * laya_prob + 0.50 * np_score, 4)
                        tier2_results.append((cand, rescued_score, "tier2_rescued_by_nanoprune"))
                    # 3. Reject: hard negative traps (Laya < 0.25) or cross-domain noise
                    else:
                        tier1_dropped.append((cand, laya_prob, "tier2_laya_dropped"))
                except Exception:
                    if np_score >= 0.50:
                        tier2_results.append((cand, np_score, "tier1_fallback_kept"))
                    else:
                        tier1_dropped.append((cand, np_score, "tier1_fallback_dropped"))
        else:
            # Without Laya, boundary cases are decided by a 0.50 cutoff on the tier-1 score
            for idx in ambiguous_indices:
                cand = candidates[idx]
                np_score = tier1_scores[idx]
                if np_score >= 0.50:
                    tier2_results.append((cand, np_score, "tier1_direct_kept"))
                else:
                    tier1_dropped.append((cand, np_score, "tier1_direct_dropped"))

        t2_elapsed = (time.perf_counter() - t2_start) * 1000.0
        total_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Combine kept results and sort by score descending
        retained = [(c, s, tag) for c, s, tag in tier1_kept + tier2_results]
        retained.sort(key=lambda x: x[1], reverse=True)

        return {
            "retained": retained,
            "dropped_count": len(tier1_dropped),
            "latency_ms": round(total_elapsed, 2),
            "stats": {
                "total_candidates": len(candidates),
                "tier1_fast_dropped": sum(1 for x in tier1_dropped if x[2] == "tier1_fast_dropped"),
                "tier1_fast_kept": len(tier1_kept),
                "tier2_arbitrated": len(ambiguous_indices),
                "laya_calls": laya_calls,
                "tier1_latency_ms": round(t1_elapsed, 2),
                "tier2_latency_ms": round(t2_elapsed, 2),
                "total_latency_ms": round(total_elapsed, 2),
                "efficiency_gain_pct": round((1.0 - (len(ambiguous_indices) / max(1, len(candidates)))) * 100, 1),
            }
        }
