import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from .pruner import NanoPruner

class HybridCascadePruner:
    """
    Two-Tier Production System One Cascade:
    - Tier 1: NanoPrune v0.3 (1.4 ms) filters 85-90% of unambiguous noise and clear matches.
    - Tier 2: Laya 421M (51 ms) arbitrates only ambiguous boundary cases [low_thresh, high_thresh].
    Combines sub-millisecond throughput with 421M-parameter semantic depth.
    """
    def __init__(
        self,
        nanoprune_model: Optional[NanoPruner] = None,
        drop_threshold: float = 0.35,
        keep_threshold: float = 0.75,
        enable_laya: bool = True,
    ):
        self.drop_threshold = drop_threshold
        self.keep_threshold = keep_threshold
        self.nanoprune = nanoprune_model or NanoPruner.load()

        self.laya_agent = None
        if enable_laya:
            try:
                import laya
                self.laya_agent = laya.load("convaiinnovations/laya")
            except Exception as e:
                print(f"[HybridCascade] Note: Laya not loaded ({e}), falling back to single-tier.")

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
                "stats": {"total": 0, "tier1_dropped": 0, "tier1_kept": 0, "tier2_arbitrated": 0},
                "latency_ms": 0.0,
            }

        # --- Tier 1: NanoPrune Fast Pass (All candidates) ---
        t1_start = time.perf_counter()
        tier1_scores = self.nanoprune.score(query, candidates)
        t1_elapsed = (time.perf_counter() - t1_start) * 1000.0

        tier1_dropped = []
        tier1_kept = []
        ambiguous_indices = []

        for idx, (cand, score) in enumerate(zip(candidates, tier1_scores)):
            if score < self.drop_threshold:
                tier1_dropped.append((cand, score, "tier1_drop"))
            elif score >= self.keep_threshold:
                tier1_kept.append((cand, score, "tier1_match"))
            else:
                ambiguous_indices.append(idx)

        # --- Tier 2: Laya Deep Arbitration (Only boundary cases) ---
        t2_start = time.perf_counter()
        tier2_results = []

        if ambiguous_indices and self.laya_agent is not None:
            for idx in ambiguous_indices:
                cand = candidates[idx]
                np_score = tier1_scores[idx]
                state = f"Document: {cand}"
                questions = {
                    "relevance": {
                        "type": "noul",
                        "instructions": f"Ce document apporte-t-il une réponse pertinente à : '{query}' ?",
                    }
                }
                try:
                    res = self.laya_agent.system_one(state, questions)
                    laya_prob = float(res["answers"]["relevance"]["noul"])
                    conf = float(res["answers"]["relevance"]["confidence"])

                    # Decision based on Laya's deep evaluation
                    if laya_prob >= 0.50:
                        tier2_results.append((cand, laya_prob, "tier2_arbitrated_kept"))
                    else:
                        tier1_dropped.append((cand, laya_prob, "tier2_arbitrated_dropped"))
                except Exception:
                    # Fallback to NanoPrune's score if Laya forward pass errors
                    if np_score >= 0.50:
                        tier2_results.append((cand, np_score, "tier1_fallback_kept"))
                    else:
                        tier1_dropped.append((cand, np_score, "tier1_fallback_dropped"))
        else:
            # If no Laya, decide boundary cases via 0.50 cutoff on NanoPrune
            for idx in ambiguous_indices:
                cand = candidates[idx]
                np_score = tier1_scores[idx]
                if np_score >= 0.50:
                    tier2_results.append((cand, np_score, "tier1_direct_kept"))
                else:
                    tier1_dropped.append((cand, np_score, "tier1_direct_dropped"))

        t2_elapsed = (time.perf_counter() - t2_start) * 1000.0
        total_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Combine kept results and sort by confidence descending
        retained = [(c, s, tag) for c, s, tag in tier1_kept + tier2_results]
        retained.sort(key=lambda x: x[1], reverse=True)

        return {
            "retained": retained,
            "dropped_count": len(tier1_dropped),
            "stats": {
                "total_candidates": len(candidates),
                "tier1_fast_dropped": len([x for x in tier1_dropped if "tier1" in x[2]]),
                "tier1_fast_kept": len(tier1_kept),
                "tier2_arbitrated": len(ambiguous_indices),
                "tier1_latency_ms": round(t1_elapsed, 2),
                "tier2_latency_ms": round(t2_elapsed, 2),
                "total_latency_ms": round(total_elapsed, 2),
                "efficiency_gain_pct": round((1.0 - (len(ambiguous_indices) / max(1, len(candidates)))) * 100, 1),
            }
        }
