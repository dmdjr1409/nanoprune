from typing import Any, Dict, List, Optional, Tuple


class PruneRankMixin:
    """``prune`` and ``rank`` on top of a ``score(query, candidates)`` method and a ``threshold``."""

    threshold: float

    def score(self, query: str, candidates: List[str]) -> List[float]:  # pragma: no cover - interface
        raise NotImplementedError

    def score_pair(self, query: str, context: str) -> float:
        """Score a single (query, context) pair."""
        return self.score(query, [context])[0]

    def prune(
        self,
        query: str,
        candidates: List[str],
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Drops candidates whose relevance score is below the threshold.
        Returns sorted list of (candidate, score).
        """
        th = self.threshold if threshold is None else threshold
        scores = self.score(query, candidates)
        retained = [(cand, score) for cand, score in zip(candidates, scores) if score >= th]
        retained.sort(key=lambda x: x[1], reverse=True)
        return retained

    def rank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        key: str = "text",
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ranks dictionaries by relevance score of items[key].
        """
        texts = [item.get(key, "") for item in items]
        scores = self.score(query, texts)
        th = self.threshold if threshold is None else threshold

        ranked = []
        for item, score in zip(items, scores):
            if score >= th:
                res = dict(item)
                res["nanoprune_score"] = score
                ranked.append(res)

        ranked.sort(key=lambda x: x["nanoprune_score"], reverse=True)
        return ranked
