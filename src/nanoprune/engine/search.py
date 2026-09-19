import time
import re
from typing import List, Dict, Any, Optional
from .indexer import LocalDocumentIndexer, DocumentChunk
from .pruner import NanoPruner

class LocalSearchEngine:
    """
    Zero-cloud, zero-LLM confidential local search engine.
    Finds exact factual proof across local documents with calibrated certainty.
    """
    def __init__(
        self,
        indexer: Optional[LocalDocumentIndexer] = None,
        pruner: Optional[NanoPruner] = None,
    ):
        self.indexer = indexer or LocalDocumentIndexer()
        self.pruner = pruner or NanoPruner.load()

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if not self.indexer.chunks:
            return {
                "query": query,
                "total_chunks": 0,
                "latency_ms": 0.0,
                "results": [],
            }

        # Step 1: Rapid lexical candidate filtering (top 30 chunks)
        candidates = self._pre_filter_candidates(query, limit=30)

        # Step 2: System One Calibrated Scoring via NanoPrune
        texts = [c.text for c in candidates]
        scores = self.pruner.score(query, texts)

        # Step 3: Filter by threshold & extract highlights
        ranked_results = []
        for chunk, score in zip(candidates, scores):
            if score >= threshold:
                highlight = self._extract_best_sentence(query, chunk.text)
                ranked_results.append({
                    "chunk_id": chunk.chunk_id,
                    "file_name": chunk.file_name,
                    "file_path": chunk.file_path,
                    "confidence": round(float(score), 4),
                    "confidence_pct": f"{round(float(score) * 100, 1)}%",
                    "highlight": highlight,
                    "text": chunk.text,
                })

        ranked_results.sort(key=lambda x: x["confidence"], reverse=True)
        top_results = ranked_results[:top_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": query,
            "total_chunks_scanned": len(self.indexer.chunks),
            "candidates_evaluated": len(candidates),
            "matches_retained": len(top_results),
            "latency_ms": round(elapsed_ms, 2),
            "results": top_results,
        }

    def _pre_filter_candidates(self, query: str, limit: int = 30) -> List[DocumentChunk]:
        query_words = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 2)
        if not query_words:
            return self.indexer.chunks[:limit]

        scored = []
        for chunk in self.indexer.chunks:
            chunk_lower = chunk.text.lower()
            overlap = sum(1 for w in query_words if w in chunk_lower)
            if overlap > 0:
                scored.append((chunk, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:limit]] if scored else self.indexer.chunks[:limit]

    def _extract_best_sentence(self, query: str, text: str) -> str:
        """Picks the single most informative sentence in the chunk matching the query."""
        sentences = re.split(r"(?<=[.?!])\s+", text)
        q_words = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 2)

        best_sentence = sentences[0] if sentences else text
        max_overlap = -1

        for s in sentences:
            s_lower = s.lower()
            overlap = sum(1 for w in q_words if w in s_lower)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = s.strip()

        return best_sentence
