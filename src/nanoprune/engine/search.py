import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from .indexer import LocalDocumentIndexer, DocumentChunk
from .lexical import BM25Index, tokenize
from .pruner import NanoPruner

_SEGMENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+|\n+")


class LocalSearchEngine:
    """
    Zero-cloud, zero-LLM local search engine.

    Candidate passages are retrieved with BM25 (or, for semantic scorers, every
    passage is considered), then scored by the relevance scorer: a semantic
    model, the NanoPrune model, or its keyword heuristic when no model is loaded.
    """
    def __init__(
        self,
        indexer: Optional[LocalDocumentIndexer] = None,
        pruner: Optional[NanoPruner] = None,
        candidate_pool: int = 30,
        full_scan_limit: int = 256,
    ):
        """
        Args:
            candidate_pool: number of BM25 candidates sent to the scorer.
            full_scan_limit: with a neural model loaded, corpora up to this many
                passages are scored entirely, so that paraphrases sharing no
                keyword with the query can still be found.
        """
        self.indexer = indexer or LocalDocumentIndexer()
        self.pruner = pruner or NanoPruner.load()
        self.candidate_pool = candidate_pool
        self.full_scan_limit = full_scan_limit
        self._bm25: Optional[BM25Index] = None
        self._bm25_signature: Optional[Tuple] = None
        self._lock = threading.RLock()

    def _signature(self) -> Tuple:
        chunks = self.indexer.chunks
        return (
            getattr(self.indexer, "version", 0),
            len(chunks),
            id(chunks[0]) if chunks else None,
            id(chunks[-1]) if chunks else None,
        )

    def _bm25_index(self) -> BM25Index:
        with self._lock:
            signature = self._signature()
            if self._bm25 is None or signature != self._bm25_signature:
                self._bm25 = BM25Index([chunk.text for chunk in self.indexer.chunks])
                self._bm25_signature = signature
            return self._bm25

    def prepare(self) -> None:
        """Build the lexical index and pre-compute passage embeddings (semantic scorers).

        Called after indexing so that the first query does not pay for it.
        """
        self._bm25_index()
        warm = getattr(self.pruner, "warm", None)
        if warm is not None:
            warm([chunk.text for chunk in self.indexer.chunks])

    def _candidates(self, query: str) -> List[Tuple[int, float]]:
        bm25 = self._bm25_index()
        chunks = self.indexer.chunks
        # Semantic scorers score every passage from cached embeddings, so passages
        # that share no keyword with the query are found whatever the corpus size.
        full_scan = getattr(self.pruner, "full_scan", False)
        if full_scan or (self.pruner.has_model and len(chunks) <= self.full_scan_limit):
            lexical = bm25.scores(query)
            return list(enumerate(lexical))
        return bm25.top_k(query, self.candidate_pool)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        chunks = self.indexer.chunks
        response: Dict[str, Any] = {
            "query": query,
            "backend": self.pruner.backend,
            "threshold": threshold,
            "total_chunks_scanned": len(chunks),
            "candidates_evaluated": 0,
            "matches_retained": 0,
            "latency_ms": 0.0,
            "results": [],
        }
        if not chunks or not query.strip():
            return response

        # Step 1: lexical candidate retrieval (BM25)
        candidates = self._candidates(query)

        # Step 2: relevance scoring of the candidates
        texts = [chunks[i].text for i, _ in candidates]
        scores = self.pruner.score(query, texts)

        # Step 3: threshold, rank, and extract the best supporting sentence
        ranked = sorted(
            ((chunks[i], bm25, float(score)) for (i, bm25), score in zip(candidates, scores) if score >= threshold),
            key=lambda item: (-item[2], -item[1]),
        )[:top_k]
        response["results"] = [self._result(query, chunk, bm25, score) for chunk, bm25, score in ranked]
        response["candidates_evaluated"] = len(candidates)
        response["matches_retained"] = len(response["results"])
        response["latency_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
        return response

    def _result(self, query: str, chunk: DocumentChunk, bm25: float, score: float) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "file_name": chunk.file_name,
            "file_path": chunk.file_path,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "score": round(score, 4),
            "score_pct": f"{round(score * 100, 1)}%",
            # Backwards-compatible aliases of "score"/"score_pct".
            "confidence": round(score, 4),
            "confidence_pct": f"{round(score * 100, 1)}%",
            "bm25": round(bm25, 3),
            "highlight": self._extract_best_sentence(query, chunk.body),
            "text": chunk.body,
        }

    def _extract_best_sentence(self, query: str, text: str) -> str:
        """Picks the sentence (or line) of the passage that covers the most query terms."""
        segments = [s.strip() for s in _SEGMENT_SPLIT_RE.split(text) if s and s.strip()]
        if not segments:
            return text.strip()
        q_terms = set(tokenize(query))
        best, best_overlap = segments[0], -1
        for segment in segments:
            overlap = len(q_terms & set(tokenize(segment)))
            if overlap > best_overlap:
                best, best_overlap = segment, overlap
        return best
