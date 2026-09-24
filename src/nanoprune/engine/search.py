import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .indexer import LocalDocumentIndexer, DocumentChunk
from .lexical import STOPWORDS, BM25Index, fold_accents, light_stem, tokenize
from .pruner import NanoPruner

_SEGMENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+|\n+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


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
        near_misses: int = 3,
    ):
        """
        Args:
            candidate_pool: number of BM25 candidates sent to the scorer.
            full_scan_limit: with a neural model loaded, corpora up to this many
                passages are scored entirely, so that paraphrases sharing no
                keyword with the query can still be found.
            near_misses: how many of the best passages just below the threshold
                are returned in ``near_misses``.
        """
        self.indexer = indexer or LocalDocumentIndexer()
        self.pruner = pruner or NanoPruner.load()
        self.candidate_pool = candidate_pool
        self.full_scan_limit = full_scan_limit
        self.near_misses = near_misses
        self._bm25: Optional[BM25Index] = None
        self._bm25_signature: Optional[Tuple] = None
        self._matrix: Optional[np.ndarray] = None
        self._matrix_signature: Optional[Tuple] = None
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

    def _embeddings(
        self,
        progress: Optional[Callable[[int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Optional[np.ndarray]:
        """Embedding matrix of every passage, for scorers that support it (semantic models)."""
        if not (hasattr(self.pruner, "embed") and hasattr(self.pruner, "score_embeddings")):
            return None
        with self._lock:
            signature = self._signature()
            if self._matrix is None or signature != self._matrix_signature:
                texts = [chunk.text for chunk in self.indexer.chunks]
                self._matrix = self.pruner.embed(texts, progress=progress, should_stop=should_stop)
                self._matrix_signature = signature
            return self._matrix

    def prepare(
        self,
        progress: Optional[Callable[[int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Build the lexical index and embed every passage (semantic scorers).

        Called after indexing so that the first query does not pay for it.
        ``progress(done, total)`` reports the passages embedded so far and
        ``should_stop()`` can cancel (``OperationCancelled``).
        """
        self._bm25_index()
        if self._embeddings(progress=progress, should_stop=should_stop) is None:
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

    def _score(self, query: str, candidates: List[Tuple[int, float]]) -> List[float]:
        chunks = self.indexer.chunks
        matrix = self._embeddings()
        if matrix is not None and len(matrix) == len(chunks):
            scores = self.pruner.score_embeddings(query, matrix)
            return [float(scores[i]) for i, _ in candidates]
        return [float(s) for s in self.pruner.score(query, [chunks[i].text for i, _ in candidates])]

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """Passages scoring at least ``threshold``, best first.

        Passages that overlap a better result (consecutive passages of the same
        paragraph) are left out. The response also counts every match
        (``matches_total``) and lists the best passages just below the threshold
        (``near_misses``, scoring at least half of it).
        """
        start_time = time.perf_counter()
        chunks = self.indexer.chunks
        response: Dict[str, Any] = {
            "query": query,
            "backend": self.pruner.backend,
            "threshold": threshold,
            "total_chunks_scanned": len(chunks),
            "candidates_evaluated": 0,
            "matches_retained": 0,
            "matches_total": 0,
            "more_available": False,
            "latency_ms": 0.0,
            "results": [],
            "near_misses": [],
        }
        if not chunks or not query.strip():
            return response

        # Step 1: candidate retrieval (BM25, or every passage for semantic scorers)
        candidates = self._candidates(query)

        # Step 2: relevance scoring of the candidates
        scores = self._score(query, candidates)

        # Step 3: threshold, rank, drop overlapping passages, extract the best sentence
        floor = threshold * 0.5
        ranked = sorted(
            ((i, bm25, score) for (i, bm25), score in zip(candidates, scores) if score >= floor),
            key=lambda item: (-item[2], -item[1], item[0]),
        )
        matches: List[Tuple[int, float, float]] = []
        near: List[Tuple[int, float, float]] = []
        shown: Set[int] = set()
        for i, bm25, score in ranked:
            if score < threshold and len(near) >= self.near_misses:
                break
            if self._overlaps(i, shown):
                continue
            if score >= threshold:
                matches.append((i, bm25, score))
            else:
                near.append((i, bm25, score))
            shown.add(i)

        # Query terms weighted by rarity; terms absent from every passage weigh nothing.
        index = self._bm25_index()
        weights = {t: (index.idf(t) if t in index.doc_freq else 0.0) for t in tokenize(query)}
        response["results"] = [self._result(weights, chunks[i], bm25, score) for i, bm25, score in matches[:top_k]]
        if len(matches) < top_k:
            response["near_misses"] = [self._result(weights, chunks[i], bm25, score) for i, bm25, score in near]
        response["candidates_evaluated"] = len(candidates)
        response["matches_retained"] = len(response["results"])
        response["matches_total"] = len(matches)
        response["more_available"] = len(matches) > top_k
        response["latency_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
        return response

    def _overlaps(self, index: int, shown: Set[int]) -> bool:
        """True when a better passage already shown repeats most of this one.

        Consecutive passages cut from one long paragraph share their boundary
        sentences (``shared_prev``); a passage is left out when more than half
        of it is such a repetition of a neighbour that ranks higher.
        """
        chunks = self.indexer.chunks
        chunk = chunks[index]
        length = max(len(chunk.body), 1)
        if index - 1 in shown and chunks[index - 1].file_path == chunk.file_path:
            if 2 * chunk.metadata.get("shared_prev", 0) > length:
                return True
        if index + 1 in shown and chunks[index + 1].file_path == chunk.file_path:
            if 2 * chunks[index + 1].metadata.get("shared_prev", 0) > length:
                return True
        return False

    def _result(self, weights: Dict[str, float], chunk: DocumentChunk, bm25: float, score: float) -> Dict[str, Any]:
        q_terms = set(weights)
        return {
            "chunk_id": chunk.chunk_id,
            "file_name": chunk.file_name,
            "file_path": chunk.file_path,
            # Path relative to the indexed folder (the file name for single files).
            "rel_path": chunk.chunk_id.rsplit("#", 1)[0],
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "score": round(score, 4),
            "score_pct": f"{round(score * 100, 1)}%",
            # Backwards-compatible aliases of "score"/"score_pct".
            "confidence": round(score, 4),
            "confidence_pct": f"{round(score * 100, 1)}%",
            "bm25": round(bm25, 3),
            "highlight": self._best_sentence(q_terms, chunk.body, weights.get),
            "text": chunk.body,
            "highlight_terms": self._matching_words(q_terms, chunk.body),
        }

    @staticmethod
    def _matching_words(q_terms: Set[str], text: str) -> List[str]:
        """Words of ``text`` (lowercased, as written) that match a query term."""
        words = set()
        for word in _WORD_RE.findall(text):
            folded = fold_accents(word)
            if folded not in STOPWORDS and light_stem(folded) in q_terms:
                words.add(word.lower())
        return sorted(words)

    @staticmethod
    def _best_sentence(
        q_terms: Set[str],
        text: str,
        weight: Optional[Callable[[str], Optional[float]]] = None,
        min_coverage: float = 0.3,
    ) -> str:
        """The sentence (or line) of the passage that best covers the query terms.

        Terms count ``weight(term)`` (default 1; the search engine uses their IDF,
        0 for terms found in no passage). When no sentence covers at least
        ``min_coverage`` of the query's weight, e.g. for a semantic match that
        shares no keyword with the query, the whole passage is returned rather
        than a sentence matching only a common word.
        """
        def w(term: str) -> float:
            value = weight(term) if weight is not None else 1.0
            return value or 0.0

        total = sum(w(term) for term in q_terms)
        best, best_weight = text.strip(), 0.0
        for segment in _SEGMENT_SPLIT_RE.split(text):
            segment = (segment or "").strip()
            if not segment:
                continue
            covered = sum(w(term) for term in q_terms & set(tokenize(segment)))
            if covered > best_weight:
                best, best_weight = segment, covered
        if total <= 0 or best_weight < min_coverage * total:
            return text.strip()
        return best

    def _extract_best_sentence(self, query: str, text: str) -> str:
        """Backwards-compatible wrapper of ``_best_sentence``."""
        index = self._bm25_index()
        terms = set(tokenize(query))
        return self._best_sentence(terms, text, lambda t: index.idf(t) if t in index.doc_freq else 0.0)
