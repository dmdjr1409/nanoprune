"""Semantic relevance with a pretrained sentence-embedding model (ONNX).

``SemanticScorer`` embeds the query and the passages, takes their cosine
similarity and maps it to a probability with Platt scaling fitted on the
development suite. It has the same interface as ``NanoPruner`` (``score``,
``prune``, ``rank``, ``describe``...), so the search engine, the local app, the
cascade and the evaluation can use it unchanged.

A model directory holds an ONNX export of a sentence-embedding transformer and
its Hugging Face ``tokenizer.json`` (optionally ``config.json``), plus the
``nanoprune-dense.json`` written by ``nanoprune download --dense`` or
``nanoprune calibrate``::

    {"name": "multilingual-e5-large-int8", "pooling": "mean",
     "query_prefix": "query: ", "passage_prefix": "passage: ",
     "max_length": 512, "calibration": {"a": 57.0, "b": -46.8, "fitted_on": "dev"}}

Only ``onnxruntime``, ``tokenizers`` and ``numpy`` are needed (no PyTorch).

Passage embeddings can also be kept on disk (``EmbeddingStore``, SQLite in the
user cache), so that a folder analysed once is ready immediately next time.
"""
import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ..core.calibration import fit_platt
from ..errors import HeadUnavailableError, OperationCancelled
from .base import PruneRankMixin

DENSE_CONFIG = "nanoprune-dense.json"

# Models `nanoprune download --dense` can install. The archive is the ONNX export
# that the fastembed project mirrors on Google Cloud Storage; its SHA-256 is pinned.
KNOWN_DENSE_MODELS: Dict[str, Dict[str, Any]] = {
    "multilingual-e5-large": {
        "url": "https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz",
        "sha256": "6de9742c12bc29e37a0ac49521eda668028424c8068df8ca4941861e504a9d40",
        "archive_dir": "fast-multilingual-e5-large",
        "source": "intfloat/multilingual-e5-large",
        "license": "MIT",
        "pooling": "mean",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_length": 512,
    },
}


def _default_settings(name: str) -> Dict[str, Any]:
    lowered = name.lower()
    if "e5" in lowered:
        return {"pooling": "mean", "query_prefix": "query: ", "passage_prefix": "passage: ", "max_length": 512}
    return {"pooling": "mean", "query_prefix": "", "passage_prefix": "", "max_length": 512}


def read_dense_config(model_dir: Path) -> Dict[str, Any]:
    path = Path(model_dir) / DENSE_CONFIG
    config = _default_settings(Path(model_dir).name)
    config["name"] = Path(model_dir).name
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            config.update(json.load(handle))
    return config


def write_dense_config(model_dir: Path, config: Dict[str, Any]) -> Path:
    path = Path(model_dir) / DENSE_CONFIG
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


class EmbeddingStore:
    """Passage embeddings of one model kept on disk (SQLite), keyed by a hash of the text.

    The store is emptied when the model file or its settings change. At most
    ``max_rows`` vectors are kept (the oldest are dropped first).
    """

    def __init__(self, path: Union[str, Path], fingerprint: str, max_rows: int = 200_000):
        import sqlite3  # imported here: some minimal Python builds lack it

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rows = max_rows
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock, self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors (key BLOB PRIMARY KEY, vector BLOB NOT NULL, created REAL)"
            )
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
            if row is None or row[0] != fingerprint:
                self._conn.execute("DELETE FROM vectors")
                self._conn.execute("INSERT OR REPLACE INTO meta VALUES ('fingerprint', ?)", (fingerprint,))

    @classmethod
    def for_model(cls, model_dir: Union[str, Path], config: Dict[str, Any]) -> "EmbeddingStore":
        """The store of a model directory, under ``<cache>/nanoprune/embeddings``."""
        from ..core.weights import cache_dir

        model_dir = Path(model_dir).expanduser().resolve()
        model_file = model_dir / "model.onnx"
        stat = model_file.stat()
        fingerprint = json.dumps([
            stat.st_size, stat.st_mtime_ns, config.get("pooling"), config.get("max_length"),
            config.get("passage_prefix"),
        ])
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(config.get("name") or model_dir.name))
        digest = hashlib.sha256(str(model_dir).encode("utf-8")).hexdigest()[:8]
        return cls(cache_dir() / "embeddings" / f"{name}-{digest}.sqlite3", fingerprint)

    @staticmethod
    def _key(text: str) -> bytes:
        return hashlib.sha256(text.encode("utf-8")).digest()[:16]

    def get_many(self, texts: Sequence[str]) -> Dict[str, np.ndarray]:
        keyed = {self._key(text): text for text in texts}
        keys = list(keyed)
        found: Dict[str, np.ndarray] = {}
        with self._lock:
            for start in range(0, len(keys), 500):
                batch = keys[start:start + 500]
                rows = self._conn.execute(
                    f"SELECT key, vector FROM vectors WHERE key IN ({','.join('?' * len(batch))})", batch
                ).fetchall()
                for key, blob in rows:
                    found[keyed[bytes(key)]] = np.frombuffer(blob, dtype=np.float32).copy()
        return found

    def put_many(self, vectors: Dict[str, np.ndarray]) -> None:
        now = time.time()
        rows = [(self._key(text), np.asarray(v, dtype=np.float32).tobytes(), now) for text, v in vectors.items()]
        with self._lock, self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)", rows)
            excess = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] - self.max_rows
            if excess > 0:
                self._conn.execute(
                    "DELETE FROM vectors WHERE key IN (SELECT key FROM vectors ORDER BY created LIMIT ?)", (excess,)
                )

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _open_store(model_dir: Union[str, Path]) -> Optional[EmbeddingStore]:
    """The disk store of a model, or None when it cannot be used (searching still works)."""
    try:
        import sqlite3
    except ImportError:
        return None
    try:
        return EmbeddingStore.for_model(model_dir, read_dense_config(Path(model_dir).expanduser()))
    except (OSError, sqlite3.Error):
        return None  # read-only or broken cache


def disk_cache_enabled() -> bool:
    """False when $NANOPRUNE_DISK_CACHE is 0/false/no/off."""
    return os.environ.get("NANOPRUNE_DISK_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


class DenseEncoder:
    """Sentence embeddings from an ONNX transformer, L2-normalised."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        num_threads: Optional[int] = None,
        batch_size: int = 16,
    ):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("Semantic search needs onnxruntime and tokenizers: pip install 'nanoprune[onnx]'") from exc

        self.model_dir = Path(model_dir).expanduser()
        model_file = self.model_dir / "model.onnx"
        tokenizer_file = self.model_dir / "tokenizer.json"
        for required in (model_file, tokenizer_file):
            if not required.exists():
                raise FileNotFoundError(f"{required} not found: a dense model directory needs model.onnx and tokenizer.json")
        self.config = config if config is not None else read_dense_config(self.model_dir)
        self.batch_size = batch_size
        self.pooling = self.config.get("pooling", "mean")
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"Unsupported pooling '{self.pooling}' (expected 'mean' or 'cls')")

        pad_id = 0
        hf_config = self.model_dir / "config.json"
        if hf_config.exists():
            with open(hf_config, "r", encoding="utf-8") as handle:
                pad_id = int(json.load(handle).get("pad_token_id") or 0)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_truncation(int(self.config.get("max_length", 512)))
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=self.tokenizer.id_to_token(pad_id) or "<pad>")

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads or int(os.environ.get("NANOPRUNE_THREADS", "0"))
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_file), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]

    def encode(
        self,
        texts: Sequence[str],
        kind: str = "passage",
        progress: Optional[Callable[[int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """Embeddings of ``texts`` (``kind`` = "query" or "passage" selects the prefix).

        ``progress(done, total)`` is called after each batch; ``should_stop()`` is
        polled before each batch and raises ``OperationCancelled`` when true.
        """
        prefix = self.config.get("query_prefix" if kind == "query" else "passage_prefix", "")
        # Batching texts of similar length wastes less computation on padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        result: Optional[np.ndarray] = None
        for start in range(0, len(order), self.batch_size):
            if should_stop is not None and should_stop():
                raise OperationCancelled("Embedding cancelled")
            indices = order[start:start + self.batch_size]
            encodings = self.tokenizer.encode_batch([prefix + texts[i] for i in indices])
            ids = np.array([e.ids for e in encodings], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                pooled = (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if result is None:
                result = np.empty((len(texts), pooled.shape[1]), dtype=np.float32)
            result[indices] = pooled / np.maximum(norms, 1e-12)
            if progress is not None:
                progress(min(start + self.batch_size, len(order)), len(order))
        if result is None:
            return np.zeros((0, 0), dtype=np.float32)
        return result


class SemanticScorer(PruneRankMixin):
    """Calibrated cosine relevance with the NanoPruner interface.

    Passage embeddings are cached, so scoring every indexed passage costs one
    query embedding plus a dot product once the index is warmed up. The search
    engine keeps the embedding matrix of its passages (``embed``) and scores it
    with ``score_embeddings``. Instances can be shared between threads.
    """

    backend = "semantic"
    has_model = True
    full_scan = True
    # With a disk store, new vectors are saved every ``persist_every`` passages.
    persist_every = 64

    def __init__(
        self,
        model_dir: Union[str, Path],
        threshold: float = 0.5,
        num_threads: Optional[int] = None,
        cache_size: int = 20_000,
        encoder: Optional[DenseEncoder] = None,
        store: Optional[EmbeddingStore] = None,
    ):
        """``store`` keeps passage embeddings on disk across runs (see ``EmbeddingStore``)."""
        self.model_dir = Path(model_dir).expanduser()
        self.config = read_dense_config(self.model_dir)
        self.encoder = encoder or DenseEncoder(self.model_dir, self.config, num_threads=num_threads)
        self.store = store
        self.threshold = threshold
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._queries: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()
        calibration = self.config.get("calibration") or {}
        self.a = calibration.get("a")
        self.b = calibration.get("b")

    @classmethod
    def load(
        cls,
        model_dir: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
        disk_cache: bool = False,
    ) -> "SemanticScorer":
        """Load ``model_dir``, $NANOPRUNE_DENSE_MODEL, or the first installed model ("auto").

        With ``disk_cache``, passage embeddings are kept in the user cache unless
        $NANOPRUNE_DISK_CACHE=0.
        """
        if model_dir in (None, "auto"):
            model_dir = os.environ.get("NANOPRUNE_DENSE_MODEL") or None
        if model_dir in (None, "auto"):
            found = installed_dense_models()
            if not found:
                raise FileNotFoundError(
                    "No semantic model installed: run 'nanoprune download --dense multilingual-e5-large' "
                    "or pass --dense /path/to/model"
                )
            model_dir = found[0]
        store = _open_store(model_dir) if disk_cache and disk_cache_enabled() else None
        return cls(model_dir, threshold=threshold, store=store)

    # ------------------------------------------------------------------ embeddings

    def embed(
        self,
        passages: Sequence[str],
        progress: Optional[Callable[[int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """Embedding matrix of ``passages`` (one row each), computing only unknown ones.

        Vectors come from the memory cache, then the disk store, then the model;
        new vectors are saved as they are computed, so a cancelled run resumes
        where it stopped. ``progress(done, total)`` counts distinct passages,
        known ones included; ``should_stop()`` cancels (``OperationCancelled``).
        """
        unique = list(dict.fromkeys(passages))
        known: Dict[str, np.ndarray] = {}
        with self._lock:
            for passage in unique:
                vector = self._cache.get(passage)
                if vector is not None:
                    self._cache.move_to_end(passage)
                    known[passage] = vector
        if self.store is not None:
            from_disk = self.store.get_many([p for p in unique if p not in known])
            known.update(from_disk)
            self._remember(from_disk)
        missing = [p for p in unique if p not in known]
        if progress is not None:
            progress(len(known), len(unique))
        step = self.persist_every if self.store is not None else max(len(missing), 1)
        for start in range(0, len(missing), step):
            part = missing[start:start + step]
            options: Dict[str, Any] = {}
            if progress is not None:
                offset = len(known)
                options["progress"] = lambda done, _total: progress(offset + done, len(unique))
            if should_stop is not None:
                options["should_stop"] = should_stop
            fresh = dict(zip(part, self.encoder.encode(part, kind="passage", **options)))
            known.update(fresh)
            self._remember(fresh)
            if self.store is not None:
                self.store.put_many(fresh)
        if not passages:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([known[p] for p in passages])

    def _remember(self, vectors: Dict[str, np.ndarray]) -> None:
        if not vectors:
            return
        with self._lock:
            self._cache.update(vectors)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def warm(self, passages: Sequence[str]) -> int:
        """Embed and cache passages that are not cached yet; returns how many were embedded."""
        with self._lock:
            missing = sum(1 for p in dict.fromkeys(passages) if p not in self._cache)
        self.embed(passages)
        return missing

    def query_vector(self, query: str) -> np.ndarray:
        with self._lock:
            vector = self._queries.get(query)
            if vector is not None:
                self._queries.move_to_end(query)
                return vector
        vector = self.encoder.encode([query], kind="query")[0]
        with self._lock:
            self._queries[query] = vector
            while len(self._queries) > 256:
                self._queries.popitem(last=False)
        return vector

    def cosine(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        return self.embed(passages) @ self.query_vector(query)

    # ------------------------------------------------------------------ scoring

    @property
    def calibrated(self) -> bool:
        return self.a is not None and self.b is not None

    def _probabilities(self, sims: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            raise RuntimeError(
                f"{self.model_dir} has no calibration yet: run 'nanoprune calibrate --dense {self.model_dir}'"
            )
        return 1.0 / (1.0 + np.exp(-(self.a * sims.astype(np.float64) + self.b)))

    def score(self, query: str, candidates: List[str], batch_size: int = 32) -> List[float]:
        """Probability of relevance for each candidate (calibrated cosine similarity)."""
        if not candidates:
            return []
        return self._probabilities(self.cosine(query, candidates)).tolist()

    def score_embeddings(self, query: str, matrix: np.ndarray) -> np.ndarray:
        """Probability of relevance of passages already embedded with ``embed``."""
        if len(matrix) == 0:
            return np.zeros(0, dtype=np.float64)
        return self._probabilities(matrix @ self.query_vector(query))

    def calibrate(self, suite: str = "dev", l2: float = 1.0, save: bool = True) -> Dict[str, Any]:
        """Fit Platt scaling on an evaluation suite and store it in ``nanoprune-dense.json``."""
        from ..evaluation import load_suite

        cases = load_suite(suite)
        groups: Dict[str, List[int]] = {}
        for i, case in enumerate(cases):
            groups.setdefault(case.query, []).append(i)
        sims = [0.0] * len(cases)
        for query, indices in groups.items():
            for i, value in zip(indices, self.cosine(query, [cases[i].doc for i in indices]).tolist()):
                sims[i] = value
        self.a, self.b = fit_platt(sims, [c.label for c in cases], l2=l2)
        calibration = {"a": round(self.a, 6), "b": round(self.b, 6), "fitted_on": str(suite), "l2": l2}
        self.config["calibration"] = calibration
        if save:
            write_dense_config(self.model_dir, self.config)
        return calibration

    # ------------------------------------------------------------------ NanoPruner-compatible status

    def available_heads(self) -> Dict[str, bool]:
        return {"relevance": True, "choice": False, "score": False}

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model_name": self.config.get("name", self.model_dir.name),
            "model_path": str(self.model_dir),
            "tokenizer": "huggingface",
            "heads": self.available_heads(),
            "calibration": self.config.get("calibration"),
            "license": self.config.get("license"),
            "disk_cache": str(self.store.path) if self.store is not None else None,
        }

    def choice(self, context: str, options: List[str]):
        raise HeadUnavailableError("choice() is not available with a semantic model; load a NanoPrune checkpoint.")

    def score_rubric(self, context: str) -> float:
        raise HeadUnavailableError("score_rubric() is not available with a semantic model; load a NanoPrune checkpoint.")


def dense_models_dir() -> Path:
    from ..core.weights import cache_dir
    return cache_dir() / "dense"


def installed_dense_models() -> List[Path]:
    """Calibrated model directories under the user cache, int8 variants first."""
    root = dense_models_dir()
    if not root.is_dir():
        return []
    found = [p for p in sorted(root.iterdir()) if (p / "model.onnx").exists() and (p / DENSE_CONFIG).exists()]
    return sorted(found, key=lambda p: (not p.name.endswith("-int8"), p.name))
