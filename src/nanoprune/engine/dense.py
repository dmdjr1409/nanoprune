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
"""
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..core.calibration import fit_platt
from ..errors import HeadUnavailableError
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

    def encode(self, texts: Sequence[str], kind: str = "passage") -> np.ndarray:
        """Embeddings of ``texts`` (``kind`` = "query" or "passage" selects the prefix)."""
        prefix = self.config.get("query_prefix" if kind == "query" else "passage_prefix", "")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[start:start + self.batch_size]]
            encodings = self.tokenizer.encode_batch(batch)
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
            vectors.append((pooled / np.maximum(norms, 1e-12)).astype(np.float32))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(vectors)


class SemanticScorer(PruneRankMixin):
    """Calibrated cosine relevance with the NanoPruner interface.

    Passage embeddings are cached, so scoring every indexed passage costs one
    query embedding plus a dot product once the index is warmed up.
    """

    backend = "semantic"
    has_model = True
    full_scan = True

    def __init__(
        self,
        model_dir: Union[str, Path],
        threshold: float = 0.5,
        num_threads: Optional[int] = None,
        cache_size: int = 200_000,
        encoder: Optional[DenseEncoder] = None,
    ):
        self.model_dir = Path(model_dir).expanduser()
        self.config = read_dense_config(self.model_dir)
        self.encoder = encoder or DenseEncoder(self.model_dir, self.config, num_threads=num_threads)
        self.threshold = threshold
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        calibration = self.config.get("calibration") or {}
        self.a = calibration.get("a")
        self.b = calibration.get("b")

    @classmethod
    def load(cls, model_dir: Optional[Union[str, Path]] = None, threshold: float = 0.5) -> "SemanticScorer":
        """Load ``model_dir``, $NANOPRUNE_DENSE_MODEL, or the first installed model ("auto")."""
        if model_dir in (None, "auto"):
            model_dir = os.environ.get("NANOPRUNE_DENSE_MODEL") or None
        if model_dir in (None, "auto"):
            found = installed_dense_models()
            if not found:
                raise FileNotFoundError(
                    "No semantic model installed: run 'nanoprune download --dense multilingual-e5-large --int8' "
                    "or pass --dense /path/to/model"
                )
            model_dir = found[0]
        return cls(model_dir, threshold=threshold)

    # ------------------------------------------------------------------ embeddings

    def _vectors(self, passages: Sequence[str]) -> List[np.ndarray]:
        missing = [p for p in dict.fromkeys(passages) if p not in self._cache]
        fresh = dict(zip(missing, self.encoder.encode(missing, kind="passage"))) if missing else {}
        rows = []
        for passage in passages:
            if passage in fresh:
                rows.append(fresh[passage])
            else:
                self._cache.move_to_end(passage)
                rows.append(self._cache[passage])
        for text, vector in fresh.items():
            self._cache[text] = vector
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return rows

    def warm(self, passages: Sequence[str]) -> int:
        """Embed and cache passages that are not cached yet; returns how many were embedded."""
        missing = sum(1 for p in dict.fromkeys(passages) if p not in self._cache)
        self._vectors(passages)
        return missing

    def cosine(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        matrix = np.vstack(self._vectors(passages))
        query_vector = self.encoder.encode([query], kind="query")[0]
        return matrix @ query_vector

    # ------------------------------------------------------------------ scoring

    @property
    def calibrated(self) -> bool:
        return self.a is not None and self.b is not None

    def score(self, query: str, candidates: List[str], batch_size: int = 32) -> List[float]:
        """Probability of relevance for each candidate (calibrated cosine similarity)."""
        if not candidates:
            return []
        sims = self.cosine(query, candidates).astype(np.float64)
        if not self.calibrated:
            raise RuntimeError(
                f"{self.model_dir} has no calibration yet: run 'nanoprune calibrate --dense {self.model_dir}'"
            )
        return (1.0 / (1.0 + np.exp(-(self.a * sims + self.b)))).tolist()

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
