import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import numpy as np

from ..core.tokenizer import NanoTokenizer
from ..core.weights import (
    LEGACY_MODELS,
    ModelSpec,
    describe_model,
    discover_candidates,
    find_default_tokenizer,
    onnx_initializer_info,
)
from ..errors import HeadUnavailableError, ModelNotFoundError, NanoPruneWarning, TokenizerMismatchError

# Categories the v0.3/v0.4 choice heads were trained on, in output order.
DEFAULT_CHOICE_LABELS = ["human_rights", "business_tax", "public_admin", "civil_family"]
CHAR_VOCAB_SIZE = NanoTokenizer.char_level().vocab_size
CHAR_SPARE_ROWS = 200
LEGACY_MODEL_NAMES = {stem for stem, _kind in LEGACY_MODELS}

NO_MODEL_MESSAGE = (
    "No NanoPrune model weights found: scores come from the keyword heuristic, not from the "
    "neural model. Run 'nanoprune download' or set NANOPRUNE_MODEL / NANOPRUNE_HOME "
    "(see README, 'Getting the weights')."
)


class NanoPruner:
    """
    NanoPruner: high-level System One inference engine.
    Scores, ranks, and prunes RAG context chunks.

    Backends: ``onnx`` (onnxruntime), ``torch`` (PyTorch checkpoint) or
    ``heuristic`` (keyword matching, used when no weights are available).
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        tokenizer: Optional[NanoTokenizer] = None,
        threshold: float = 0.70,
        tokenizer_path: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = None,
    ):
        self.threshold = threshold
        self.session = None
        self.torch_model = None
        self.spec: Optional[ModelSpec] = None
        self.max_length = 256
        self.choice_labels = list(DEFAULT_CHOICE_LABELS)
        self._onnx_outputs: List[str] = []
        self._onnx_fixed_len: Optional[int] = None
        self._rubric_warned = False
        self.tokenizer = tokenizer
        self._tokenizer_explicit = tokenizer is not None

        if model_path is not None:
            spec = describe_model(Path(model_path), Path(tokenizer_path) if tokenizer_path else None)
            self._init_backend(spec, num_threads)
        elif self.tokenizer is None:
            # The keyword heuristic only needs text normalisation.
            self.tokenizer = NanoTokenizer.char_level()

    # ------------------------------------------------------------------ loading

    def _init_backend(self, spec: ModelSpec, num_threads: Optional[int] = None) -> None:
        self.spec = spec
        manifest = spec.manifest or {}
        arch = manifest.get("architecture", {})
        self.max_length = int(arch.get("max_seq_len", 256))
        self.choice_labels = list(manifest.get("choice_labels", DEFAULT_CHOICE_LABELS))

        if spec.backend == "onnx":
            self._init_onnx(spec, num_threads)
        else:
            self._init_torch(spec, arch)

    def _resolve_tokenizer(self, spec: ModelSpec, model_vocab: Optional[int]) -> None:
        """Pick the tokenizer the model was trained with and check it matches the weights."""
        if spec.tokenizer_kind is None:
            if model_vocab is None:
                warnings.warn(
                    f"Cannot tell which tokenizer {spec.path.name} expects (no manifest, unknown file name, "
                    "'onnx' package not installed); assuming the char-level vocabulary.",
                    NanoPruneWarning,
                    stacklevel=4,
                )
                spec.tokenizer_kind = "char"
            else:
                spec.tokenizer_kind = "char" if model_vocab <= CHAR_VOCAB_SIZE + CHAR_SPARE_ROWS else "wordpiece"
                if spec.tokenizer_kind == "wordpiece" and spec.tokenizer_path is None:
                    spec.tokenizer_path = find_default_tokenizer(spec.path.parent)
        if not self._tokenizer_explicit:
            self.tokenizer = self._tokenizer_for(spec)
        self._check_vocab(model_vocab)

    @staticmethod
    def _tokenizer_for(spec: ModelSpec) -> NanoTokenizer:
        if spec.tokenizer_kind == "wordpiece":
            if spec.tokenizer_path is None or not spec.tokenizer_path.exists():
                raise FileNotFoundError(
                    f"{spec.path.name} was trained with the WordPiece tokenizer "
                    "'tokenizer_legal.json', which was not found next to the weights, in a data/ "
                    "directory, or under NANOPRUNE_HOME. Pass tokenizer_path=... or copy the file there."
                )
            tokenizer = NanoTokenizer.from_file(spec.tokenizer_path)
            expected = (spec.manifest or {}).get("tokenizer", {}).get("sha256")
            if expected and tokenizer.sha256 != expected:
                raise TokenizerMismatchError(
                    f"{spec.tokenizer_path} is not the tokenizer {spec.name} was trained with (sha256 differs)."
                )
            return tokenizer
        return NanoTokenizer.char_level()

    def _check_vocab(self, model_vocab: Optional[int]) -> None:
        if model_vocab is None:
            return
        tok_vocab = self.tokenizer.vocab_size
        if self.tokenizer.kind == "wordpiece":
            ok = tok_vocab == model_vocab
        else:
            # v0.1/v0.2 char-level checkpoints were trained with 200 spare embedding rows.
            ok = tok_vocab <= model_vocab <= tok_vocab + CHAR_SPARE_ROWS
        if not ok:
            raise TokenizerMismatchError(
                f"The model expects a vocabulary of {model_vocab} tokens but the {self.tokenizer.kind} "
                f"tokenizer has {tok_vocab}: scores would be meaningless. Use the tokenizer the model was trained with."
            )

    def _init_onnx(self, spec: ModelSpec, num_threads: Optional[int]) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required to run .onnx models. Install via: pip install 'nanoprune[onnx]'"
            )
        info = onnx_initializer_info(spec.path)
        missing = [name for name in info.get("external_files", []) if not (spec.path.parent / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"{spec.path.name} stores its weights in {', '.join(missing)}, which is missing. "
                "Keep the .onnx and .onnx.data files together."
            )
        self._resolve_tokenizer(spec, info.get("vocab_size"))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads or int(os.environ.get("NANOPRUNE_THREADS", "0"))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(str(spec.path), sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as exc:
            hint = ""
            if not (spec.path.parent / (spec.path.name + ".data")).exists():
                hint = f" If the graph uses external weights, '{spec.path.name}.data' must sit next to it."
            raise RuntimeError(f"Failed to load ONNX model {spec.path}: {exc}.{hint}") from exc
        self._onnx_outputs = [o.name for o in self.session.get_outputs()]
        seq_dim = self.session.get_inputs()[0].shape[1]
        if isinstance(seq_dim, int):
            self._onnx_fixed_len = seq_dim
        else:
            # Graphs traced by the TorchScript exporter often keep the sequence length
            # they were traced with, even when the axis is declared dynamic.
            self._onnx_fixed_len = None if self._onnx_accepts_short_inputs() else self.max_length

    def _onnx_accepts_short_inputs(self) -> bool:
        import onnxruntime as ort
        run_options = ort.RunOptions()
        run_options.log_severity_level = 4
        ids = np.array([[self.tokenizer.cls_id, self.tokenizer.sep_id, self.tokenizer.sep_id]], dtype=np.int64)
        try:
            self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)}, run_options)
            return True
        except Exception:
            return False

    def _init_torch(self, spec: ModelSpec, arch: Dict[str, Any]) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to run .pt checkpoints. Install via: pip install 'nanoprune[train]'")
        from ..core.model import NanoPruneModel

        state = torch.load(str(spec.path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        emb = state.get("token_embeddings.weight")
        if emb is None:
            raise RuntimeError(f"{spec.path} is not a NanoPrune checkpoint (no token_embeddings.weight)")
        self._resolve_tokenizer(spec, int(emb.shape[0]))

        if spec.manifest is None and spec.path.stem not in LEGACY_MODEL_NAMES:
            warnings.warn(
                f"{spec.path.name} has no manifest: its number of attention heads cannot be read from the "
                "weights and is assumed to follow the released models (8 heads for d_model >= 256, else 4).",
                NanoPruneWarning,
                stacklevel=4,
            )
        model = NanoPruneModel.from_state_dict(state, n_heads=arch.get("n_heads"))
        model.eval()
        self.torch_model = model
        self.max_length = model.max_seq_len

    @classmethod
    def load(
        cls,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.70,
        strict: bool = False,
        tokenizer_path: Optional[Union[str, Path]] = None,
    ) -> "NanoPruner":
        """Load the best available model.

        Without ``model_path`` the weights are discovered (see
        ``nanoprune.core.weights``). When none is found, ``strict=True`` raises
        :class:`ModelNotFoundError`; otherwise a :class:`NanoPruneWarning` is
        emitted and the keyword heuristic is used.
        """
        if model_path is not None:
            return cls(model_path=model_path, threshold=threshold, tokenizer_path=tokenizer_path)

        specs, notes = discover_candidates()
        for spec in specs:
            try:
                return cls(model_path=spec.path, threshold=threshold, tokenizer_path=tokenizer_path)
            except (OSError, ImportError, RuntimeError, ValueError) as exc:
                # e.g. an ONNX graph whose .onnx.data file is missing: try the next file.
                notes.append(f"{spec.path} could not be loaded: {exc}")
        details = (" " + " ".join(notes)) if notes else ""
        if strict:
            raise ModelNotFoundError(NO_MODEL_MESSAGE + details)
        warnings.warn(NO_MODEL_MESSAGE + details, NanoPruneWarning, stacklevel=2)
        return cls(model_path=None, threshold=threshold)

    @classmethod
    def from_torch_model(cls, model, tokenizer: NanoTokenizer, threshold: float = 0.70) -> "NanoPruner":
        """Wrap an in-memory NanoPruneModel (e.g. right after training)."""
        pruner = cls(tokenizer=tokenizer, threshold=threshold)
        pruner.torch_model = model.eval()
        pruner.max_length = model.max_seq_len
        return pruner

    # ------------------------------------------------------------------ status

    @property
    def backend(self) -> str:
        if self.session is not None:
            return "onnx"
        if self.torch_model is not None:
            return "torch"
        return "heuristic"

    @property
    def has_model(self) -> bool:
        return self.backend != "heuristic"

    def available_heads(self) -> Dict[str, bool]:
        if self.torch_model is not None:
            heads = dict(getattr(self.torch_model, "loaded_heads", {"relevance": True, "choice": True, "score": True}))
        elif self.session is not None:
            heads = {
                "relevance": True,
                "choice": "choice_probs" in self._onnx_outputs,
                "score": "score" in self._onnx_outputs,
            }
        else:
            return {"relevance": True, "choice": False, "score": False}
        # A manifest can declare heads that exist in the weights but were never trained.
        trained = ((self.spec.manifest or {}) if self.spec else {}).get("heads", {})
        return {name: present and bool(trained.get(name, True)) for name, present in heads.items()}

    def describe(self) -> Dict[str, Any]:
        """Human-readable summary of what is actually loaded."""
        info: Dict[str, Any] = {
            "backend": self.backend,
            "model_name": self.spec.name if self.spec else None,
            "model_path": str(self.spec.path) if self.spec else None,
            "tokenizer": self.tokenizer.kind if self.has_model else None,
            "heads": self.available_heads(),
        }
        if self.torch_model is not None:
            info["parameters"] = self.torch_model.count_parameters()
        if not self.has_model:
            info["warning"] = NO_MODEL_MESSAGE
        return info

    # ------------------------------------------------------------------ scoring

    def score_pair(self, query: str, context: str) -> float:
        """Score a single (query, context) pair."""
        return self.score(query, [context])[0]

    def score(self, query: str, candidates: List[str], batch_size: int = 32) -> List[float]:
        """
        Relevance scores in [0.0, 1.0] for each candidate (same order as ``candidates``).
        """
        if not candidates:
            return []

        if self.session is not None:
            return self._score_batched(query, candidates, batch_size, self._run_onnx)
        if self.torch_model is not None:
            return self._score_batched(query, candidates, batch_size, self._run_torch)

        # Keyword heuristic when no model weights are loaded.
        return self._score_keyword_heuristic(query, candidates)

    def _score_batched(self, query: str, candidates: List[str], batch_size: int, run) -> List[float]:
        # Group candidates of similar length so that each batch needs little padding.
        order = sorted(range(len(candidates)), key=lambda i: len(candidates[i]))
        scores = [0.0] * len(candidates)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            ids, masks = self.tokenizer.encode_batch(
                query, [candidates[i] for i in idx], max_length=self.max_length, pad_to=self._onnx_fixed_len
            )
            for i, prob in zip(idx, run(ids, masks)):
                scores[i] = float(prob)
        return scores

    def _run_onnx(self, ids: List[List[int]], masks: List[List[int]]) -> List[float]:
        feeds = {
            "input_ids": np.array(ids, dtype=np.int64),
            "attention_mask": np.array(masks, dtype=np.int64),
        }
        try:
            outputs = self.session.run(None, feeds)
        except Exception:
            if self._onnx_fixed_len is not None:
                raise
            # Graph exported with a static sequence length: pad to the full length from now on.
            self._onnx_fixed_len = self.max_length
            pad = self.max_length - len(ids[0])
            feeds = {
                "input_ids": np.array([row + [self.tokenizer.pad_id] * pad for row in ids], dtype=np.int64),
                "attention_mask": np.array([row + [0] * pad for row in masks], dtype=np.int64),
            }
            outputs = self.session.run(None, feeds)
        named = dict(zip(self._onnx_outputs, outputs))
        if "probabilities" in named:
            probs = named["probabilities"]
        elif "logits" in named:
            probs = 1.0 / (1.0 + np.exp(-named["logits"]))
        else:
            probs = outputs[1] if len(outputs) > 1 else outputs[0]
        return np.asarray(probs, dtype=np.float64).reshape(len(ids), -1)[:, 0].tolist()

    def _run_torch(self, ids: List[List[int]], masks: List[List[int]]) -> List[float]:
        import torch
        with torch.no_grad():
            _, probs = self.torch_model(torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long))
        return probs.squeeze(-1).tolist()

    # ------------------------------------------------------------------ other primitives

    def _head_outputs(self, context: str, query: str) -> Dict[str, np.ndarray]:
        heads = self.available_heads()
        if self.torch_model is not None:
            import torch
            ids, masks = self.tokenizer.encode_batch(query, [context], max_length=self.max_length)
            with torch.no_grad():
                res = self.torch_model.forward_all(torch.tensor(ids), torch.tensor(masks))
            out = {"choice_probs": res["choice_probs"][0].cpu().numpy(), "score": res["score"][0].cpu().numpy()}
        elif self.session is not None and (heads["choice"] or heads["score"]):
            ids, masks = self.tokenizer.encode_batch(query, [context], max_length=self.max_length, pad_to=self._onnx_fixed_len)
            outputs = self.session.run(None, {
                "input_ids": np.array(ids, dtype=np.int64),
                "attention_mask": np.array(masks, dtype=np.int64),
            })
            named = dict(zip(self._onnx_outputs, outputs))
            out = {k: np.asarray(v)[0] for k, v in named.items() if k in ("choice_probs", "score")}
        else:
            out = {}
        return out

    def choice(self, context: str, options: List[str]) -> Tuple[int, str, float]:
        """
        Primitive 2: pick one of the categories the choice head was trained on.

        The head only knows its training categories (``self.choice_labels``);
        ``options`` are mapped onto them by position, so arbitrary labels are
        not zero-shot classification. Raises :class:`HeadUnavailableError` when
        the loaded backend has no choice head.
        """
        if not options:
            raise ValueError("options list cannot be empty")
        if not self.available_heads().get("choice"):
            raise HeadUnavailableError(
                "choice() needs a trained choice head: load the PyTorch checkpoint (.pt) or an ONNX file "
                f"exported with all heads. Current backend: {self.backend}."
            )
        probs = np.asarray(self._head_outputs(context, "Category selection")["choice_probs"], dtype=np.float64)
        if len(options) > len(probs):
            raise ValueError(f"The choice head was trained on {len(probs)} categories; got {len(options)} options.")
        if list(options) != self.choice_labels[:len(options)]:
            warnings.warn(
                f"choice() options are mapped by position onto the trained categories {self.choice_labels}.",
                NanoPruneWarning,
                stacklevel=2,
            )
        sub_probs = probs[:len(options)] / max(1e-6, probs[:len(options)].sum())
        best_idx = int(sub_probs.argmax())
        return best_idx, options[best_idx], float(sub_probs[best_idx])

    def score_rubric(self, context: str) -> float:
        """
        Primitive 3 (experimental): rate text on a continuous [0.0, 4.0] scale.

        The v0.4 training pipeline does not train this head; its output is not
        validated. Raises :class:`HeadUnavailableError` without a score head.
        """
        if not self.available_heads().get("score"):
            raise HeadUnavailableError(
                "score_rubric() needs a trained score head: load the PyTorch checkpoint (.pt) or an ONNX file "
                f"exported with all heads. Current backend: {self.backend}."
            )
        if not self._rubric_warned:
            warnings.warn(
                "score_rubric() is experimental: the v0.4 pipeline does not train the score head.",
                NanoPruneWarning,
                stacklevel=2,
            )
            self._rubric_warned = True
        return float(np.asarray(self._head_outputs(context, "Score evaluation")["score"]).reshape(-1)[0])

    # ------------------------------------------------------------------ heuristic

    def _score_keyword_heuristic(self, query: str, candidates: List[str]) -> List[float]:
        """
        Keyword, stem and negation overlap scoring used when no model is loaded.
        The values are fixed score buckets, not calibrated probabilities.
        """
        q_norm = self.tokenizer.normalize(query)
        q_words = re_words(q_norm)
        if not q_words:
            return [0.0] * len(candidates)

        def stem(w: str) -> str:
            # Common French & English root extractor
            w_clean = w.rstrip("s").rstrip("e")
            if len(w_clean) >= 6 and w_clean.endswith("iqu"):
                w_clean = w_clean[:-3]
            elif len(w_clean) >= 6 and w_clean.endswith("abl"):
                w_clean = w_clean[:-3]
            return w_clean[:5] if len(w_clean) >= 5 else w_clean

        q_stems = [stem(w) for w in q_words]
        negations = {"aucun", "aucune", "pas", "sans", "jamais", "non", "no", "not", "none", "never"}
        has_q_negation = any(w in negations for w in q_words)

        scores = []
        for c in candidates:
            c_norm = self.tokenizer.normalize(c)
            c_words = re_words(c_norm)
            c_stems = set(stem(w) for w in c_words)
            has_c_negation = any(w in negations for w in c_words)

            matches = sum(1 for qs in q_stems if qs in c_stems or any(qs in cs for cs in c_stems))
            match_ratio = matches / len(q_stems)

            if match_ratio == 0:
                scores.append(0.005)
                continue

            # Base score scaled from 0.0 to 1.0
            if match_ratio >= 0.99:
                base = 0.95
            elif match_ratio >= 0.66:
                base = 0.78
            elif match_ratio >= 0.49:
                base = 0.52
            else:
                base = 0.20

            # Proximity and exact string boost
            if q_norm in c_norm:
                base = min(0.99, base + 0.10)

            # Negation divergence: if query asks for allergy and text says "aucune allergie"
            if has_q_negation != has_c_negation and ("allerg" in q_norm or "contr" in q_norm):
                # Still relevant because it answers the allergy question, but distinguished
                base = round(base * 0.85, 4)

            scores.append(round(min(0.995, max(0.005, base)), 4))

        return scores

    # Backwards-compatible name.
    _score_calibrated_heuristic = _score_keyword_heuristic

    # ------------------------------------------------------------------ pruning

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


def re_words(text: str) -> List[str]:
    import re
    return [w for w in re.findall(r"\w+", text) if len(w) > 1]
