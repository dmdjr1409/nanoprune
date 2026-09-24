"""Locate NanoPrune weights and the tokenizer they were trained with.

Search order (the first location containing a model wins, newest model first):

1. ``$NANOPRUNE_HOME/weights`` and ``$NANOPRUNE_HOME``
2. ``weights/`` of a source checkout (editable install)
3. the user cache, ``$XDG_CACHE_HOME/nanoprune/weights`` (default
   ``~/.cache/nanoprune/weights``), where ``nanoprune download`` stores files

A model is either described by a manifest ``nanoprune-<version>.json`` (written
by the training scripts) or recognised by its legacy file name.
"""
import importlib.util
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tokenizer import DEFAULT_TOKENIZER_NAME, file_sha256

MANIFEST_FORMAT = "nanoprune-model"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Released file stems, newest first, with the tokenizer each one was trained with.
LEGACY_MODELS: List[Tuple[str, str]] = [
    ("nanoprune-v0.4", "wordpiece"),
    ("nanoprune-v0.3", "char"),
    ("nanoprune-legal-v0.2", "char"),
    ("nanoprune-v0.1", "char"),
]
BACKEND_SUFFIXES = {"onnx": (".onnx",), "torch": (".pt", ".pth", ".bin")}
RUNTIME_MODULES = {"onnx": "onnxruntime", "torch": "torch"}


@dataclass
class ModelSpec:
    """Everything needed to load one weights file."""
    path: Path
    backend: str
    name: str
    tokenizer_kind: Optional[str]
    tokenizer_path: Optional[Path] = None
    manifest: Optional[Dict[str, Any]] = None
    manifest_path: Optional[Path] = None
    notes: List[str] = field(default_factory=list)


def backend_for(path: Path) -> str:
    suffix = path.suffix.lower()
    for backend, suffixes in BACKEND_SUFFIXES.items():
        if suffix in suffixes:
            return backend
    raise ValueError(f"Unsupported model file '{path.name}' (expected .onnx, .pt, .pth or .bin)")


def runtime_available(backend: str) -> bool:
    return importlib.util.find_spec(RUNTIME_MODULES[backend]) is not None


def nanoprune_home() -> Optional[Path]:
    value = os.environ.get("NANOPRUNE_HOME")
    return Path(value).expanduser() if value else None


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".cache") / "nanoprune"


def _is_source_checkout(root: Path) -> bool:
    return (root / "pyproject.toml").exists() and (root / "src" / "nanoprune").is_dir()


def weight_dirs() -> List[Path]:
    dirs: List[Path] = []
    home = nanoprune_home()
    if home is not None:
        dirs += [home / "weights", home]
    if _is_source_checkout(REPO_ROOT):
        dirs.append(REPO_ROOT / "weights")
    dirs.append(cache_dir() / "weights")
    return dirs


def data_dirs() -> List[Path]:
    dirs: List[Path] = []
    home = nanoprune_home()
    if home is not None:
        dirs += [home / "data", home]
    if _is_source_checkout(REPO_ROOT):
        dirs.append(REPO_ROOT / "data")
    dirs.append(cache_dir() / "data")
    return dirs


def read_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"{path} is not a NanoPrune model manifest")
    return manifest


def write_manifest(
    path: Path,
    name: str,
    architecture: Dict[str, Any],
    files: Dict[str, str],
    tokenizer_file: Optional[Path] = None,
    temperature: float = 1.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the manifest that ties weights files to their tokenizer."""
    tokenizer: Dict[str, Any] = {"type": "char"}
    if tokenizer_file is not None:
        tokenizer = {
            "type": "wordpiece",
            "file": os.path.relpath(tokenizer_file, path.parent),
            "sha256": file_sha256(tokenizer_file),
        }
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": 1,
        "name": name,
        "architecture": architecture,
        "tokenizer": tokenizer,
        "files": files,
        "temperature": temperature,
        "query_max_tokens": 64,
    }
    manifest.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def _version_key(name: str) -> Tuple[int, ...]:
    match = re.search(r"v(\d+(?:\.\d+)*)", name)
    return tuple(int(part) for part in match.group(1).split(".")) if match else (0,)


def _manifest_for(weights_path: Path) -> Optional[Path]:
    for candidate in sorted(weights_path.parent.glob("*.json")):
        try:
            manifest = read_manifest(candidate)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if weights_path.name in manifest.get("files", {}).values():
            return candidate
    return None


def find_default_tokenizer(weights_dir: Path) -> Optional[Path]:
    """Look for ``tokenizer_legal.json`` next to the weights, then in the data directories."""
    for directory in [weights_dir, weights_dir.parent / "data", *data_dirs()]:
        candidate = directory / DEFAULT_TOKENIZER_NAME
        if candidate.exists():
            return candidate
    return None


def describe_model(weights_path: Path, tokenizer_path: Optional[Path] = None) -> ModelSpec:
    """Describe a weights file: backend, tokenizer kind and tokenizer location."""
    weights_path = Path(weights_path).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"Model file not found: {weights_path}")
    backend = backend_for(weights_path)
    spec = ModelSpec(path=weights_path, backend=backend, name=weights_path.stem, tokenizer_kind=None)

    manifest_path = _manifest_for(weights_path)
    if manifest_path is not None:
        manifest = read_manifest(manifest_path)
        spec.manifest, spec.manifest_path = manifest, manifest_path
        spec.name = manifest.get("name", spec.name)
        tok = manifest.get("tokenizer", {})
        spec.tokenizer_kind = tok.get("type", "char")
        if spec.tokenizer_kind == "wordpiece" and tok.get("file"):
            spec.tokenizer_path = (manifest_path.parent / tok["file"]).resolve()
    else:
        for stem, kind in LEGACY_MODELS:
            if weights_path.stem == stem:
                spec.tokenizer_kind = kind
                break

    if tokenizer_path is not None:
        spec.tokenizer_kind, spec.tokenizer_path = "wordpiece", Path(tokenizer_path).expanduser()
    elif spec.tokenizer_kind == "wordpiece" and spec.tokenizer_path is None:
        spec.tokenizer_path = find_default_tokenizer(weights_path.parent)
    return spec


def _candidates_in(directory: Path) -> List[Tuple[str, Dict[str, Path]]]:
    """(model name, {backend: file}) pairs found in one directory, newest first."""
    found: Dict[str, Dict[str, Path]] = {}
    for manifest_path in directory.glob("*.json"):
        try:
            manifest = read_manifest(manifest_path)
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        files: Dict[str, Path] = {}
        for backend, filename in manifest.get("files", {}).items():
            path = directory / filename
            if path.exists():
                files["torch" if backend in ("pt", "torch") else "onnx"] = path
        if files:
            found[manifest.get("name", manifest_path.stem)] = files
    for stem, _kind in LEGACY_MODELS:
        files = {}
        for backend, suffixes in BACKEND_SUFFIXES.items():
            for suffix in suffixes:
                path = directory / f"{stem}{suffix}"
                if path.exists():
                    files.setdefault(backend, path)
        if files and stem not in found:
            found[stem] = files
    return sorted(found.items(), key=lambda item: _version_key(item[0]), reverse=True)


def discover_candidates(prefer: Sequence[str] = ("onnx", "torch")) -> Tuple[List[ModelSpec], List[str]]:
    """Every loadable model, best first, plus diagnostic notes.

    Models from the first directory that contains any are listed newest
    first; within a model the ``prefer`` order decides between ONNX and PyTorch
    files. Files whose runtime is not installed are reported in the notes.
    ``$NANOPRUNE_MODEL`` overrides the search.
    """
    notes: List[str] = []
    explicit = os.environ.get("NANOPRUNE_MODEL")
    if explicit:
        return [describe_model(Path(explicit))], notes

    for directory in weight_dirs():
        if not directory.is_dir():
            continue
        specs: List[ModelSpec] = []
        for name, files in _candidates_in(directory):
            for backend in prefer:
                path = files.get(backend)
                if path is None:
                    continue
                if not runtime_available(backend):
                    notes.append(
                        f"{path} ignored: install {RUNTIME_MODULES[backend]} to use it "
                        f"(pip install 'nanoprune[{'onnx' if backend == 'onnx' else 'train'}]')."
                    )
                    continue
                specs.append(describe_model(path))
        if specs:
            return specs, notes
    return [], notes


def discover_model(prefer: Sequence[str] = ("onnx", "torch")) -> Tuple[Optional[ModelSpec], List[str]]:
    """The best loadable model (or None) and diagnostic notes."""
    specs, notes = discover_candidates(prefer)
    return (specs[0] if specs else None), notes


def onnx_initializer_info(path: Path) -> Dict[str, Any]:
    """Inspect an ONNX file without loading its weights (needs the ``onnx`` package).

    Returns ``{"vocab_size": int | None, "external_files": [...], "outputs": [...]}``
    or an empty dict when ``onnx`` is not installed.
    """
    try:
        import onnx
    except ImportError:
        return {}
    model = onnx.load(str(path), load_external_data=False)
    info: Dict[str, Any] = {"vocab_size": None, "external_files": [], "outputs": [o.name for o in model.graph.output]}
    external = set()
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    external.add(entry.value)
        if tensor.name.replace("_", ".").endswith("token.embeddings.weight"):
            info["vocab_size"] = int(tensor.dims[0])
    info["external_files"] = sorted(external)
    return info
