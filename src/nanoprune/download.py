"""Download model files published as GitHub release assets.

Every asset is checked against the SHA-256 digest GitHub reports for it before
it is moved into place (by default ``~/.cache/nanoprune/weights``, where
``NanoPruner.load()`` looks for weights).
"""
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .core.weights import cache_dir, onnx_initializer_info

DEFAULT_REPO = "dmdjr1409/nanoprune"
API_URL = "https://api.github.com/repos/{repo}/releases/{ref}"
MODEL_ASSET_SUFFIXES = (".onnx", ".onnx.data", ".pt", ".pth", ".json")


class DownloadError(RuntimeError):
    pass


def _get(url: str, accept: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "nanoprune-downloader"})


def release_info(repo: str = DEFAULT_REPO, tag: Optional[str] = None) -> Dict:
    ref = f"tags/{tag}" if tag else "latest"
    try:
        with urllib.request.urlopen(_get(API_URL.format(repo=repo, ref=ref), "application/vnd.github+json"), timeout=30) as resp:
            return json.load(resp)
    except Exception as exc:
        raise DownloadError(f"Cannot read release '{tag or 'latest'}' of {repo}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_release(
    tag: Optional[str] = None,
    dest: Optional[Path] = None,
    repo: Optional[str] = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> List[Path]:
    """Download the model assets of a release; returns the local paths."""
    repo = repo or os.environ.get("NANOPRUNE_REPO", DEFAULT_REPO)
    dest = Path(dest).expanduser() if dest else cache_dir() / "weights"
    info = release_info(repo, tag)
    assets = [a for a in info.get("assets", []) if a.get("name", "").endswith(MODEL_ASSET_SUFFIXES)]
    if not assets:
        raise DownloadError(f"Release {info.get('tag_name')} of {repo} has no model assets.")
    dest.mkdir(parents=True, exist_ok=True)
    log(f"Release {info.get('tag_name')} ({repo}) -> {dest}")

    paths = []
    for asset in assets:
        name = Path(asset["name"]).name
        target = dest / name
        expected = (asset.get("digest") or "")
        expected = expected.split(":", 1)[1] if expected.startswith("sha256:") else ""
        if target.exists() and not force and expected and _sha256(target) == expected:
            log(f"  = {name} (already present)")
            paths.append(target)
            continue
        fd, tmp_name = tempfile.mkstemp(dir=str(dest), prefix=".download-")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
                _get(asset["browser_download_url"], "application/octet-stream"), timeout=120
            ) as resp:
                for block in iter(lambda: resp.read(1 << 20), b""):
                    out.write(block)
            actual = _sha256(tmp)
            if expected and actual != expected:
                raise DownloadError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
            tmp.replace(target)
        finally:
            if tmp.exists():
                tmp.unlink()
        log(f"  + {name} ({target.stat().st_size / 1024:.0f} KiB, sha256 {'verified' if expected else 'not published'})")
        paths.append(target)

    for path in paths:
        if path.suffix == ".onnx":
            missing = [f for f in onnx_initializer_info(path).get("external_files", []) if not (path.parent / f).exists()]
            if missing:
                log(f"  ! {path.name} is unusable: the release lacks {', '.join(missing)} (its weights file).")
    return paths


def _download_verified(url: str, target: Path, sha256: str, log: Callable[[str], None]) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".download-")
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        done, next_report = 0, 0
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(_get(url, "application/octet-stream"), timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            for block in iter(lambda: resp.read(1 << 20), b""):
                out.write(block)
                digest.update(block)
                done += len(block)
                if total and done >= next_report:
                    log(f"  {done / total * 100:5.1f} % of {total / 1e9:.2f} GB")
                    next_report += total // 10
        if digest.hexdigest() != sha256:
            raise DownloadError(f"Checksum mismatch for {url}: expected {sha256}, got {digest.hexdigest()}")
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract regular files and directories only, refusing paths that leave ``dest``."""
    import tarfile

    root = dest.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.name.startswith("._") or not (member.isfile() or member.isdir()):
                continue  # macOS resource forks, links, devices
            target = (dest / name).resolve()
            if root not in target.parents and target != root:
                raise DownloadError(f"Unsafe path in archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(target, "wb") as out:
                for block in iter(lambda: src.read(1 << 20), b""):
                    out.write(block)


def download_dense_model(
    name: str = "multilingual-e5-large",
    int8: bool = True,
    keep_fp32: bool = False,
    dest: Optional[Path] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Install a sentence-embedding model for semantic search and calibrate it.

    With ``int8`` (default) the model is quantised with onnxruntime (about 4x
    smaller, 2x faster, same accuracy on the evaluation suites); the float32
    copy is then removed unless ``keep_fp32``. Returns the model directory.
    """
    import shutil

    from .engine.dense import KNOWN_DENSE_MODELS, SemanticScorer, dense_models_dir, write_dense_config

    if name not in KNOWN_DENSE_MODELS:
        raise DownloadError(f"Unknown model '{name}'. Known: {', '.join(KNOWN_DENSE_MODELS)}")
    spec = KNOWN_DENSE_MODELS[name]
    root = Path(dest).expanduser() if dest else dense_models_dir()
    root.mkdir(parents=True, exist_ok=True)
    fp32_dir = root / name
    settings = {k: spec[k] for k in ("pooling", "query_prefix", "passage_prefix", "max_length")}
    settings.update({"source": spec["source"], "license": spec["license"]})

    if not (fp32_dir / "model.onnx").exists():
        archive = root / f".{name}.tar.gz"
        log(f"Downloading {spec['source']} ({spec['license']}) from {spec['url']}")
        _download_verified(spec["url"], archive, spec["sha256"], log)
        staging = root / f".{name}-extract"
        shutil.rmtree(staging, ignore_errors=True)
        try:
            _safe_extract(archive, staging)
            shutil.move(str(staging / spec["archive_dir"]), str(fp32_dir))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if archive.exists():
                archive.unlink()
    write_dense_config(fp32_dir, {"name": name, **settings})
    target = fp32_dir

    if int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as exc:
            raise DownloadError("int8 quantisation needs the onnx package: pip install 'nanoprune[train]' "
                                "or use --no-int8") from exc
        target = root / f"{name}-int8"
        target.mkdir(exist_ok=True)
        log("Quantising to int8 (about a minute)...")
        quantize_dynamic(str(fp32_dir / "model.onnx"), str(target / "model.onnx"), weight_type=QuantType.QInt8)
        for extra in ("tokenizer.json", "config.json", "tokenizer_config.json", "special_tokens_map.json"):
            if (fp32_dir / extra).exists():
                shutil.copyfile(fp32_dir / extra, target / extra)
        write_dense_config(target, {"name": f"{name}-int8", **settings, "quantization": "int8 dynamic (onnxruntime)"})
        if not keep_fp32:
            shutil.rmtree(fp32_dir)

    log("Calibrating on the dev suite...")
    calibration = SemanticScorer(target).calibrate("dev")
    log(f"Installed {target} (calibration a={calibration['a']:.2f}, b={calibration['b']:.2f})")
    return target
