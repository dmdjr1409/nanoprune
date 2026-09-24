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
