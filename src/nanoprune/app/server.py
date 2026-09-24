"""Local web app: a small HTTP server bound to 127.0.0.1.

Security model: the API reads local files and returns their contents, so it must
only ever answer the page it serves itself.

- The ``Host`` header must name this server on the loopback interface, which
  defeats DNS-rebinding attacks.
- API requests must be same-origin: a foreign ``Origin`` or a cross-site
  ``Sec-Fetch-Site`` is refused.
- POST bodies must be ``application/json``, so another site cannot reach the
  API with a "simple" cross-origin request; the CORS preflight such a request
  would need is never granted, and no CORS header is ever sent.
- The page is served with a strict Content-Security-Policy (no inline script,
  no third-party resources) and documents are rendered as text, not HTML.
"""
import base64
import binascii
import json
import os
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from ..engine.extractors import (
    SUPPORTED_SUFFIXES,
    MissingDependencyError,
    extract_text_from_bytes,
)
from ..engine.indexer import LocalDocumentIndexer
from ..engine.pruner import NanoPruner
from ..engine.search import LocalSearchEngine

STATIC_DIR = Path(__file__).parent / "static"
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_UPLOAD_FILES = 2000
MAX_QUERY_CHARS = 2000
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class NanoPruneServerContext:
    """State shared by all requests: the index, the model and the search engine."""

    def __init__(self, sample_data_dir: Optional[Path] = None, pruner: Optional[NanoPruner] = None):
        self.lock = threading.RLock()
        self.indexer = LocalDocumentIndexer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.pruner = pruner or NanoPruner.load()
        self.model_warnings = sorted({str(w.message) for w in caught})
        self.engine = LocalSearchEngine(indexer=self.indexer, pruner=self.pruner)
        self.current_folder: Optional[str] = None
        self.last_skipped: List[Dict[str, str]] = []

        if sample_data_dir and Path(sample_data_dir).is_dir():
            self.load_directory(str(sample_data_dir))

    def load_directory(self, dir_path: str) -> Dict[str, Any]:
        with self.lock:
            self.indexer.clear()
            count = self.indexer.index_directory(dir_path)
            self.current_folder = dir_path
            report = self.indexer.last_report
            self.last_skipped = [{"file": f, "reason": r} for f, r in report.get("skipped", [])]
            return {
                "files_indexed": count,
                "total_chunks": len(self.indexer.chunks),
                "skipped": self.last_skipped[:50],
                "skipped_count": len(self.last_skipped),
                "truncated": report.get("truncated", False),
            }

    def index_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Replace the index with documents sent by the browser (drag and drop)."""
        with self.lock:
            self.indexer.clear()
            indexed, skipped = 0, []
            seen: Dict[str, int] = {}
            for entry in files:
                name = Path(str(entry.get("name") or "document.txt")).name
                seen[name] = seen.get(name, 0) + 1
                if seen[name] > 1:
                    # Same file name dropped from different folders: keep chunk ids unique.
                    name = f"{Path(name).stem} ({seen[name]}){Path(name).suffix}"
                try:
                    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                        raise ValueError("unsupported format")
                    if entry.get("data_base64") is not None:
                        data = base64.b64decode(str(entry["data_base64"]), validate=True)
                        content = extract_text_from_bytes(data, name)
                    else:
                        content = str(entry.get("content") or "")
                except (ValueError, binascii.Error, KeyError, MissingDependencyError) as exc:
                    skipped.append({"file": name, "reason": str(exc)})
                    continue
                if content.strip():
                    self.indexer.index_text(content, name, f"import/{name}")
                    indexed += 1
            self.current_folder = f"Import manuel ({indexed} fichiers)"
            self.last_skipped = skipped
            return {
                "files_indexed": indexed,
                "total_chunks": len(self.indexer.chunks),
                "skipped": skipped[:50],
                "skipped_count": len(skipped),
            }

    def search(self, query: str, threshold: float, top_k: int) -> Dict[str, Any]:
        with self.lock:
            return self.engine.search(query=query, top_k=top_k, threshold=threshold)

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": "ready",
                "indexed_chunks": len(self.indexer.chunks),
                "current_folder": self.current_folder,
                "model": self.pruner.describe(),
                "model_warnings": self.model_warnings,
                "offline": True,
                "supported_formats": sorted(SUPPORTED_SUFFIXES),
            }


class NanoPruneHandler(BaseHTTPRequestHandler):
    context: Optional[NanoPruneServerContext] = None
    server_version = "NanoPrune"
    sys_version = ""

    # ------------------------------------------------------------------ checks

    def _port(self) -> int:
        return self.server.server_address[1]

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        return host in {f"{name}:{self._port()}" for name in LOOPBACK_HOSTS}

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is not None:
            parts = urlsplit(origin)
            if parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS or parts.port != self._port():
                return False
        site = self.headers.get("Sec-Fetch-Site")
        return site is None or site in ("same-origin", "none")

    def _guard(self, api: bool) -> bool:
        if not self._host_allowed():
            self._send_json({"error": "Host non autorisé"}, status=403)
            return False
        if api and not self._same_origin():
            self._send_json({"error": "Requête d'une autre origine refusée"}, status=403)
            return False
        return True

    # ------------------------------------------------------------------ responses

    def _common_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._common_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, filename: str, content_type: str) -> None:
        path = STATIC_DIR / filename
        try:
            body = path.read_bytes()
        except OSError:
            self._send_json({"error": "Fichier introuvable"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.send_header("X-Frame-Options", "DENY")
        self._common_headers()
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------ methods

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in STATIC_ROUTES:
            if self._guard(api=False):
                self._send_static(*STATIC_ROUTES[path])
            return
        if path == "/api/status":
            if self._guard(api=True):
                self._send_json(self.context.status())
            return
        if self._guard(api=False):
            self._send_json({"error": "Introuvable"}, status=404)

    def do_POST(self):
        if not self._guard(api=True):
            return
        try:
            data = self._read_json()
            path = urlsplit(self.path).path
            if path == "/api/search":
                self._send_json(self._search(data))
            elif path == "/api/load_folder":
                self._send_json(self._load_folder(data))
            elif path == "/api/index_direct":
                self._send_json(self._index_direct(data))
            else:
                raise RequestError(404, "Point d'accès inconnu")
        except RequestError as exc:
            self._send_json({"error": exc.message}, status=exc.status)

    def do_OPTIONS(self):
        # CORS preflights are never granted.
        self._send_json({"error": "Méthode non autorisée"}, status=405)

    # ------------------------------------------------------------------ API

    def _read_json(self) -> Dict[str, Any]:
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise RequestError(415, "Content-Type application/json requis")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise RequestError(411, "Content-Length requis")
        if length < 0 or length > MAX_BODY_BYTES:
            raise RequestError(413, "Requête trop volumineuse")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RequestError(400, "JSON invalide")
        if not isinstance(data, dict):
            raise RequestError(400, "Objet JSON attendu")
        return data

    def _search(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query = str(data.get("query", "")).strip()[:MAX_QUERY_CHARS]
        try:
            threshold = min(1.0, max(0.0, float(data.get("threshold", 0.50))))
            top_k = min(50, max(1, int(data.get("top_k", 5))))
        except (TypeError, ValueError):
            raise RequestError(400, "Paramètres threshold/top_k invalides")
        return self.context.search(query, threshold, top_k)

    def _load_folder(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw_folder = str(data.get("folder_path", "")).strip()
        folder = os.path.expanduser(raw_folder)
        if not raw_folder or not Path(folder).is_dir():
            raise RequestError(400, f"Dossier introuvable : {raw_folder}")
        report = self.context.load_directory(folder)
        return {"status": "ok", "folder": folder, **report}

    def _index_direct(self, data: Dict[str, Any]) -> Dict[str, Any]:
        files = data.get("files")
        if not isinstance(files, list) or not files:
            raise RequestError(400, "Aucun fichier reçu")
        if len(files) > MAX_UPLOAD_FILES:
            raise RequestError(413, f"Trop de fichiers (maximum {MAX_UPLOAD_FILES})")
        if not all(isinstance(entry, dict) for entry in files):
            raise RequestError(400, "Format de fichier invalide")
        report = self.context.index_files(files)
        return {"status": "ok", "folder": self.context.current_folder, **report}


class _LocalServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_server(port: int = 7860, sample_data_dir: Optional[Path] = None,
                pruner: Optional[NanoPruner] = None, log_requests: bool = True) -> _LocalServer:
    """Create the server (bound to 127.0.0.1 only) without starting it."""
    context = NanoPruneServerContext(sample_data_dir=sample_data_dir, pruner=pruner)
    attributes: Dict[str, Any] = {"context": context}
    if not log_requests:
        attributes["log_message"] = lambda self, *args: None
    handler = type("BoundNanoPruneHandler", (NanoPruneHandler,), attributes)
    return _LocalServer(("127.0.0.1", port), handler)


def run_app(port: int = 7860, sample_data_dir: Optional[Path] = None, pruner: Optional[NanoPruner] = None):
    try:
        server = make_server(port=port, sample_data_dir=sample_data_dir, pruner=pruner)
    except OSError as exc:
        print(f"Impossible d'écouter sur 127.0.0.1:{port} ({exc}). Essayez --port.", file=sys.stderr)
        raise SystemExit(1)
    context = server.RequestHandlerClass.context
    port = server.server_address[1]
    info = context.pruner.describe()
    print(f"\n⚡ NanoPrune lancé sur http://127.0.0.1:{port}")
    print("🔒 Écoute uniquement en local (127.0.0.1) ; aucune requête réseau sortante.")
    if info["backend"] == "heuristic":
        print("⚠️  Aucun modèle chargé : les scores viennent de l'heuristique par mots-clés.")
    else:
        print(f"🧠 Modèle : {info['model_name']} ({info['backend']})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur NanoPrune.")
    finally:
        server.server_close()
