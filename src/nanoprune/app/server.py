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
- ``/api/open`` only opens files that are part of the current index, with the
  operating system's default application (no shell involved).

Imports run in a background thread: the previous index stays searchable until
the new one is ready, ``/api/status`` reports progress and ``/api/cancel``
stops the import.
"""
import base64
import binascii
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit

from ..engine.extractors import (
    SUPPORTED_SUFFIXES,
    MissingDependencyError,
    extract_text_from_bytes,
)
from ..engine.indexer import DocumentChunk, LocalDocumentIndexer
from ..engine.pruner import NanoPruner
from ..engine.search import LocalSearchEngine
from ..errors import OperationCancelled

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


def desktop_session() -> bool:
    """True when files and pages can be shown to a user (not a headless server or an SSH session)."""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def open_with_default_app(path: str) -> None:
    """Open a file with the desktop's default application (arguments, never a shell).

    Raises OSError when the opener is missing or fails straight away.
    """
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
        return
    command = "open" if sys.platform == "darwin" else "xdg-open"
    process = subprocess.Popen(
        [command, path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        code = process.wait(timeout=1.5)
    except subprocess.TimeoutExpired:
        threading.Thread(target=process.wait, daemon=True).start()  # still running: reap it later
        return
    if code != 0:
        raise OSError(f"{command} a échoué (code {code})")


class ImportJob:
    """A folder or upload import running in the background."""

    def __init__(self, kind: str, label: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind            # "folder" or "upload"
        self.label = label
        self.state = "running"      # running, done, error, cancelled
        self.phase = "listing"      # listing, reading, embedding
        self.done = 0
        self.total = 0
        self.error: Optional[str] = None
        self.report: Optional[Dict[str, Any]] = None
        self.started = time.monotonic()
        self.phase_started = self.started
        self.finished_at: Optional[float] = None
        self.finished = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self._cancel.set()

    def should_stop(self) -> bool:
        return self._cancel.is_set()

    def set_phase(self, phase: str, done: int = 0, total: int = 0) -> None:
        with self._lock:
            if phase != self.phase:
                self.phase, self.phase_started = phase, time.monotonic()
            self.done, self.total = done, total

    def progress(self, phase: str) -> Callable[[int, int], None]:
        return lambda done, total: self.set_phase(phase, done, total)

    def finish(self, state: str, report: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        with self._lock:
            self.state, self.report, self.error = state, report, error
            self.finished_at = time.monotonic()
        self.finished.set()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            now = self.finished_at or time.monotonic()
            return {
                "id": self.id,
                "kind": self.kind,
                "label": self.label,
                "state": self.state,
                "phase": self.phase,
                "done": self.done,
                "total": self.total,
                "elapsed_s": round(now - self.started, 2),
                "phase_elapsed_s": round(now - self.phase_started, 2),
                "cancel_requested": self._cancel.is_set(),
                "error": self.error,
                "report": self.report,
            }


class JobConflict(Exception):
    """An import is already running."""


class NanoPruneServerContext:
    """State shared by all requests: the index, the model and the search engine."""

    def __init__(
        self,
        sample_data_dir: Optional[Path] = None,
        pruner: Optional[Any] = None,
        background: bool = False,
        opener: Optional[Callable[[str], None]] = None,
    ):
        """``pruner`` is any relevance scorer: NanoPruner, SemanticScorer... (default: NanoPruner.load()).

        ``sample_data_dir`` is indexed at start-up, in the background when
        ``background`` is true. ``opener(path)`` opens a file for ``/api/open``;
        by default the desktop's application, and nothing on a headless machine.
        """
        self.lock = threading.RLock()
        self.indexer = LocalDocumentIndexer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.pruner = pruner or NanoPruner.load()
        self.model_warnings = sorted({str(w.message) for w in caught})
        self.engine = LocalSearchEngine(indexer=self.indexer, pruner=self.pruner)
        self.chunks_by_id: Dict[str, DocumentChunk] = {}
        self.opener = opener if opener is not None else (open_with_default_app if desktop_session() else None)
        self.sample_dir = str(Path(sample_data_dir).resolve()) if sample_data_dir else None
        self.current_folder: Optional[str] = None
        self.folder_path: Optional[str] = None
        self.files_indexed = 0
        self.last_skipped: List[Dict[str, str]] = []
        self.job: Optional[ImportJob] = None

        if sample_data_dir and Path(sample_data_dir).is_dir():
            self.import_folder(str(sample_data_dir), wait=not background)

    # ------------------------------------------------------------------ imports

    def import_folder(self, dir_path: str, wait: bool = False) -> ImportJob:
        """Index a folder of the local disk (replaces the index once finished)."""
        def work(indexer: LocalDocumentIndexer, job: ImportJob) -> Dict[str, Any]:
            count = indexer.index_directory(dir_path, progress=job.progress("reading"), should_stop=job.should_stop)
            report = indexer.last_report
            return {
                "files_indexed": count,
                "skipped": [{"file": f, "reason": r} for f, r in report.get("skipped", [])],
                "truncated": report.get("truncated", False),
                "folder": dir_path,
                "folder_path": str(Path(dir_path).resolve()),
            }
        return self._start("folder", Path(dir_path).name or dir_path, work, wait)

    def import_files(self, files: List[Dict[str, Any]], wait: bool = False) -> ImportJob:
        """Index documents sent by the browser (drag and drop, file pickers)."""
        def work(indexer: LocalDocumentIndexer, job: ImportJob) -> Dict[str, Any]:
            indexed, skipped = 0, []
            seen: Dict[str, int] = {}
            job.set_phase("reading", 0, len(files))
            for done, entry in enumerate(files, 1):
                if job.should_stop():
                    raise OperationCancelled("Import cancelled")
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
                    content = ""
                chunks = indexer.index_text(content, name, f"import/{name}") if content.strip() else []
                for chunk in chunks:
                    chunk.metadata["uploaded"] = True
                if chunks:
                    indexed += 1
                elif name not in {s["file"] for s in skipped}:
                    skipped.append({"file": name, "reason": "aucun texte lisible"})
                job.set_phase("reading", done, len(files))
            return {
                "files_indexed": indexed,
                "skipped": skipped,
                "truncated": False,
                "folder": "Fichiers importés",
                "folder_path": None,
            }
        return self._start("upload", f"{len(files)} fichier(s)", work, wait)

    def load_directory(self, dir_path: str) -> Dict[str, Any]:
        """Index a folder and wait for the report (synchronous form of ``import_folder``)."""
        return self._report(self.import_folder(dir_path, wait=True))

    def index_files(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Index uploaded documents and wait for the report (synchronous form of ``import_files``)."""
        return self._report(self.import_files(files, wait=True))

    @staticmethod
    def _report(job: ImportJob) -> Dict[str, Any]:
        if job.state != "done":
            raise RuntimeError(job.error or f"import {job.state}")
        return dict(job.report or {})

    def _start(self, kind: str, label: str, work, wait: bool) -> ImportJob:
        with self.lock:
            if self.job is not None and self.job.state == "running":
                raise JobConflict("Un import est déjà en cours")
            job = self.job = ImportJob(kind, label)
        if wait:
            self._run(job, work)
        else:
            threading.Thread(target=self._run, args=(job, work), name=f"nanoprune-import-{job.id}", daemon=True).start()
        return job

    def _run(self, job: ImportJob, work) -> None:
        """Build a new index next to the current one, then swap them."""
        try:
            indexer = LocalDocumentIndexer()
            report = work(indexer, job)
            engine = LocalSearchEngine(indexer=indexer, pruner=self.pruner)
            if hasattr(self.pruner, "embed"):
                job.set_phase("embedding", 0, len(indexer.chunks))
            engine.prepare(progress=job.progress("embedding"), should_stop=job.should_stop)
            if job.should_stop():
                raise OperationCancelled("Import cancelled")
            skipped = report.pop("skipped")
            chunks_by_id = {chunk.chunk_id: chunk for chunk in indexer.chunks}
            with self.lock:
                self.indexer, self.engine, self.chunks_by_id = indexer, engine, chunks_by_id
                self.current_folder = report.pop("folder")
                self.folder_path = report.pop("folder_path")
                self.files_indexed = report["files_indexed"]
                self.last_skipped = skipped
            report.update(
                folder=self.current_folder,
                total_chunks=len(indexer.chunks),
                skipped=skipped[:50],
                skipped_count=len(skipped),
            )
            job.finish("done", report=report)
        except OperationCancelled:
            job.finish("cancelled")
        except Exception as exc:  # reported to the UI rather than lost in a thread
            job.finish("error", error=f"{type(exc).__name__}: {exc}")

    def cancel(self) -> Optional[ImportJob]:
        with self.lock:
            job = self.job
        if job is not None and job.state == "running":
            job.cancel()
            return job
        return None

    # ------------------------------------------------------------------ queries

    def search(self, query: str, threshold: float, top_k: int) -> Dict[str, Any]:
        with self.lock:
            engine, chunks = self.engine, self.chunks_by_id
        response = engine.search(query=query, top_k=top_k, threshold=threshold)
        for result in response["results"] + response["near_misses"]:
            result["can_open"] = self.opener is not None and self._file_of(chunks.get(result["chunk_id"])) is not None
        return response

    @staticmethod
    def _file_of(chunk: Optional[DocumentChunk]) -> Optional[Path]:
        """The document on disk behind an indexed passage (None for browser uploads).

        The resolved path must still be a supported document: a link that now
        points to another kind of file is never handed to the opener.
        """
        if chunk is None or chunk.metadata.get("uploaded"):
            return None
        path = Path(chunk.file_path).resolve()
        return path if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES else None

    def open_passage(self, chunk_id: str) -> str:
        """Open the file of an indexed passage with the default application."""
        with self.lock:
            chunk = self.chunks_by_id.get(chunk_id)
        if chunk is None:
            raise LookupError("Passage introuvable : l'index a changé, relancez la recherche.")
        if chunk.metadata.get("uploaded"):
            raise ValueError("Ce document a été importé par le navigateur : son emplacement sur le disque "
                             "est inconnu. Indiquez le chemin du dossier pour pouvoir ouvrir les fichiers.")
        path = self._file_of(chunk)
        if path is None:
            raise LookupError(f"Le fichier n'existe plus : {chunk.file_path}")
        if self.opener is None:
            raise OSError("aucune session graphique sur la machine du serveur ; fichier : " + str(path))
        self.opener(str(path))
        return str(path)

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": "indexing" if self.job is not None and self.job.state == "running" else "ready",
                "indexed_chunks": len(self.indexer.chunks),
                "files_indexed": self.files_indexed,
                "current_folder": self.current_folder,
                "folder_path": self.folder_path,
                "is_sample": self.sample_dir is not None and self.folder_path == self.sample_dir,
                "job": self.job.to_dict() if self.job is not None else None,
                "model": self.pruner.describe(),
                "model_warnings": self.model_warnings,
                "offline": True,
                "supported_formats": sorted(SUPPORTED_SUFFIXES),
            }


class NanoPruneHandler(BaseHTTPRequestHandler):
    context: Optional[NanoPruneServerContext] = None
    server_version = "NanoPrune"
    sys_version = ""
    quiet = False  # log failed requests only (the page polls /api/status during imports)

    def log_request(self, code="-", size="-"):
        if self.quiet and isinstance(code, int) and code < 400:
            return
        super().log_request(code, size)

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
                self._send_job(*self._load_folder(data))
            elif path == "/api/index_direct":
                self._send_job(*self._index_direct(data))
            elif path == "/api/cancel":
                job = self.context.cancel()
                self._send_json({"status": "cancelling" if job else "idle",
                                 "job": job.to_dict() if job else None})
            elif path == "/api/open":
                self._send_json(self._open(data))
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

    def _start_job(self, start, wait: bool):
        try:
            return start(wait), wait
        except JobConflict as exc:
            raise RequestError(409, str(exc))

    def _send_job(self, job: ImportJob, waited: bool) -> None:
        """202 while the import runs; with ``"wait": true``, the final report."""
        if not waited:
            self._send_json({"status": "started", "job": job.to_dict()}, status=202)
            return
        if job.state == "error":
            self._send_json({"error": f"Échec de l'import : {job.error}", "job": job.to_dict()}, status=500)
            return
        if job.state == "cancelled":
            self._send_json({"status": "cancelled", "job": job.to_dict()})
            return
        self._send_json({"status": "ok", **(job.report or {}), "job": job.to_dict()})

    def _load_folder(self, data: Dict[str, Any]):
        raw_folder = str(data.get("folder_path", "")).strip()
        folder = os.path.expanduser(raw_folder)
        if not raw_folder or not Path(folder).is_dir():
            raise RequestError(400, f"Dossier introuvable : {raw_folder}")
        return self._start_job(lambda wait: self.context.import_folder(folder, wait=wait), bool(data.get("wait")))

    def _index_direct(self, data: Dict[str, Any]):
        files = data.get("files")
        if not isinstance(files, list) or not files:
            raise RequestError(400, "Aucun fichier reçu")
        if len(files) > MAX_UPLOAD_FILES:
            raise RequestError(413, f"Trop de fichiers (maximum {MAX_UPLOAD_FILES})")
        if not all(isinstance(entry, dict) for entry in files):
            raise RequestError(400, "Format de fichier invalide")
        return self._start_job(lambda wait: self.context.import_files(files, wait=wait), bool(data.get("wait")))

    def _open(self, data: Dict[str, Any]) -> Dict[str, Any]:
        chunk_id = str(data.get("chunk_id", ""))
        try:
            path = self.context.open_passage(chunk_id)
        except LookupError as exc:
            raise RequestError(404, str(exc))
        except ValueError as exc:
            raise RequestError(400, str(exc))
        except OSError as exc:
            raise RequestError(500, f"Impossible d'ouvrir le fichier sur cette machine ({exc}).")
        return {"status": "opened", "path": path}


class _LocalServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_server(port: int = 7860, sample_data_dir: Optional[Path] = None,
                pruner: Optional[Any] = None, log_requests: bool = True,
                background: bool = False, opener: Optional[Callable[[str], None]] = None) -> _LocalServer:
    """Create the server (bound to 127.0.0.1 only) without starting it.

    With ``background=True`` the sample folder is indexed in a background
    thread, so the server can answer (and show progress) right away.
    """
    # Bind first: a busy port fails before the (possibly long) start-up indexing.
    server_socket = _LocalServer(("127.0.0.1", port), NanoPruneHandler)
    try:
        context = NanoPruneServerContext(sample_data_dir=sample_data_dir, pruner=pruner,
                                         background=background, opener=opener)
    except BaseException:
        server_socket.server_close()
        raise
    attributes: Dict[str, Any] = {"context": context}
    if not log_requests:
        attributes["log_message"] = lambda self, *args: None
    server_socket.RequestHandlerClass = type("BoundNanoPruneHandler", (NanoPruneHandler,), attributes)
    return server_socket


def run_app(port: int = 7860, sample_data_dir: Optional[Path] = None, pruner: Optional[Any] = None,
            open_browser: bool = True):
    try:
        server = make_server(port=port, sample_data_dir=sample_data_dir, pruner=pruner, background=True)
    except OSError as exc:
        print(f"Impossible d'écouter sur 127.0.0.1:{port} ({exc}). Essayez --port.", file=sys.stderr)
        raise SystemExit(1)
    server.RequestHandlerClass.quiet = True
    context = server.RequestHandlerClass.context
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    info = context.pruner.describe()
    print(f"\n⚡ NanoPrune est prêt : {url}")
    if sample_data_dir:
        print(f"   Dossier : {sample_data_dir} (indexation en cours, suivie dans la page)")
    if info["backend"] == "heuristic":
        print("   Moteur  : mots-clés (aucun modèle). Pour la recherche sémantique :")
        print("             nanoprune download --dense multilingual-e5-large")
    elif info["backend"] == "semantic":
        print(f"   Moteur  : recherche sémantique ({info['model_name']})")
    else:
        print(f"   Moteur  : {info['model_name']} ({info['backend']})")
    print("   🔒 100 % local : écoute sur 127.0.0.1 uniquement, aucune requête sortante.")
    print("   Ctrl+C pour arrêter.")
    if open_browser and desktop_session():  # never a text browser in a terminal
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur NanoPrune.")
    finally:
        server.server_close()
