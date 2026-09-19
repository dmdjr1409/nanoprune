import http.server
import json
import socketserver
import os
import sys
from pathlib import Path
from typing import Optional

from ..engine.indexer import LocalDocumentIndexer
from ..engine.pruner import NanoPruner
from ..engine.search import LocalSearchEngine

STATIC_DIR = Path(__file__).parent / "static"

class NanoPruneServerContext:
    def __init__(self, sample_data_dir: Optional[Path] = None):
        self.indexer = LocalDocumentIndexer()
        self.pruner = NanoPruner.load()
        self.engine = LocalSearchEngine(indexer=self.indexer, pruner=self.pruner)
        self.current_folder = None

        if sample_data_dir and sample_data_dir.exists():
            self.load_directory(str(sample_data_dir))

    def load_directory(self, dir_path: str) -> int:
        self.indexer.chunks.clear()
        count = self.indexer.index_directory(dir_path)
        self.current_folder = dir_path
        return count


class NanoPruneHandler(http.server.SimpleHTTPRequestHandler):
    context: Optional[NanoPruneServerContext] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/status":
            self._send_json({
                "status": "ready",
                "indexed_chunks": len(self.context.indexer.chunks),
                "current_folder": self.context.current_folder,
                "model_type": "System One Calibrated Transformer (2.8MB)",
                "offline": True,
            })
            return
        elif self.path == "/" or not self.path.startswith("/api"):
            super().do_GET()
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        data = json.loads(body) if body else {}

        if self.path == "/api/search":
            query = data.get("query", "").strip()
            threshold = float(data.get("threshold", 0.50))
            top_k = int(data.get("top_k", 5))

            results = self.context.engine.search(query=query, top_k=top_k, threshold=threshold)
            self._send_json(results)
            return

        elif self.path == "/api/load_folder":
            raw_folder = data.get("folder_path", "").strip()
            folder = os.path.expanduser(raw_folder)
            target = Path(folder)
            if not folder or not target.exists():
                self._send_json({"error": f"Dossier introuvable : {raw_folder}"}, status=400)
                return
            count = self.context.load_directory(folder)
            self._send_json({
                "status": "ok",
                "files_indexed": count,
                "total_chunks": len(self.context.indexer.chunks),
                "folder": folder,
            })
            return

        elif self.path == "/api/index_direct":
            files = data.get("files", [])
            if not files:
                self._send_json({"error": "Aucun fichier reçu"}, status=400)
                return
            
            # Index directly into the engine
            self.context.indexer.chunks.clear()
            for f in files:
                fname = f.get("name", "document.txt")
                content = f.get("content", "")
                if content:
                    self.context.indexer._chunk_content(content, fname, fname)

            self.context.current_folder = f"Import manuel ({len(files)} fichiers)"
            self._send_json({
                "status": "ok",
                "files_indexed": len(files),
                "total_chunks": len(self.context.indexer.chunks),
                "folder": self.context.current_folder,
            })
            return

        self.send_error(404, "Unknown endpoint")

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_app(port: int = 7860, sample_data_dir: Optional[Path] = None):
    context = NanoPruneServerContext(sample_data_dir=sample_data_dir)
    NanoPruneHandler.context = context

    server = socketserver.TCPServer(("127.0.0.1", port), NanoPruneHandler)
    server.allow_reuse_address = True
    print(f"\n⚡ NanoPrune Desktop Engine lancé sur http://127.0.0.1:{port}")
    print(f"🔒 Mode 100% Hors-Ligne & Confidentiel actif (Zéro donnée cloud)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur NanoPrune.")
        server.server_close()
