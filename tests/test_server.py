import base64
import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from nanoprune.app.server import CONTENT_SECURITY_POLICY, make_server
from nanoprune.engine.indexer import DocumentChunk
from nanoprune.engine.pruner import NanoPruner
from nanoprune.errors import OperationCancelled

sys.path.insert(0, str(Path(__file__).parent))
from test_indexer import make_docx  # noqa: E402

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data" / "medical"


class ServerTestCase(unittest.TestCase):
    sample_dir = SAMPLE_DIR

    @classmethod
    def make_pruner(cls):
        return NanoPruner(model_path=None)

    @classmethod
    def setUpClass(cls):
        cls.opened = []
        cls.server = make_server(port=0, sample_data_dir=cls.sample_dir, pruner=cls.make_pruner(),
                                 log_requests=False, opener=cls.opened.append)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", host or f"127.0.0.1:{self.port}")
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            conn.putheader("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            conn.putheader(key, value)
        conn.endheaders(data)
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        return resp, payload

    def post_json(self, path, body, **headers):
        return self.request("POST", path, body, {"Content-Type": "application/json", **headers})

    def status(self):
        return json.loads(self.request("GET", "/api/status")[1])

    def wait_for_job(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.status()["job"]
            if job and job["state"] != "running":
                return job
            time.sleep(0.02)
        self.fail("the import did not finish")


class TestServer(ServerTestCase):
    # ------------------------------------------------------------------ happy path

    def test_page_is_served_with_security_headers(self):
        resp, body = self.request("GET", "/")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Security-Policy"), CONTENT_SECURITY_POLICY)
        self.assertEqual(resp.getheader("X-Frame-Options"), "DENY")
        self.assertIsNone(resp.getheader("Access-Control-Allow-Origin"))
        self.assertNotIn(b"onclick", body)
        self.assertNotIn(b"googleapis", body)
        # Everything the CSP would block: inline styles and scripts.
        self.assertNotIn(b"style=", body)
        self.assertNotIn(b"<style", body)
        self.assertEqual(body.count(b"<script"), body.count(b'<script src="/app.js">'))

    def test_static_assets(self):
        for path in ("/app.js", "/app.css"):
            resp, _ = self.request("GET", path)
            self.assertEqual(resp.status, 200)
        resp, _ = self.request("GET", "/server.py")
        self.assertEqual(resp.status, 404)

    def test_status_reports_the_real_backend(self):
        resp, body = self.request("GET", "/api/status")
        status = json.loads(body)
        self.assertEqual(resp.status, 200)
        self.assertEqual(status["model"]["backend"], "heuristic")
        self.assertIn("warning", status["model"])

    def test_search(self):
        resp, body = self.post_json("/api/search", {"query": "Allergie pénicilline Dupont", "threshold": 0.45},
                                    Origin=f"http://127.0.0.1:{self.port}", **{"Sec-Fetch-Site": "same-origin"})
        self.assertEqual(resp.status, 200)
        results = json.loads(body)["results"]
        self.assertTrue(results)
        self.assertIn("dupont", results[0]["file_name"])

    # ------------------------------------------------------------------ attacks

    def test_cross_origin_simple_request_is_refused(self):
        resp, body = self.request("POST", "/api/load_folder", b'{"folder_path": "/etc"}',
                                  {"Content-Type": "text/plain", "Origin": "https://evil.example"})
        self.assertEqual(resp.status, 403)
        self.assertIsNone(resp.getheader("Access-Control-Allow-Origin"))

    def test_non_json_content_type_is_refused(self):
        resp, _ = self.request("POST", "/api/search", b'{"query": "x"}', {"Content-Type": "text/plain"})
        self.assertEqual(resp.status, 415)

    def test_cross_site_fetch_metadata_is_refused(self):
        resp, _ = self.post_json("/api/search", {"query": "x"}, **{"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(resp.status, 403)

    def test_foreign_host_is_refused(self):
        resp, _ = self.request("GET", "/api/status", host=f"evil.example:{self.port}")
        self.assertEqual(resp.status, 403)
        resp, _ = self.request("GET", "/", host="evil.example")
        self.assertEqual(resp.status, 403)

    def test_preflight_is_never_granted(self):
        resp, _ = self.request("OPTIONS", "/api/search", headers={"Origin": "https://evil.example",
                                                                   "Access-Control-Request-Method": "POST"})
        self.assertEqual(resp.status, 405)
        self.assertIsNone(resp.getheader("Access-Control-Allow-Origin"))

    def test_invalid_bodies(self):
        resp, _ = self.request("POST", "/api/search", b"{bad", {"Content-Type": "application/json"})
        self.assertEqual(resp.status, 400)
        resp, _ = self.post_json("/api/search", ["not", "an", "object"])
        self.assertEqual(resp.status, 400)
        resp, _ = self.post_json("/api/search", {"query": "x", "threshold": "abc"})
        self.assertEqual(resp.status, 400)
        resp, _ = self.post_json("/api/unknown", {})
        self.assertEqual(resp.status, 404)

    def test_oversized_body_is_refused(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.putrequest("POST", "/api/search")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(10 ** 9))
        conn.endheaders()
        resp = conn.getresponse()
        self.assertEqual(resp.status, 413)
        conn.close()

    def test_load_folder_rejects_missing_directory(self):
        resp, body = self.post_json("/api/load_folder", {"folder_path": "/definitely/not/here"})
        self.assertEqual(resp.status, 400)
        self.assertIn("error", json.loads(body))

    # ------------------------------------------------------------------ imports

    def test_drag_and_drop_import_indexes_documents(self):
        files = [
            {"name": "note.md", "content": "Le patient Durand présente une allergie au latex.\n\nAutre paragraphe."},
            {"name": "note.md", "content": "Deuxième fichier du même nom : contrôle du latex."},
            {"name": "contrat.docx", "data_base64": base64.b64encode(make_docx(["Clause de préavis de trois mois."])).decode()},
            {"name": "virus.exe", "content": "MZ"},
            {"name": "broken.docx", "data_base64": "***"},
            {"name": "vide.txt", "content": "   "},
        ]
        try:
            resp, body = self.post_json("/api/index_direct", {"files": files, "wait": True})
            report = json.loads(body)
            self.assertEqual(resp.status, 200)
            self.assertEqual(report["files_indexed"], 3)
            self.assertEqual(report["skipped_count"], 3)
            self.assertIn({"file": "vide.txt", "reason": "aucun texte lisible"}, report["skipped"])
            self.assertGreaterEqual(report["total_chunks"], 4)
            self.assertEqual(report["job"]["state"], "done")
            self.assertFalse(self.status()["is_sample"])

            resp, body = self.post_json("/api/search", {"query": "allergie latex", "threshold": 0.3})
            top = json.loads(body)["results"][0]
            self.assertEqual(top["file_name"], "note.md")
            self.assertFalse(top["can_open"])  # uploaded: no file on disk
            resp, body = self.post_json("/api/open", {"chunk_id": top["chunk_id"]})
            self.assertEqual(resp.status, 400)
            resp, body = self.post_json("/api/search", {"query": "clause de préavis", "threshold": 0.3})
            self.assertEqual(json.loads(body)["results"][0]["file_name"], "contrat.docx")
        finally:
            self.post_json("/api/load_folder", {"folder_path": str(SAMPLE_DIR), "wait": True})

    def test_background_folder_import(self):
        resp, body = self.post_json("/api/load_folder", {"folder_path": str(SAMPLE_DIR)})
        self.assertEqual(resp.status, 202)
        self.assertEqual(json.loads(body)["job"]["kind"], "folder")
        job = self.wait_for_job()
        self.assertEqual(job["state"], "done")
        self.assertEqual(job["report"]["files_indexed"], 3)
        self.assertEqual(job["done"], job["total"])
        status = self.status()
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["files_indexed"], 3)
        self.assertTrue(status["is_sample"])

    def test_open_file_of_a_result(self):
        resp, body = self.post_json("/api/search", {"query": "Allergie pénicilline Dupont", "threshold": 0.45})
        top = json.loads(body)["results"][0]
        self.assertTrue(top["can_open"])
        self.assertEqual(top["rel_path"], "patient_dupont_marc.md")
        del self.opened[:]
        resp, body = self.post_json("/api/open", {"chunk_id": top["chunk_id"]})
        self.assertEqual(resp.status, 200)
        self.assertEqual(self.opened, [str((SAMPLE_DIR / "patient_dupont_marc.md").resolve())])
        resp, _ = self.post_json("/api/open", {"chunk_id": "../../etc/passwd#0"})
        self.assertEqual(resp.status, 404)
        self.assertEqual(len(self.opened), 1)

    def test_open_needs_a_desktop_and_a_document(self):
        context = self.server.RequestHandlerClass.context
        resp, body = self.post_json("/api/search", {"query": "Allergie pénicilline Dupont", "threshold": 0.45})
        top = json.loads(body)["results"][0]
        opener, context.opener = context.opener, None  # headless machine
        try:
            resp, body = self.post_json("/api/search", {"query": "Allergie pénicilline Dupont", "threshold": 0.45})
            self.assertFalse(json.loads(body)["results"][0]["can_open"])
            resp, body = self.post_json("/api/open", {"chunk_id": top["chunk_id"]})
            self.assertEqual(resp.status, 500)
            self.assertIn("session graphique", json.loads(body)["error"])
        finally:
            context.opener = opener
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "script.sh"
            script.write_text("echo owned", encoding="utf-8")
            link = Path(tmp) / "note.md"
            link.symlink_to(script)
            document = Path(tmp) / "vrai.md"
            document.write_text("Un document.", encoding="utf-8")

            def chunk(path):
                return DocumentChunk(f"{path.name}#0", str(path), path.name, "x")

            self.assertIsNone(context._file_of(chunk(link)))  # a ".md" link to a script is never opened
            self.assertEqual(context._file_of(chunk(document)), document.resolve())

    def test_synchronous_helpers(self):
        context = self.server.RequestHandlerClass.context
        report = context.load_directory(str(SAMPLE_DIR))
        self.assertEqual(report["files_indexed"], 3)

    def test_cancel_without_import_is_harmless(self):
        resp, body = self.post_json("/api/cancel", {})
        self.assertEqual(resp.status, 200)
        self.assertEqual(json.loads(body)["status"], "idle")

    def test_import_requires_files(self):
        resp, _ = self.post_json("/api/index_direct", {"files": []})
        self.assertEqual(resp.status, 400)


class BlockingScorer:
    """A semantic-like scorer whose embedding step waits until released (or cancelled)."""

    backend = "semantic"
    has_model = True
    full_scan = True
    threshold = 0.5

    def __init__(self):
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.fail = False

    def embed(self, passages, progress=None, should_stop=None):
        self.started.set()
        if self.fail:
            raise RuntimeError("model crashed")
        while not self.release.is_set():
            if should_stop is not None and should_stop():
                raise OperationCancelled("cancelled")
            time.sleep(0.01)
        if progress is not None:
            progress(len(passages), len(passages))
        return np.ones((len(passages), 2), dtype=np.float32)

    def score_embeddings(self, query, matrix):
        return np.full(len(matrix), 0.9)

    def score(self, query, candidates):
        return [0.9] * len(candidates)

    def describe(self):
        return {"backend": "semantic", "model_name": "blocking"}


class TestImportJobs(ServerTestCase):
    @classmethod
    def make_pruner(cls):
        cls.scorer = BlockingScorer()
        return cls.scorer

    def setUp(self):
        self.scorer.release.set()
        self.scorer.started.clear()
        self.scorer.fail = False

    def test_cancel_keeps_the_previous_index(self):
        before = self.status()
        self.assertEqual(before["files_indexed"], 3)
        self.scorer.release.clear()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "autre.md").write_text("Un tout autre document.", encoding="utf-8")
            resp, _ = self.post_json("/api/load_folder", {"folder_path": tmp})
            self.assertEqual(resp.status, 202)
            self.assertTrue(self.scorer.started.wait(5))
            status = self.status()
            self.assertEqual(status["status"], "indexing")
            self.assertEqual(status["job"]["phase"], "embedding")

            # One import at a time; the previous index stays searchable meanwhile.
            resp, _ = self.post_json("/api/load_folder", {"folder_path": tmp})
            self.assertEqual(resp.status, 409)
            resp, body = self.post_json("/api/search", {"query": "allergie", "threshold": 0.5})
            self.assertTrue(json.loads(body)["results"])

            resp, body = self.post_json("/api/cancel", {})
            self.assertEqual(json.loads(body)["status"], "cancelling")
            self.assertEqual(self.wait_for_job()["state"], "cancelled")
        after = self.status()
        self.assertEqual(after["indexed_chunks"], before["indexed_chunks"])
        self.assertEqual(after["current_folder"], before["current_folder"])

    def test_failed_import_is_reported(self):
        self.scorer.fail = True
        resp, body = self.post_json("/api/load_folder", {"folder_path": str(SAMPLE_DIR), "wait": True})
        self.assertEqual(resp.status, 500)
        self.assertIn("model crashed", json.loads(body)["error"])
        self.assertEqual(self.status()["job"]["state"], "error")


if __name__ == "__main__":
    unittest.main()
