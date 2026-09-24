import base64
import http.client
import json
import sys
import threading
import unittest
from pathlib import Path

from nanoprune.app.server import CONTENT_SECURITY_POLICY, make_server
from nanoprune.engine.pruner import NanoPruner

sys.path.insert(0, str(Path(__file__).parent))
from test_indexer import make_docx  # noqa: E402

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data" / "medical"


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(port=0, sample_data_dir=SAMPLE_DIR, pruner=NanoPruner(model_path=None),
                                 log_requests=False)
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

    # ------------------------------------------------------------------ happy path

    def test_page_is_served_with_security_headers(self):
        resp, body = self.request("GET", "/")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Security-Policy"), CONTENT_SECURITY_POLICY)
        self.assertEqual(resp.getheader("X-Frame-Options"), "DENY")
        self.assertIsNone(resp.getheader("Access-Control-Allow-Origin"))
        self.assertNotIn(b"onclick", body)
        self.assertNotIn(b"googleapis", body)

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
        ]
        try:
            resp, body = self.post_json("/api/index_direct", {"files": files})
            report = json.loads(body)
            self.assertEqual(resp.status, 200)
            self.assertEqual(report["files_indexed"], 3)
            self.assertEqual(report["skipped_count"], 2)
            self.assertGreaterEqual(report["total_chunks"], 4)

            resp, body = self.post_json("/api/search", {"query": "allergie latex", "threshold": 0.3})
            self.assertEqual(json.loads(body)["results"][0]["file_name"], "note.md")
            resp, body = self.post_json("/api/search", {"query": "clause de préavis", "threshold": 0.3})
            self.assertEqual(json.loads(body)["results"][0]["file_name"], "contrat.docx")
        finally:
            self.post_json("/api/load_folder", {"folder_path": str(SAMPLE_DIR)})

    def test_import_requires_files(self):
        resp, _ = self.post_json("/api/index_direct", {"files": []})
        self.assertEqual(resp.status, 400)


if __name__ == "__main__":
    unittest.main()
