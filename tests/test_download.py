import functools
import hashlib
import http.server
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from nanoprune import download


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.served = root / "served"
        self.served.mkdir()
        self.dest = root / "dest"
        (self.served / "nanoprune-v9.json").write_text('{"format": "nanoprune-model"}', encoding="utf-8")
        (self.served / "nanoprune-v9.pt").write_bytes(b"weights" * 100)
        (self.served / "notes.txt").write_text("not a model asset", encoding="utf-8")

        handler = functools.partial(_QuietHandler, directory=str(self.served))
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def release(self, digests=None):
        assets = []
        for path in sorted(self.served.iterdir()):
            digest = (digests or {}).get(path.name, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
            assets.append({"name": path.name, "browser_download_url": f"{self.base}/{path.name}", "digest": digest})
        return {"tag_name": "v9.0.0", "assets": assets}

    def test_downloads_model_assets_with_checksums(self):
        logs = []
        with mock.patch.object(download, "release_info", return_value=self.release()):
            paths = download.download_release(tag="v9.0.0", dest=self.dest, log=logs.append)
        self.assertEqual(sorted(p.name for p in paths), ["nanoprune-v9.json", "nanoprune-v9.pt"])
        self.assertEqual((self.dest / "nanoprune-v9.pt").read_bytes(), b"weights" * 100)
        self.assertFalse(any(p.name.startswith(".download-") for p in self.dest.iterdir()))

        with mock.patch.object(download, "release_info", return_value=self.release()):
            download.download_release(tag="v9.0.0", dest=self.dest, log=logs.append)
        self.assertTrue(any("already present" in line for line in logs))

    def test_checksum_mismatch_is_fatal(self):
        bad = self.release({"nanoprune-v9.pt": "sha256:" + "0" * 64})
        with mock.patch.object(download, "release_info", return_value=bad):
            with self.assertRaises(download.DownloadError):
                download.download_release(tag="v9.0.0", dest=self.dest, log=lambda _: None)
        self.assertFalse((self.dest / "nanoprune-v9.pt").exists())

    def test_release_without_model_assets(self):
        with mock.patch.object(download, "release_info", return_value={"tag_name": "x", "assets": []}):
            with self.assertRaises(download.DownloadError):
                download.download_release(dest=self.dest, log=lambda _: None)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()
