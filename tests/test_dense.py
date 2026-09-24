import contextlib
import functools
import hashlib
import http.server
import importlib.util
import io
import json
import os
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nanoprune.core.calibration import fit_platt

HAS_STACK = all(importlib.util.find_spec(m) is not None for m in ("torch", "onnx", "onnxruntime", "tokenizers"))

CORPUS = [
    "le salarié démissionnaire effectue un préavis",
    "le locataire donne congé au bailleur",
    "les chiens doivent être tenus en laisse",
    "allergie sévère aux pénicillines",
    "contrat de travail et période d essai",
]


def build_tiny_model(model_dir: Path, pooling: str = "mean") -> Path:
    """A 16-dimensional embedding 'transformer' exported to ONNX, with its tokenizer.json."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(CORPUS + ["query: passage:"], trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]"]))
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "config.json").write_text(json.dumps({"pad_token_id": 0}), encoding="utf-8")

    class TinyEncoder(torch.nn.Module):
        def __init__(self, vocab):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(vocab, 16)
            self.proj = torch.nn.Linear(16, 16)

        def forward(self, input_ids, attention_mask):
            hidden = self.proj(self.emb(input_ids))
            return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)

    module = TinyEncoder(tokenizer.get_vocab_size()).eval()
    ids = torch.ones((2, 5), dtype=torch.long)
    kwargs = dict(input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
                  dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                "last_hidden_state": {0: "b", 1: "s"}}, opset_version=17)
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(module, (ids, torch.ones_like(ids)), str(model_dir / "model.onnx"), **kwargs)
    (model_dir / "nanoprune-dense.json").write_text(json.dumps({
        "name": model_dir.name, "pooling": pooling, "query_prefix": "query: ", "passage_prefix": "passage: ",
        "max_length": 32, "license": "test",
    }), encoding="utf-8")
    return model_dir


class TestPlatt(unittest.TestCase):
    def test_separable_data_gives_finite_increasing_calibration(self):
        a, b = fit_platt([0.70, 0.72, 0.75, 0.85, 0.88, 0.90], [0, 0, 0, 1, 1, 1])
        self.assertTrue(np.isfinite(a) and np.isfinite(b))
        self.assertGreater(a, 0)
        prob = lambda x: 1 / (1 + np.exp(-(a * x + b)))
        self.assertLess(prob(0.70), 0.5)
        self.assertGreater(prob(0.90), 0.5)

    def test_needs_both_classes(self):
        with self.assertRaises(ValueError):
            fit_platt([0.1, 0.2], [1, 1])


@unittest.skipUnless(HAS_STACK, "needs torch, onnx, onnxruntime and tokenizers")
class TestSemanticScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stderr(io.StringIO()):
            cls.model_dir = build_tiny_model(Path(cls.tmp.name) / "tiny-e5")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scorer(self):
        from nanoprune.engine.dense import SemanticScorer
        return SemanticScorer(self.model_dir)

    def test_embeddings_are_normalised_and_padding_invariant(self):
        from nanoprune.engine.dense import DenseEncoder
        encoder = DenseEncoder(self.model_dir)
        alone = encoder.encode([CORPUS[3]])
        batched = encoder.encode([CORPUS[3], CORPUS[0] + " " + CORPUS[4]])
        self.assertEqual(alone.shape, (1, 16))
        self.assertAlmostEqual(float(np.linalg.norm(alone[0])), 1.0, places=5)
        np.testing.assert_allclose(alone[0], batched[0], atol=1e-5)

    def test_calibration_scores_and_cache(self):
        scorer = self.scorer()
        with self.assertRaises(RuntimeError):
            scorer.score("préavis", CORPUS)  # not calibrated yet
        calibration = scorer.calibrate("dev", save=False)
        self.assertEqual(calibration["fitted_on"], "dev")
        scores = scorer.score("préavis démission", CORPUS)
        self.assertEqual(len(scores), len(CORPUS))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

        calls = []
        original = scorer.encoder.encode
        scorer.encoder.encode = lambda texts, kind="passage": calls.append((kind, len(texts))) or original(texts, kind)
        self.assertEqual(scorer.warm(CORPUS), 0)  # already cached by score()
        scorer.score("autre requête", CORPUS)
        self.assertEqual(calls, [("query", 1)])
        self.assertEqual(scorer.prune("préavis", CORPUS, threshold=0.0)[0][1], max(scorer.score("préavis", CORPUS)))

    def test_cache_eviction_keeps_scores_correct(self):
        scorer = self.scorer()
        scorer.calibrate("dev", save=False)
        expected = scorer.score("préavis", CORPUS)
        small = self.scorer()
        small.a, small.b, small.cache_size = scorer.a, scorer.b, 2
        np.testing.assert_allclose(small.score("préavis", CORPUS), expected, atol=1e-6)
        self.assertLessEqual(len(small._cache), 2)

    def test_search_engine_scores_every_passage(self):
        from nanoprune.engine.indexer import LocalDocumentIndexer
        from nanoprune.engine.search import LocalSearchEngine
        scorer = self.scorer()
        scorer.calibrate("dev", save=False)
        indexer = LocalDocumentIndexer()
        for i, text in enumerate(CORPUS):
            indexer.index_text(text, f"doc{i}.md")
        engine = LocalSearchEngine(indexer=indexer, pruner=scorer)
        engine.prepare()
        self.assertTrue(all(chunk.text in scorer._cache for chunk in indexer.chunks))
        res = engine.search("zzz inconnu", top_k=10, threshold=0.0)
        self.assertEqual(res["backend"], "semantic")
        self.assertEqual(res["candidates_evaluated"], len(CORPUS))  # no keyword match needed
        scores = [r["score"] for r in res["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_calibration_is_saved_and_cli_uses_it(self):
        from nanoprune.cli import main
        from nanoprune.engine.dense import read_dense_config
        with tempfile.TemporaryDirectory() as tmp:
            model = build_tiny_model(Path(tmp) / "tiny", pooling="cls")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["calibrate", "--dense", str(model)]), 0)
            self.assertIn("calibration", read_dense_config(model))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["prune", "--dense", str(model), "préavis", *CORPUS[:2], "--threshold", "0"]), 0)
            self.assertIn("sémantique", out.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(main(["prune", "--dense", str(model), "--model", "x.pt", "q", "t"]), 1)
            self.assertIn("either --dense", err.getvalue())

    def test_installed_models_are_discovered(self):
        from nanoprune.engine.dense import SemanticScorer, installed_dense_models
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"XDG_CACHE_HOME": tmp}):
            os.environ.pop("NANOPRUNE_DENSE_MODEL", None)
            self.assertEqual(installed_dense_models(), [])
            with self.assertRaises(FileNotFoundError):
                SemanticScorer.load("auto")
            root = Path(tmp) / "nanoprune" / "dense"
            build_tiny_model(root / "model-a")
            build_tiny_model(root / "model-a-int8")
            self.assertEqual([p.name for p in installed_dense_models()], ["model-a-int8", "model-a"])
            self.assertEqual(SemanticScorer.load().model_dir.name, "model-a-int8")

    def test_teacher_labels_with_a_scorer(self):
        from nanoprune.training.teacher import label_file
        scorer = self.scorer()
        scorer.calibrate("dev", save=False)
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
            rows = [{"pair_id": f"p{i}", "query": "préavis", "content": text, "label": 1.0} for i, text in enumerate(CORPUS)]
            src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
            self.assertEqual(label_file(src, dst, scorer=scorer, log=lambda _: None), len(CORPUS))
            self.assertEqual(label_file(src, dst, scorer=scorer, log=lambda _: None), 0)  # resumable
            labelled = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(0.0 <= r["teacher"] <= 1.0 for r in labelled))
            with self.assertRaises(ValueError):
                label_file(src, dst, log=lambda _: None)


@unittest.skipUnless(HAS_STACK, "needs torch, onnx, onnxruntime and tokenizers")
class TestDenseDownload(unittest.TestCase):
    def test_download_quantise_and_calibrate(self):
        from nanoprune import download
        from nanoprune.engine import dense
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with contextlib.redirect_stderr(io.StringIO()):
                source = build_tiny_model(tmp / "src" / "fast-tiny")
            archive = tmp / "served" / "tiny.tar.gz"
            archive.parent.mkdir()
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(source, arcname="fast-tiny")
            handler = functools.partial(_QuietHandler, directory=str(archive.parent))
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                spec = dict(dense.KNOWN_DENSE_MODELS["multilingual-e5-large"])
                spec.update(url=f"http://127.0.0.1:{server.server_address[1]}/tiny.tar.gz",
                            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), archive_dir="fast-tiny",
                            query_prefix="", passage_prefix="")
                with mock.patch.dict(dense.KNOWN_DENSE_MODELS, {"tiny": spec}):
                    target = download.download_dense_model("tiny", dest=tmp / "dense", log=lambda _: None)
                    self.assertEqual(target.name, "tiny-int8")
                    self.assertFalse((tmp / "dense" / "tiny").exists())  # float32 copy removed
                    config = dense.read_dense_config(target)
                    self.assertEqual(config["quantization"], "int8 dynamic (onnxruntime)")
                    self.assertIn("calibration", config)

                    spec["sha256"] = "0" * 64
                    with self.assertRaises(download.DownloadError):
                        download.download_dense_model("tiny", dest=tmp / "other", log=lambda _: None)
            finally:
                server.shutdown()
                server.server_close()

    def test_unsafe_archive_is_refused(self):
        from nanoprune import download
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "evil.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                data = b"owned"
                info = tarfile.TarInfo("../evil.txt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            with self.assertRaises(download.DownloadError):
                download._safe_extract(archive, Path(tmp) / "out")
            self.assertFalse((Path(tmp) / "evil.txt").exists())


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    unittest.main()
