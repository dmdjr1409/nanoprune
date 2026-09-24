import importlib.util
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from nanoprune.engine.pruner import NanoPruner
from nanoprune.errors import HeadUnavailableError, ModelNotFoundError, NanoPruneWarning, TokenizerMismatchError

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ONNX = HAS_TORCH and importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


class TestNanoPruner(unittest.TestCase):
    def test_pruner_heuristic_scoring(self):
        pruner = NanoPruner(model_path=None)
        query = "Allergie pénicilline"
        c1 = "Le patient présente une allergie sévère aux pénicillines."
        c2 = "Le patient n'a aucune allergie connue."
        c3 = "Le traitement comprend du paracétamol 1g le matin."

        scores = pruner.score(query, [c1, c2, c3])
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], scores[2])
        self.assertTrue(0.0 <= scores[0] <= 1.0)
        self.assertEqual(pruner.backend, "heuristic")

    def test_pruner_prune_threshold(self):
        pruner = NanoPruner(model_path=None, threshold=0.60)
        query = "Allergie pénicilline"
        candidates = [
            "Patient allergique à la pénicilline",
            "Ordonnance de lunettes de repos",
        ]
        retained = pruner.prune(query, candidates)
        self.assertEqual(len(retained), 1)
        self.assertIn("allergique", retained[0][0])

    def test_rank(self):
        pruner = NanoPruner(model_path=None, threshold=0.5)
        ranked = pruner.rank("allergie", [{"text": "Allergie connue."}, {"text": "Rien."}])
        self.assertEqual(len(ranked), 1)
        self.assertIn("nanoprune_score", ranked[0])

    def test_heuristic_has_no_choice_or_score_head(self):
        pruner = NanoPruner(model_path=None)
        with self.assertRaises(HeadUnavailableError):
            pruner.choice("texte", ["a", "b"])
        with self.assertRaises(HeadUnavailableError):
            pruner.score_rubric("texte")
        with self.assertRaises(ValueError):
            pruner.choice("texte", [])


class TestModelDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        patcher = mock.patch.dict(os.environ, {
            "NANOPRUNE_HOME": str(self.home / "home"),
            "XDG_CACHE_HOME": str(self.home / "cache"),
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("NANOPRUNE_MODEL", None)
        # Ignore weights that may exist in a developer's source checkout.
        repo_patch = mock.patch("nanoprune.core.weights._is_source_checkout", return_value=False)
        repo_patch.start()
        self.addCleanup(repo_patch.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_weights_warn_and_fall_back(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pruner = NanoPruner.load()
        self.assertEqual(pruner.backend, "heuristic")
        self.assertTrue(any(issubclass(w.category, NanoPruneWarning) for w in caught))
        self.assertIn("warning", pruner.describe())

    def test_missing_weights_strict(self):
        with self.assertRaises(ModelNotFoundError):
            NanoPruner.load(strict=True)

    def test_explicit_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            NanoPruner(model_path=self.home / "nope.pt")

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_legacy_char_checkpoint_is_discovered(self):
        import torch
        from nanoprune.core.model import NanoPruneModel
        weights = self.home / "home" / "weights"
        weights.mkdir(parents=True)
        model = NanoPruneModel(vocab_size=321, d_model=32, n_heads=2, d_ff=64, n_layers=1, head_hidden=64)
        torch.save(model.state_dict(), weights / "nanoprune-v0.3.pt")
        pruner = NanoPruner.load(strict=True)
        self.assertEqual(pruner.backend, "torch")
        self.assertEqual(pruner.tokenizer.kind, "char")
        scores = pruner.score("allergie", ["Allergie connue.", "Rien."])
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_wordpiece_checkpoint_without_tokenizer_is_refused(self):
        import torch
        from nanoprune.core.model import NanoPruneModel
        path = self.home / "nanoprune-v0.4.pt"
        torch.save(NanoPruneModel(vocab_size=8192, d_model=32, n_heads=2, d_ff=64, n_layers=1).state_dict(), path)
        with self.assertRaises(FileNotFoundError):
            NanoPruner(model_path=path)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_unknown_checkpoint_without_manifest_warns(self):
        import torch
        from nanoprune.core.model import NanoPruneModel
        path = self.home / "my-model.pt"
        torch.save(NanoPruneModel(vocab_size=321, d_model=32, n_heads=4, d_ff=64, n_layers=1).state_dict(), path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pruner = NanoPruner(model_path=path)
        self.assertEqual(pruner.backend, "torch")
        self.assertTrue(any("no manifest" in str(w.message) for w in caught))

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
    def test_vocabulary_mismatch_is_detected(self):
        import torch
        from nanoprune.core.model import NanoPruneModel
        from nanoprune.core.tokenizer import NanoTokenizer
        path = self.home / "custom.pt"
        torch.save(NanoPruneModel(vocab_size=8192, d_model=32, n_heads=2, d_ff=64, n_layers=1).state_dict(), path)
        with self.assertRaises(TokenizerMismatchError):
            NanoPruner(model_path=path, tokenizer=NanoTokenizer.char_level())

    @unittest.skipUnless(HAS_ONNX, "torch, onnx and onnxruntime are needed")
    def test_onnx_matches_torch_and_broken_onnx_falls_back(self):
        import onnx
        import torch
        from nanoprune.core.export import export_to_onnx
        from nanoprune.core.model import NanoPruneModel

        weights = self.home / "home" / "weights"
        weights.mkdir(parents=True)
        # Released v0.3 layout: 4 attention heads below d_model 256 (no manifest needed).
        model = NanoPruneModel(vocab_size=321, d_model=32, n_heads=4, d_ff=64, n_layers=1, head_hidden=64).eval()
        torch.save(model.state_dict(), weights / "nanoprune-v0.3.pt")
        export_to_onnx(model, weights / "nanoprune-v0.3.onnx", heads="all")
        self.assertFalse(model.training)
        self.assertFalse((weights / "nanoprune-v0.3.onnx.data").exists())

        docs = ["Allergie sévère aux pénicillines.", "Ordonnance de lunettes."]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from_onnx = NanoPruner.load(strict=True)
            from_torch = NanoPruner(model_path=weights / "nanoprune-v0.3.pt")
        self.assertEqual(from_onnx.backend, "onnx")
        for a, b in zip(from_onnx.score("allergie", docs), from_torch.score("allergie", docs)):
            self.assertAlmostEqual(a, b, places=5)
        labels = ["human_rights", "business_tax", "public_admin", "civil_family"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(from_onnx.choice("texte", labels)[0], from_torch.choice("texte", labels)[0])
            self.assertAlmostEqual(from_onnx.score_rubric("texte"), from_torch.score_rubric("texte"), places=4)

        # Graph whose external weights file went missing (like the v0.3 release asset).
        graph = onnx.load(str(weights / "nanoprune-v0.3.onnx"))
        onnx.save_model(graph, str(weights / "nanoprune-v0.3.onnx"), save_as_external_data=True,
                        location="nanoprune-v0.3.onnx.data", size_threshold=0)
        (weights / "nanoprune-v0.3.onnx.data").unlink()
        with self.assertRaises(FileNotFoundError):
            NanoPruner(model_path=weights / "nanoprune-v0.3.onnx")
        fallback = NanoPruner.load(strict=True)
        self.assertEqual(fallback.backend, "torch")


if __name__ == "__main__":
    unittest.main()
