import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanoprune import __version__
from nanoprune.cli import main

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data" / "medical"


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        env = mock.patch.dict(os.environ, {
            "NANOPRUNE_HOME": str(Path(self.tmp.name) / "home"),
            "XDG_CACHE_HOME": str(Path(self.tmp.name) / "cache"),
        })
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("NANOPRUNE_MODEL", None)
        repo = mock.patch("nanoprune.core.weights._is_source_checkout", return_value=False)
        repo.start()
        self.addCleanup(repo.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_version(self):
        code, out, _ = self.run_cli("--version")
        self.assertEqual(code, 0)
        self.assertIn(__version__, out)

    def test_prune_reports_heuristic_mode(self):
        code, out, err = self.run_cli("prune", "Allergie pénicilline", "Allergie à la pénicilline.", "Lunettes.")
        self.assertEqual(code, 0)
        self.assertIn("heuristique", out)
        self.assertIn("Conservés 1/2", out)
        self.assertIn("No NanoPrune model weights found", err)

    def test_strict_mode_fails_without_weights(self):
        code, _, err = self.run_cli("prune", "--strict", "q", "t")
        self.assertEqual(code, 1)
        self.assertIn("Erreur", err)

    def test_search(self):
        code, out, _ = self.run_cli("search", "Allergie pénicilline Dupont", "--dir", str(SAMPLE_DIR))
        self.assertEqual(code, 0)
        self.assertIn("patient_dupont_marc.md:", out)
        self.assertIn("Score mots-clés", out)

    def test_search_missing_directory(self):
        code, _, err = self.run_cli("search", "x", "--dir", "/definitely/not/here")
        self.assertEqual(code, 1)
        self.assertIn("Directory does not exist", err)

    def test_choice_and_score_need_a_model(self):
        self.assertEqual(self.run_cli("choice", "texte")[0], 2)
        self.assertEqual(self.run_cli("score", "texte")[0], 2)

    def test_info(self):
        code, out, _ = self.run_cli("info")
        self.assertEqual(code, 0)
        self.assertIn("Répertoires de poids", out)

    def test_eval_baselines(self):
        report = Path(self.tmp.name) / "eval.json"
        code, out, _ = self.run_cli("eval", "--no-model", "--suite", "dev", "--bootstrap", "0", "--json", str(report))
        self.assertEqual(code, 0)
        self.assertIn("keyword heuristic", out)
        self.assertTrue(report.exists())

    def test_no_command_prints_help(self):
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("usage", out)


if __name__ == "__main__":
    unittest.main()
