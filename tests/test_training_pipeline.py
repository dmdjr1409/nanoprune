import importlib.util
import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path

from nanoprune.training.data import (
    CATEGORY_LABELS,
    IGNORE_CATEGORY,
    Record,
    build_pairs,
    read_legal_tsv,
    split_of,
    weak_category,
    write_jsonl,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
HAS_ORT = importlib.util.find_spec("onnxruntime") is not None and importlib.util.find_spec("onnx") is not None

TOPICS = [
    ("CEDH - droits de l'homme", "liberté d'expression", "La Cour examine une ingérence dans la liberté d'expression d'un journaliste condamné pour diffamation"),
    ("Luxembourg - commerce", "faillite commerciale", "Le tribunal de commerce prononce la faillite de la société et désigne un curateur pour les créanciers"),
    ("Luxembourg - administration", "nomination ministérielle", "Par arrêté ministériel, le fonctionnaire est nommé au poste de directeur de l'administration"),
    ("Luxembourg - civil", "succession familiale", "Le partage de la succession entre les enfants respecte la réserve héréditaire prévue par le code civil"),
    ("Luxembourg - commerce", "taxe sur la valeur ajoutée", "Les livraisons intracommunautaires sont exonérées de TVA lorsque le bien quitte le territoire"),
    ("Luxembourg - civil", "divorce et garde des enfants", "Le juge fixe la résidence des enfants et la pension alimentaire après le divorce des époux"),
]


def synthetic_tsv(path: Path, n_per_topic: int = 30, seed: int = 0) -> None:
    rng = random.Random(seed)
    lines = []
    for jurisdiction, title, sentence in TOPICS:
        for i in range(n_per_topic):
            extra = rng.choice(["en première instance", "en appel", "selon la jurisprudence", "par décision motivée"])
            lines.append(f"{jurisdiction}\t{title.capitalize()} n°{i}\t{sentence} {extra}, affaire numéro {i}.")
    rng.shuffle(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestPairGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tsv = Path(self.tmp.name) / "legal_raw.tsv"
        synthetic_tsv(self.tsv)

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_is_deterministic(self):
        self.assertEqual(split_of("Titre A"), split_of("Titre A"))
        self.assertIn(split_of("Titre B"), ("train", "val", "test"))

    def test_no_title_or_passage_shared_between_splits(self):
        records = read_legal_tsv(self.tsv)
        splits = build_pairs(records, seed=1)
        seen = {}
        for split, rows in splits.items():
            for row in rows:
                for key in (row["query"], row["content"]):
                    seen.setdefault(key, set()).add(split)
        self.assertTrue(all(len(s) == 1 for s in seen.values()))
        self.assertGreater(len(splits["train"]), len(splits["val"]))

    def test_hard_negatives_share_vocabulary_but_not_the_document(self):
        records = read_legal_tsv(self.tsv)
        rows = build_pairs(records, seed=1)["train"]
        hard = [r for r in rows if r["type"] == "hard_negative"]
        self.assertTrue(hard)
        by_id = {r.doc_id: r for r in records}
        for row in hard:
            self.assertNotEqual(by_id[row["doc_id"]].title, row["query"])
        from nanoprune.engine.lexical import tokenize
        overlapping = sum(1 for r in hard if set(tokenize(r["query"])) & set(tokenize(r["content"])))
        self.assertGreater(overlapping / len(hard), 0.8)

    def test_category_follows_the_passage(self):
        record = Record("x", "Luxembourg", "Faillite", "Le tribunal de commerce prononce la faillite de la société.")
        self.assertEqual(CATEGORY_LABELS[weak_category(record)], "business_tax")
        unclear = Record("y", "Luxembourg", "Divers", "Texte sans mot-clé particulier.")
        self.assertEqual(weak_category(unclear), IGNORE_CATEGORY)


@unittest.skipUnless(HAS_TORCH and HAS_TOKENIZERS and HAS_ORT, "needs torch, tokenizers, onnx and onnxruntime")
class TestTrainingEndToEnd(unittest.TestCase):
    def test_train_export_and_reload(self):
        from nanoprune import NanoPruner
        from nanoprune.training.data import train_wordpiece
        from nanoprune.training.trainer import TrainConfig, train

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tsv = tmp / "legal_raw.tsv"
            synthetic_tsv(tsv)
            records = read_legal_tsv(tsv)
            tokenizer_path = tmp / "tokenizer.json"
            train_wordpiece((f"{r.title} {r.content}" for r in records), tokenizer_path, vocab_size=400, min_frequency=1)
            for split, rows in build_pairs(records, seed=3).items():
                write_jsonl(rows, tmp / f"contrastive_{split}.jsonl")

            config = TrainConfig(
                data_dir=str(tmp), out_dir=str(tmp / "weights"), name="nanoprune-test",
                tokenizer_path=str(tokenizer_path), epochs=2, batch_size=16, max_len=64,
                d_model=32, n_heads=2, d_ff=64, n_layers=1, head_hidden=16, device="cpu",
                eval_suites=("dev",),
            )
            metrics = train(config, log=lambda message: None)
            self.assertIn("val", metrics)
            self.assertIn("dev", metrics["suites"])

            out = tmp / "weights"
            manifest = json.loads((out / "nanoprune-test.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tokenizer"]["type"], "wordpiece")
            self.assertEqual(manifest["files"], {"pt": "nanoprune-test.pt", "onnx": "nanoprune-test.onnx"})
            self.assertFalse(manifest["heads"]["score"])
            self.assertFalse(any(p.suffix == ".data" for p in out.iterdir()))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from_pt = NanoPruner(model_path=out / "nanoprune-test.pt")
                from_onnx = NanoPruner(model_path=out / "nanoprune-test.onnx")
            self.assertEqual(from_pt.tokenizer.kind, "wordpiece")
            query, docs = "faillite commerciale", ["Le tribunal de commerce prononce la faillite.", "Divorce des époux."]
            for a, b in zip(from_pt.score(query, docs), from_onnx.score(query, docs)):
                self.assertAlmostEqual(a, b, places=4)
            # The score head is declared untrained by the manifest.
            self.assertFalse(from_onnx.available_heads()["score"])
            self.assertTrue(from_onnx.available_heads()["choice"])


if __name__ == "__main__":
    unittest.main()
