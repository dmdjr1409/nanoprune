import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from nanoprune.engine.extractors import MissingDependencyError, extract_text_from_bytes
from nanoprune.engine.indexer import LocalDocumentIndexer
from nanoprune.engine.pruner import NanoPruner
from nanoprune.engine.search import LocalSearchEngine
from nanoprune.errors import OperationCancelled

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data" / "medical"


def make_docx(paragraphs):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def make_pdf(text):
    """Smallest valid one-page PDF showing ``text`` (Latin-1)."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 144] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


HAS_PYPDF = importlib.util.find_spec("pypdf") is not None


class TestLocalIndexerAndSearch(unittest.TestCase):
    def test_indexing_and_search(self):
        self.assertTrue(SAMPLE_DIR.exists())

        indexer = LocalDocumentIndexer()
        files_indexed = indexer.index_directory(str(SAMPLE_DIR))
        self.assertGreaterEqual(files_indexed, 3)
        self.assertGreaterEqual(len(indexer.chunks), 3)

        engine = LocalSearchEngine(indexer=indexer, pruner=NanoPruner(model_path=None))
        res = engine.search("Allergie pénicilline Dupont")

        self.assertGreater(res["matches_retained"], 0)
        self.assertEqual(res["backend"], "heuristic")
        top = res["results"][0]
        self.assertIn("dupont", top["file_name"].lower())
        self.assertGreater(top["confidence"], 0.50)
        self.assertEqual(top["score"], top["confidence"])
        self.assertIn("allergie", top["highlight"].lower())
        self.assertGreaterEqual(top["line_end"], top["line_start"])

    def test_search_without_lexical_match_returns_nothing(self):
        indexer = LocalDocumentIndexer()
        indexer.index_directory(str(SAMPLE_DIR))
        engine = LocalSearchEngine(indexer=indexer, pruner=NanoPruner(model_path=None))
        self.assertEqual(engine.search("xylophone quantique")["results"], [])
        self.assertEqual(engine.search("   ")["results"], [])

    def test_index_is_rebuilt_after_changes(self):
        indexer = LocalDocumentIndexer()
        engine = LocalSearchEngine(indexer=indexer, pruner=NanoPruner(model_path=None))
        indexer.index_text("Clause de non-concurrence de douze mois.", "contrat.md")
        self.assertEqual(len(engine.search("non-concurrence")["results"]), 1)
        indexer.clear()
        indexer.index_text("Le bail commercial est conclu pour neuf ans.", "bail.md")
        self.assertEqual(engine.search("non-concurrence")["results"], [])
        self.assertEqual(len(engine.search("bail commercial")["results"]), 1)


class FixedScorer:
    """Scores passages from a {substring: score} table (0 otherwise)."""

    backend = "fixed"
    has_model = True
    threshold = 0.5

    def __init__(self, table):
        self.table = table

    def score(self, query, candidates):
        return [max([v for k, v in self.table.items() if k in text] or [0.0]) for text in candidates]


class TestSearchResults(unittest.TestCase):
    def engine(self, table, **kwargs):
        indexer = LocalDocumentIndexer(chunk_size=120, chunk_overlap=60)
        # One long paragraph (cut into overlapping passages) and two short ones.
        indexer.index_text(
            "Le bail commence en janvier. Le loyer est payable chaque mois. Le dépôt de garantie vaut deux mois. "
            "Le locataire assure le logement. Le bailleur fait les grosses réparations. "
            "Les charges sont réglées chaque trimestre. Le logement est rendu en bon état. "
            "Les clés sont remises à la sortie des lieux.\n\n"
            "Les animaux sont acceptés.\n\nLe préavis de départ est de trois mois.",
            "bail.md",
        )
        return LocalSearchEngine(indexer=indexer, pruner=FixedScorer(table), **kwargs), indexer

    def test_mostly_repeated_passages_are_not_shown_twice(self):
        indexer = LocalDocumentIndexer(chunk_size=120, chunk_overlap=80)
        indexer.index_text("Le bail commence en janvier pour trois ans. "
                           "Le loyer est payable chaque mois avant le cinq du mois courant. "
                           "Aucune pénalité de retard.", "bail.md")
        first, second = indexer.chunks
        self.assertEqual(second.metadata["shared_prev"], len("Le loyer est payable chaque mois avant le cinq du mois courant."))
        engine = LocalSearchEngine(indexer=indexer, pruner=FixedScorer({first.body: 0.9, second.body: 0.8}))
        self.assertEqual([r["chunk_id"] for r in engine.search("bail", threshold=0.5)["results"]], [first.chunk_id])

    def test_passages_sharing_little_text_are_all_shown(self):
        engine, indexer = self.engine({})
        long_passages = [c for c in indexer.chunks if c.line_start == 1]
        self.assertGreaterEqual(len(long_passages), 3)
        self.assertTrue(all(c.metadata.get("shared_prev", 0) > 0 for c in long_passages[1:]))
        engine.pruner.table = {c.body: 0.9 - 0.05 * i for i, c in enumerate(long_passages)}
        res = engine.search("bail", top_k=10, threshold=0.5)
        mostly_new = [c.chunk_id for c in long_passages if 2 * c.metadata.get("shared_prev", 0) < len(c.body)]
        self.assertGreaterEqual(len(mostly_new), 3)
        self.assertEqual([r["chunk_id"] for r in res["results"] if r["chunk_id"] in mostly_new], mostly_new)

    def test_near_misses_and_counts(self):
        engine, _ = self.engine({"préavis": 0.8, "animaux": 0.4, "janvier": 0.1})
        res = engine.search("préavis", top_k=5, threshold=0.5)
        self.assertEqual([r["rel_path"] for r in res["results"]], ["bail.md"])
        self.assertEqual(res["matches_total"], 1)
        self.assertFalse(res["more_available"])
        # 0.4 is at least half the threshold, 0.1 is not.
        self.assertEqual([r["score"] for r in res["near_misses"]], [0.4])

        engine.pruner.table = {"préavis": 0.8, "animaux": 0.7}
        res = engine.search("préavis", top_k=1, threshold=0.5)
        self.assertEqual(res["matches_total"], 2)
        self.assertTrue(res["more_available"])
        self.assertEqual(res["near_misses"], [])

    def test_highlight_terms_and_best_sentence(self):
        engine = LocalSearchEngine(indexer=LocalDocumentIndexer(), pruner=NanoPruner(model_path=None))
        engine.indexer.index_text("Dossier de M. Dupont.\nAllergies : pénicillines (sévère).", "dupont.md")
        top = engine.search("allergie à la pénicilline", threshold=0.0)["results"][0]
        self.assertEqual(top["highlight"], "Allergies : pénicillines (sévère).")
        self.assertEqual(top["highlight_terms"], ["allergies", "pénicillines"])
        # No shared word (a semantic match): the whole passage is quoted.
        self.assertEqual(LocalSearchEngine._best_sentence({"xyz"}, "Titre\nPremière ligne."), "Titre\nPremière ligne.")
        # A sentence matching only a common word does not stand for the passage.
        idf = {"pas": 0.2, "antibio": 3.0}.get
        text = "Pas de fièvre.\nAllergie aux pénicillines."
        self.assertEqual(LocalSearchEngine._best_sentence({"pas", "antibio"}, text, idf), text)
        # Query words found in no passage (weight 0) do not dilute the coverage.
        self.assertEqual(LocalSearchEngine._best_sentence({"allergi", "inconnu"}, text, {"allergi": 1.0}.get),
                         "Allergie aux pénicillines.")
        self.assertEqual(LocalSearchEngine._best_sentence({"pas", "fievr"}, text), "Pas de fièvre.")


class TestChunking(unittest.TestCase):
    def test_index_text_adds_chunks_with_line_numbers(self):
        indexer = LocalDocumentIndexer()
        chunks = indexer.index_text("Titre\n\nPremier paragraphe.\nSuite du paragraphe.\n\n---\n\nDernier.", "note.md")
        self.assertEqual(len(indexer.chunks), len(chunks))
        self.assertEqual([c.body for c in chunks], ["Titre", "Premier paragraphe.\nSuite du paragraphe.", "Dernier."])
        self.assertEqual([(c.line_start, c.line_end) for c in chunks], [(1, 1), (3, 4), (8, 8)])
        self.assertTrue(chunks[1].text.startswith("[Note] "))

    def test_long_paragraph_overlap(self):
        text = " ".join(f"Phrase numéro {i} sur le contrat de travail." for i in range(40))
        chunks = LocalDocumentIndexer(chunk_size=300, chunk_overlap=80).index_text(text, "long.md")
        self.assertGreater(len(chunks), 3)
        for previous, current in zip(chunks, chunks[1:]):
            last_sentence = previous.body.split(". ")[-1].rstrip(".")
            self.assertTrue(current.body.startswith(last_sentence))
            self.assertLessEqual(len(current.text), 300 + 80)

    def test_directory_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("Contrat de bail.", encoding="utf-8")
            (root / "b.exe").write_bytes(b"MZ")
            (root / ".cache").mkdir()
            (root / ".cache" / "c.md").write_text("caché", encoding="utf-8")
            (root / "node_modules").mkdir()
            (root / "node_modules" / "d.md").write_text("dépendance", encoding="utf-8")
            (root / "big.txt").write_text("x" * 2000, encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub" / "e.docx").write_bytes(make_docx(["Clause de préavis.", "Deuxième paragraphe."]))

            indexer = LocalDocumentIndexer(max_file_bytes=1000)
            self.assertEqual(indexer.index_directory(str(root)), 2)
            names = sorted({c.chunk_id.split("#")[0] for c in indexer.chunks})
            self.assertEqual(names, ["a.md", "sub/e.docx"])
            self.assertEqual(indexer.last_report["skipped"][0][1], "file too large")

    def test_directory_progress_and_cancel(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(4):
                (Path(tmp) / f"f{i}.txt").write_text(f"document {i}", encoding="utf-8")
            calls = []
            indexer = LocalDocumentIndexer()
            indexer.index_directory(tmp, progress=lambda done, total: calls.append((done, total)))
            self.assertEqual(calls, [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)])

            stop_after = iter([False, False, False, False, False, True])
            with self.assertRaises(OperationCancelled):
                LocalDocumentIndexer().index_directory(tmp, should_stop=lambda: next(stop_after, True))

    def test_max_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                (Path(tmp) / f"f{i}.txt").write_text(f"document {i}", encoding="utf-8")
            indexer = LocalDocumentIndexer(max_files=3)
            self.assertEqual(indexer.index_directory(tmp), 3)
            self.assertTrue(indexer.last_report["truncated"])


class TestExtractors(unittest.TestCase):
    def test_docx(self):
        text = extract_text_from_bytes(make_docx(["Premier.", "Second &amp; dernier."]), "x.docx")
        self.assertEqual(text, "Premier.\n\nSecond & dernier.")

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            extract_text_from_bytes(b"x", "x.exe")

    @unittest.skipUnless(HAS_PYPDF, "pypdf is not installed")
    def test_pdf(self):
        text = extract_text_from_bytes(make_pdf("Clause de preavis de trois mois"), "bail.pdf")
        self.assertIn("Clause de preavis", text)

    @unittest.skipIf(HAS_PYPDF, "pypdf is installed")
    def test_pdf_without_pypdf_is_reported(self):
        with self.assertRaises(MissingDependencyError):
            extract_text_from_bytes(make_pdf("x"), "x.pdf")
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.pdf").write_bytes(make_pdf("x"))
            indexer = LocalDocumentIndexer()
            self.assertEqual(indexer.index_directory(tmp), 0)
            self.assertIn("pypdf", indexer.last_report["skipped"][0][1])


if __name__ == "__main__":
    unittest.main()
