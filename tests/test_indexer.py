import unittest
from pathlib import Path
from nanoprune.engine.indexer import LocalDocumentIndexer
from nanoprune.engine.search import LocalSearchEngine

class TestLocalIndexerAndSearch(unittest.TestCase):
    def test_indexing_and_search(self):
        sample_dir = Path(__file__).parent.parent / "sample_data" / "medical"
        self.assertTrue(sample_dir.exists())

        indexer = LocalDocumentIndexer()
        files_indexed = indexer.index_directory(str(sample_dir))
        self.assertGreaterEqual(files_indexed, 3)
        self.assertGreaterEqual(len(indexer.chunks), 3)

        from nanoprune.engine.pruner import NanoPruner
        engine = LocalSearchEngine(indexer=indexer, pruner=NanoPruner(model_path=None))
        res = engine.search("Allergie pénicilline Dupont")

        self.assertGreater(res["matches_retained"], 0)
        top = res["results"][0]
        self.assertIn("dupont", top["file_name"].lower())
        self.assertGreater(top["confidence"], 0.50)
        self.assertIn("allergie", top["highlight"].lower())

if __name__ == "__main__":
    unittest.main()
