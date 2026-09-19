import unittest
from nanoprune.engine.pruner import NanoPruner

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

if __name__ == "__main__":
    unittest.main()
