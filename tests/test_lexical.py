import unittest

from nanoprune.engine.lexical import (
    BM25Index,
    fold_accents,
    idf_table,
    idf_weighted_coverage,
    light_stem,
    tokenize,
)


class TestNormalisation(unittest.TestCase):
    def test_fold_accents(self):
        self.assertEqual(fold_accents("Indemnité ÉCRITE"), "indemnite ecrite")

    def test_stemming_conflates_inflections_but_not_traps(self):
        self.assertEqual(light_stem("indemnite"), light_stem("indemnisation"))
        self.assertEqual(light_stem("salaries"), light_stem("salarie"))
        self.assertNotEqual(light_stem("licenciement"), light_stem("licence"))
        self.assertEqual(light_stem("cdd"), "cdd")

    def test_stopwords_removed_but_negations_kept(self):
        terms = tokenize("Aucune allergie de la patiente")
        self.assertIn("aucun", terms)
        self.assertNotIn("de", terms)
        self.assertNotIn("la", terms)
        self.assertIn("pas", tokenize("pas de fièvre"))


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "Le salarié démissionnaire effectue un préavis de trois mois.",
            "Le locataire donne congé avec un préavis réduit.",
            "Les chiens doivent être tenus en laisse.",
        ]
        self.index = BM25Index(self.docs)

    def test_ranking(self):
        top = self.index.top_k("préavis de démission du salarié", 3)
        self.assertEqual(top[0][0], 0)
        self.assertNotIn(2, [i for i, _ in top])

    def test_no_match_gives_no_candidates(self):
        self.assertEqual(self.index.top_k("vaccination grippe", 5), [])

    def test_empty_index(self):
        self.assertEqual(BM25Index([]).scores("x"), [])


class TestCoverage(unittest.TestCase):
    def test_bounds(self):
        idf = idf_table(["contrat de travail", "bail commercial", "contrat de bail"])
        self.assertEqual(idf_weighted_coverage("contrat de travail", "Le contrat de travail", idf), 1.0)
        self.assertEqual(idf_weighted_coverage("contrat", "Le chien", idf), 0.0)
        partial = idf_weighted_coverage("contrat travail", "Le contrat de bail", idf)
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 0.5)  # "travail" is the rarer, more informative term

    def test_empty_query(self):
        self.assertEqual(idf_weighted_coverage("de la", "texte"), 0.0)


if __name__ == "__main__":
    unittest.main()
