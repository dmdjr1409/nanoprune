import unittest
import warnings

from nanoprune.engine.cascade import HybridCascadePruner
from nanoprune.errors import NanoPruneWarning


class FixedScorer:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, candidates):
        return [self.scores[c] for c in candidates]


class ScriptedLaya:
    def __init__(self, probs):
        self.probs = probs
        self.seen = []

    def system_one(self, state, questions):
        doc = state[len("Document: "):]
        self.seen.append(doc)
        if self.probs[doc] is None:
            raise RuntimeError("teacher failure")
        return {"answers": {"relevance": {"noul": self.probs[doc]}}}


QUERY = "préavis de démission"
DOCS = {
    "no_overlap": "Les chiens sont tenus en laisse.",
    "confident": "Le préavis de démission est d'un mois.",
    "laya_yes": "Démission : durée du préavis selon la convention.",
    "rescued": "Préavis applicable à la démission du cadre.",
    "trap": "Préavis de congé du locataire.",
    "teacher_error": "Démission et préavis en période d'essai.",
}


class TestCascade(unittest.TestCase):
    def test_decision_rules(self):
        tier1 = FixedScorer({
            DOCS["no_overlap"]: 0.2,
            DOCS["confident"]: 0.97,
            DOCS["laya_yes"]: 0.6,
            DOCS["rescued"]: 0.8,
            DOCS["trap"]: 0.6,
            DOCS["teacher_error"]: 0.55,
        })
        laya = ScriptedLaya({
            DOCS["laya_yes"]: 0.8,
            DOCS["rescued"]: 0.3,
            DOCS["trap"]: 0.1,
            DOCS["teacher_error"]: None,
        })
        cascade = HybridCascadePruner(nanoprune_model=tier1, enable_laya=False, laya_agent=laya)
        res = cascade.prune_cascade(QUERY, list(DOCS.values()))
        tags = {text: tag for text, _, tag in res["retained"]}

        self.assertEqual(tags[DOCS["confident"]], "tier1_match")
        self.assertEqual(tags[DOCS["laya_yes"]], "tier2_laya_kept")
        self.assertEqual(tags[DOCS["rescued"]], "tier2_rescued_by_nanoprune")
        self.assertEqual(tags[DOCS["teacher_error"]], "tier1_fallback_kept")
        self.assertNotIn(DOCS["trap"], tags)
        self.assertNotIn(DOCS["no_overlap"], tags)
        self.assertNotIn(DOCS["no_overlap"], laya.seen)
        self.assertNotIn(DOCS["confident"], laya.seen)

        stats = res["stats"]
        self.assertEqual(stats["tier1_fast_dropped"], 1)
        self.assertEqual(stats["tier1_fast_kept"], 1)
        self.assertEqual(stats["tier2_arbitrated"], 4)
        self.assertEqual(stats["laya_calls"], 4)
        self.assertEqual(res["dropped_count"], 2)

    def test_without_laya(self):
        tier1 = FixedScorer({DOCS["laya_yes"]: 0.6, DOCS["trap"]: 0.4})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cascade = HybridCascadePruner(nanoprune_model=tier1, enable_laya=True, laya_model="not/installed")
        self.assertIsNone(cascade.laya_agent)
        self.assertTrue(any(issubclass(w.category, NanoPruneWarning) for w in caught))
        res = cascade.prune_cascade(QUERY, [DOCS["laya_yes"], DOCS["trap"]])
        self.assertEqual([tag for _, _, tag in res["retained"]], ["tier1_direct_kept"])
        self.assertEqual(res["stats"]["laya_calls"], 0)

    def test_empty(self):
        cascade = HybridCascadePruner(nanoprune_model=FixedScorer({}), enable_laya=False)
        res = cascade.prune_cascade(QUERY, [])
        self.assertEqual(res["retained"], [])
        self.assertEqual(res["stats"]["total_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
