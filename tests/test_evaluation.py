import unittest
from collections import Counter

from nanoprune.evaluation import (
    Case,
    binary_metrics,
    bootstrap_ci,
    constant_scorer,
    evaluate_scores,
    format_categories,
    format_table,
    load_suite,
    query_auc,
    roc_auc,
    run_standard_evaluation,
    score_cases,
)


class FakeLaya:
    """Answers like Laya's system_one: relevant when the query's first word is in the document."""

    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        instructions = questions["relevance"]["instructions"]
        query = instructions.split("'")[1]
        first_word = query.split()[0].lower()
        prob = 0.9 if first_word in state.lower() else 0.1
        return {"answers": {"relevance": {"noul": prob}}}


class TestMetrics(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
        self.assertEqual(roc_auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(roc_auc([0, 1, 0, 1], [0.1, 0.4, 0.5, 0.8]), 0.75)
        self.assertIsNone(roc_auc([1, 1], [0.2, 0.3]))

    def test_binary_metrics(self):
        m = binary_metrics([1, 1, 0, 0], [0.9, 0.2, 0.6, 0.1], 0.5)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (1, 1, 1, 1))
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["precision"], 0.5)

    def test_bootstrap_is_deterministic_and_bounded(self):
        labels = [0, 1] * 20
        scores = [0.2, 0.7] * 19 + [0.8, 0.3]
        ci1 = bootstrap_ci(labels, scores, roc_auc, n_resamples=200, seed=1)
        ci2 = bootstrap_ci(labels, scores, roc_auc, n_resamples=200, seed=1)
        self.assertEqual(ci1, ci2)
        self.assertLessEqual(ci1[0], roc_auc(labels, scores))
        self.assertLessEqual(ci1[1], 1.0)

    def test_query_auc_and_categories(self):
        cases = [
            Case("1", "direct_positive", "q1", "a", 1),
            Case("2", "lexical_trap_negative", "q1", "b", 0),
            Case("3", "direct_positive", "q2", "c", 1),
            Case("4", "unrelated_negative", "q2", "d", 0),
        ]
        scores = [0.9, 0.95, 0.8, 0.1]
        self.assertEqual(query_auc(cases, scores), 0.5)
        result = evaluate_scores(cases, scores, threshold=0.5, n_bootstrap=0)
        self.assertEqual(result["traps_fooled"], (1, 1))
        self.assertEqual(result["per_category"]["direct_positive"], (2, 2))
        self.assertIn("ece", result)


class TestSuites(unittest.TestCase):
    def test_bundled_suites(self):
        dev = load_suite("dev")
        heldout = load_suite("heldout")
        self.assertEqual(len(dev), 50)
        self.assertEqual(len(heldout), 200)
        self.assertEqual(Counter(c.label for c in dev), Counter({0: 25, 1: 25}))
        self.assertEqual(Counter(c.category for c in heldout),
                         Counter({"direct_positive": 50, "paraphrase_positive": 50,
                                  "lexical_trap_negative": 50, "unrelated_negative": 50}))
        self.assertEqual(len({c.id for c in heldout}), 200)
        self.assertEqual(len({c.query for c in heldout}), 50)

    def test_heldout_construction_rules(self):
        from nanoprune.engine.lexical import tokenize
        for case in load_suite("heldout"):
            shared = set(tokenize(case.query)) & set(tokenize(case.doc))
            if case.category == "paraphrase_positive":
                self.assertLessEqual(len(shared), 1, case.id)
            elif case.category in ("direct_positive", "lexical_trap_negative"):
                self.assertGreaterEqual(len(shared), 2, case.id)
            else:
                self.assertEqual(shared, set(), case.id)

    def test_score_cases_batches_by_query(self):
        calls = []

        def scorer(query, docs):
            calls.append((query, len(docs)))
            return [0.5] * len(docs)

        cases = load_suite("heldout")
        self.assertEqual(len(score_cases(cases, scorer)), 200)
        self.assertEqual(len(calls), 50)


class TestStandardEvaluation(unittest.TestCase):
    def test_baselines_only(self):
        report = run_standard_evaluation(suites=["dev"], n_bootstrap=0)
        results = report["dev"]
        self.assertAlmostEqual(results["always relevant"]["accuracy"], 0.5)
        self.assertGreaterEqual(results["keyword heuristic (no model)"]["accuracy"], 0.85)
        self.assertNotIn("accuracy", results["BM25 (ranking only)"])
        table = format_table(results)
        self.assertIn("| keyword heuristic (no model) |", table)
        self.assertIn("direct_positive", format_categories(results))

    def test_cascade_ablations_with_a_fake_laya(self):
        laya = FakeLaya()
        report = run_standard_evaluation(suites=["dev"], laya_agent=laya, n_bootstrap=0)
        results = report["dev"]
        self.assertIn("Laya", results)
        self.assertIn("cascade, tier-1 = 0.5 (ablation)", results)
        rate = results["cascade, tier-1 = 0.5 (ablation)"]["laya_call_rate"]
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)  # the lexical gate drops candidates without any shared word
        self.assertGreater(laya.calls, 0)

    def test_constant_scorer(self):
        self.assertEqual(constant_scorer(0.3)("q", ["a", "b"]), [0.3, 0.3])


if __name__ == "__main__":
    unittest.main()
