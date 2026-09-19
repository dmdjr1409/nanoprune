import unittest
import torch
from nanoprune.core.model import NanoPruneModel
from nanoprune.core.tokenizer import NanoTokenizer

class TestNanoPrunePrimitives(unittest.TestCase):
    def setUp(self):
        self.tokenizer = NanoTokenizer()
        self.vocab_size = len(self.tokenizer.vocab)
        self.model = NanoPruneModel(
            vocab_size=self.vocab_size,
            d_model=128,
            n_heads=4,
            d_ff=512,
            n_layers=2,
            num_choices=4,
        )
        self.model.eval()

    def test_forward_compatibility(self):
        """Verify default forward() still returns (logits, probs) of shape (batch, 1)."""
        input_ids = torch.randint(0, self.vocab_size, (2, 64))
        mask = torch.ones((2, 64), dtype=torch.long)

        logits, probs = self.model(input_ids, mask)
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(probs.shape, (2, 1))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_forward_all_multi_primitives(self):
        """Verify forward_all returns relevance, choice, and score predictions with valid bounds."""
        input_ids = torch.randint(0, self.vocab_size, (3, 64))
        mask = torch.ones((3, 64), dtype=torch.long)

        res = self.model.forward_all(input_ids, mask)

        # 1. Relevance
        self.assertIn("relevance_probs", res)
        self.assertEqual(res["relevance_probs"].shape, (3, 1))
        self.assertTrue(torch.all(res["relevance_probs"] >= 0.0) and torch.all(res["relevance_probs"] <= 1.0))

        # 2. Choice
        self.assertIn("choice_probs", res)
        self.assertEqual(res["choice_probs"].shape, (3, 4))
        sums = res["choice_probs"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(3), atol=1e-5))

        # 3. Score
        self.assertIn("score", res)
        self.assertEqual(res["score"].shape, (3, 1))
        self.assertTrue(torch.all(res["score"] >= 0.0) and torch.all(res["score"] <= 4.0))

    def test_multi_task_gradients(self):
        """Verify gradients flow properly to all 3 heads during training."""
        self.model.train()
        input_ids = torch.randint(0, self.vocab_size, (2, 32))
        res = self.model.forward_all(input_ids)

        loss = res["relevance_probs"].sum() + res["choice_probs"].sum() + res["score"].sum()
        loss.backward()

        self.assertIsNotNone(self.model.relevance_head[0].weight.grad)
        self.assertIsNotNone(self.model.choice_head[0].weight.grad)
        self.assertIsNotNone(self.model.score_head[0].weight.grad)
        self.assertIsNotNone(self.model.token_embeddings.weight.grad)

if __name__ == "__main__":
    unittest.main()
