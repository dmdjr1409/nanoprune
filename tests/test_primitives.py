import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TestNanoPrunePrimitives(unittest.TestCase):
    def setUp(self):
        from nanoprune.core.model import NanoPruneModel
        from nanoprune.core.tokenizer import NanoTokenizer

        self.tokenizer = NanoTokenizer.char_level()
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
        import torch
        input_ids = torch.randint(0, self.vocab_size, (2, 64))
        mask = torch.ones((2, 64), dtype=torch.long)

        logits, probs = self.model(input_ids, mask)
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(probs.shape, (2, 1))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_forward_all_multi_primitives(self):
        """Verify forward_all returns relevance, choice, and score predictions with valid bounds."""
        import torch
        input_ids = torch.randint(0, self.vocab_size, (3, 64))
        mask = torch.ones((3, 64), dtype=torch.long)

        res = self.model.forward_all(input_ids, mask)

        self.assertEqual(res["relevance_probs"].shape, (3, 1))
        self.assertTrue(torch.all(res["relevance_probs"] >= 0.0) and torch.all(res["relevance_probs"] <= 1.0))
        self.assertEqual(res["choice_probs"].shape, (3, 4))
        self.assertTrue(torch.allclose(res["choice_probs"].sum(dim=-1), torch.ones(3), atol=1e-5))
        self.assertEqual(res["score"].shape, (3, 1))
        self.assertTrue(torch.all(res["score"] >= 0.0) and torch.all(res["score"] <= 4.0))

    def test_multi_task_gradients(self):
        """Verify gradients flow properly to all 3 heads during training."""
        import torch
        self.model.train()
        input_ids = torch.randint(0, self.vocab_size, (2, 32))
        res = self.model.forward_all(input_ids)

        loss = res["relevance_probs"].sum() + res["choice_probs"].sum() + res["score"].sum()
        loss.backward()

        self.assertIsNotNone(self.model.relevance_head[0].weight.grad)
        self.assertIsNotNone(self.model.choice_head[0].weight.grad)
        self.assertIsNotNone(self.model.score_head[0].weight.grad)
        self.assertIsNotNone(self.model.token_embeddings.weight.grad)

    def test_padding_does_not_change_scores(self):
        """Dynamic padding (used at inference) must match padding to the full length."""
        import torch
        self.model.eval()
        texts = ["Allergie sévère aux pénicillines.", "Ordonnance de lunettes pour la lecture de près."]
        short_ids, short_mask = self.tokenizer.encode_batch("allergie pénicilline", texts)
        full_ids, full_mask = self.tokenizer.encode_batch("allergie pénicilline", texts, pad_to=256)
        with torch.no_grad():
            _, short = self.model(torch.tensor(short_ids), torch.tensor(short_mask))
            _, full = self.model(torch.tensor(full_ids), torch.tensor(full_mask))
        self.assertTrue(torch.allclose(short, full, atol=1e-5))

    def test_architecture_is_recovered_from_weights(self):
        from nanoprune.core.model import NanoPruneModel, infer_architecture
        legacy = NanoPruneModel(vocab_size=321, d_model=128, n_heads=4, d_ff=512, n_layers=2, head_hidden=64)
        state = legacy.state_dict()
        arch = infer_architecture(state)
        self.assertEqual((arch["n_layers"], arch["d_ff"], arch["head_hidden"], arch["n_heads"]), (2, 512, 64, 4))
        restored = NanoPruneModel.from_state_dict(state)
        self.assertEqual(restored.count_parameters(), legacy.count_parameters())

    def test_legacy_checkpoint_without_extra_heads(self):
        from nanoprune.core.model import NanoPruneModel
        model = NanoPruneModel(vocab_size=321, d_model=64, n_heads=4, d_ff=128, n_layers=1, head_hidden=64)
        state = {k: v for k, v in model.state_dict().items()
                 if not k.startswith(("relevance_head.", "choice_head.", "score_head."))}
        restored = NanoPruneModel.from_state_dict(state)
        self.assertEqual(restored.loaded_heads, {"relevance": True, "choice": False, "score": False})

    def test_incompatible_checkpoint_is_rejected(self):
        from nanoprune.core.model import NanoPruneModel
        state = dict(NanoPruneModel(vocab_size=50, d_model=64, n_heads=4, d_ff=128, n_layers=1).state_dict())
        state["unexpected.weight"] = state["layer_norm.weight"]
        with self.assertRaises(RuntimeError):
            NanoPruneModel.from_state_dict(state)


if __name__ == "__main__":
    unittest.main()
