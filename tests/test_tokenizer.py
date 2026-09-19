import unittest
from nanoprune.core.tokenizer import NanoTokenizer

class TestNanoTokenizer(unittest.TestCase):
    def test_tokenizer_encoding(self):
        tok = NanoTokenizer()
        text = "Patient allergique aux pénicillines"
        ids = tok.encode(text)
        self.assertGreater(len(ids), 0)
        self.assertNotIn(tok.cls_id, ids)

    def test_tokenizer_pair_encoding(self):
        tok = NanoTokenizer()
        query = "Allergie pénicilline"
        context = "Dossier patient Dupont. Allergie avérée aux pénicillines."
        input_ids, mask = tok.encode_pair(query, context, max_length=64)

        self.assertEqual(len(input_ids), 64)
        self.assertEqual(len(mask), 64)
        self.assertEqual(input_ids[0], tok.cls_id)
        self.assertIn(tok.sep_id, input_ids)
        self.assertGreater(sum(mask), 5)

if __name__ == "__main__":
    unittest.main()
