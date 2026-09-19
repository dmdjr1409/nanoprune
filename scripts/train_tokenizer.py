#!/usr/bin/env python3
"""
Train a fast WordPiece/BPE tokenizer on legal_raw.tsv (vocab size = 8192).
Saves tokenizer to data/tokenizer_legal.json.
"""
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

def train_tokenizer(tsv_path: str = "data/legal_raw.tsv", output_json: str = "data/tokenizer_legal.json", vocab_size: int = 8192):
    print("=" * 60)
    print(f"🔤 Training WordPiece Tokenizer on {tsv_path} (vocab={vocab_size})")
    print("=" * 60)

    # Read raw texts from tsv
    texts = []
    with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t") if "\t" in line else line.strip().split(r"\t")
            if len(parts) >= 3:
                texts.append(parts[1] + " " + parts[2])
            elif len(parts) == 2:
                texts.append(parts[1])

    print(f"Loaded {len(texts)} documents for tokenizer training.")

    # Initialize WordPiece tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.WordPiece()

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"✅ Tokenizer saved to {out_path} (actual vocab: {tokenizer.get_vocab_size()})")

    # Quick test
    sample = "Rupture conventionnelle et indemnité de départ pour salarié."
    encoded = tokenizer.encode(sample)
    print(f"Sample: '{sample}'")
    print(f"Tokens: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")

if __name__ == "__main__":
    train_tokenizer()
