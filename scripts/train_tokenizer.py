#!/usr/bin/env python3
"""
Train the WordPiece tokenizer on legal_raw.tsv (default vocabulary: 8192).

The text is normalised exactly like NanoTokenizer.encode() (lowercase, single
spaces) before training, so that no vocabulary slot is wasted on forms the
model never sees.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoprune.core.tokenizer import NanoTokenizer  # noqa: E402
from nanoprune.training.data import read_legal_tsv, train_wordpiece  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", default="data/legal_raw.tsv")
    parser.add_argument("--out", default="data/tokenizer_legal.json")
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args(argv)

    records = read_legal_tsv(args.tsv, max_content_chars=100_000)
    print(f"Training WordPiece (vocab={args.vocab_size}) on {len(records)} documents from {args.tsv}")
    size = train_wordpiece((f"{r.title} {r.content}" for r in records), args.out, vocab_size=args.vocab_size)
    print(f"Tokenizer saved to {args.out} (actual vocab: {size})")

    tok = NanoTokenizer.from_file(args.out)
    sample = "Rupture conventionnelle et indemnité de départ pour salarié."
    print(f"Sample: {sample!r}")
    print(f"Token IDs: {tok.encode(sample)}")


if __name__ == "__main__":
    main()
