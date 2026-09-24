#!/usr/bin/env python3
"""
Train NanoPrune on the pairs built by generate_contrastive_dataset.py.

- model selection on the validation split only (AUC, then loss);
- temperature scaling fitted on the validation split;
- self-contained ONNX export (all heads) and a manifest tying the weights to
  their tokenizer (read by NanoPruner.load());
- test split and bundled suites reported at the end, never used for selection.

Defaults reproduce the v0.4 architecture (4 layers, d_model 256, 5.45M parameters).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoprune.training.trainer import TrainConfig, train  # noqa: E402


def main(argv=None):
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=defaults.data_dir)
    parser.add_argument("--out-dir", default=defaults.out_dir)
    parser.add_argument("--name", default=defaults.name)
    parser.add_argument("--tokenizer", default=defaults.tokenizer_path,
                        help="WordPiece JSON; use '' for the built-in char vocabulary")
    parser.add_argument("--suffix", default="", help="read contrastive_<split><suffix>.jsonl (e.g. .teacher)")
    parser.add_argument("--teacher-weight", type=float, default=defaults.teacher_weight,
                        help="weight of the teacher probability in the relevance target (0 = hard labels)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--max-len", type=int, default=defaults.max_len)
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--heads", type=int, default=defaults.n_heads)
    parser.add_argument("--ffn", type=int, default=defaults.d_ff)
    parser.add_argument("--layers", type=int, default=defaults.n_layers)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--no-suites", action="store_true", help="skip the dev/heldout suite report")
    args = parser.parse_args(argv)

    config = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        name=args.name,
        tokenizer_path=args.tokenizer or None,
        train_file=f"contrastive_train{args.suffix}.jsonl",
        val_file=f"contrastive_val{args.suffix}.jsonl",
        test_file="contrastive_test.jsonl",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        seed=args.seed,
        device=args.device,
        teacher_weight=args.teacher_weight,
        d_model=args.d_model,
        n_heads=args.heads,
        d_ff=args.ffn,
        n_layers=args.layers,
        export_onnx=not args.no_onnx,
        eval_suites=() if args.no_suites else ("dev", "heldout"),
    )
    train(config)


if __name__ == "__main__":
    main()
