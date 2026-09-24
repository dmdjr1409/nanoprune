#!/usr/bin/env python3
"""
Label the generated training pairs with a teacher for distillation.

Teachers: Laya (--teacher laya, needs the `laya` package) or a calibrated
semantic model (--teacher dense --dense DIR, e.g. the model installed by
`nanoprune download --dense multilingual-e5-large`). Both are usable commercially
(Laya: Apache 2.0, multilingual-e5: MIT).

Reads data/contrastive_{split}.jsonl and appends rows with a "teacher"
probability to data/contrastive_{split}.teacher.jsonl (resumable). Then train
with: scripts/train_contrastive.py --suffix .teacher --teacher-weight 0.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoprune.engine.cascade import DEFAULT_LAYA_MODEL  # noqa: E402
from nanoprune.training.teacher import label_file  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="splits to label (the test split is kept teacher-free by default)")
    parser.add_argument("--teacher", choices=["laya", "dense"], default="laya")
    parser.add_argument("--laya-model", default=DEFAULT_LAYA_MODEL)
    parser.add_argument("--dense", default="auto", help="semantic model directory (with --teacher dense)")
    parser.add_argument("--limit", type=int, default=None, help="label at most N new pairs per split")
    args = parser.parse_args(argv)

    agent = scorer = None
    if args.teacher == "laya":
        import laya
        agent = laya.load(args.laya_model)
        print(f"Teacher: {args.laya_model} on {getattr(agent, 'device', '?')}")
    else:
        from nanoprune.engine.dense import SemanticScorer
        scorer = SemanticScorer.load(args.dense)
        print(f"Teacher: {scorer.describe()['model_name']} (semantic)")

    for split in args.splits:
        src = Path(args.data_dir) / f"contrastive_{split}.jsonl"
        dst = Path(args.data_dir) / f"contrastive_{split}.teacher.jsonl"
        written = label_file(src, dst, agent=agent, scorer=scorer, limit=args.limit)
        print(f"{split}: {written} pairs labelled -> {dst}")


if __name__ == "__main__":
    main()
