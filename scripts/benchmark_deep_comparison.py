#!/usr/bin/env python3
"""
NanoPrune vs Laya vs hybrid cascade, with baselines and ablations.

Equivalent to `nanoprune eval --laya`. Evaluates on the development suite (the
50 cases the v0.4 cascade was tuned on) and on the held-out suite, and runs the
cascade with its tier-1 score replaced by a constant: if the real cascade does
not beat these ablations, the NanoPrune score adds nothing to it.

Requires the `laya` package and a NanoPrune model (see the README).
Writes data/benchmark_results.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoprune.engine.cascade import DEFAULT_LAYA_MODEL  # noqa: E402
from nanoprune.engine.pruner import NanoPruner  # noqa: E402
from nanoprune.evaluation import format_categories, format_table, run_standard_evaluation  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="NanoPrune weights (default: discovered)")
    parser.add_argument("--laya-model", default=DEFAULT_LAYA_MODEL)
    parser.add_argument("--suite", action="append", help="dev, heldout or a JSONL path (default: both)")
    parser.add_argument("--out", default="data/benchmark_results.json")
    args = parser.parse_args(argv)

    import laya

    pruner = NanoPruner.load(model_path=args.model, strict=True)
    agent = laya.load(args.laya_model)
    report = run_standard_evaluation(suites=args.suite or ["dev", "heldout"], pruner=pruner, laya_agent=agent)
    for suite, results in report.items():
        print(f"\n## {suite}\n")
        print(format_table(results))
        print()
        print(format_categories(results))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": pruner.describe(), "laya_model": args.laya_model, "results": report},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
