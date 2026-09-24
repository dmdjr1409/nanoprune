#!/usr/bin/env python3
"""
Prepare model files for a GitHub release that `nanoprune download` can install.

Takes a PyTorch checkpoint and the tokenizer it was trained with, then writes
to --out: the checkpoint, a self-contained ONNX export (all heads), the
tokenizer and a manifest (architecture, tokenizer SHA-256, temperature). It
checks that the ONNX and PyTorch files give the same scores and prints the
`gh release create` command to publish them.

Example for the existing v0.4 weights:

    python scripts/package_release.py --pt weights/nanoprune-v0.4.pt \
        --tokenizer data/tokenizer_legal.json --name nanoprune-v0.4 --tag v0.4.0
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoprune.core.weights import write_manifest  # noqa: E402
from nanoprune.engine.pruner import DEFAULT_CHOICE_LABELS, NanoPruner  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt", required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--tokenizer", default=None, help="WordPiece JSON (omit for char-level models)")
    parser.add_argument("--name", required=True, help="model name, e.g. nanoprune-v0.4")
    parser.add_argument("--tag", default=None, help="release tag for the printed gh command")
    parser.add_argument("--out", default="dist/release")
    parser.add_argument("--score-trained", action="store_true", help="declare the score head as trained")
    args = parser.parse_args(argv)

    from nanoprune.core.export import export_to_onnx

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pt_src = Path(args.pt)
    pruner = NanoPruner(model_path=pt_src, tokenizer_path=args.tokenizer)
    print(f"Loaded {pt_src} ({pruner.describe()['parameters']:,} parameters, tokenizer {pruner.tokenizer.kind})")

    pt_dst = out / f"{args.name}.pt"
    shutil.copyfile(pt_src, pt_dst)
    onnx_dst = out / f"{args.name}.onnx"
    export_to_onnx(pruner.torch_model, onnx_dst, seq_len=pruner.max_length, heads="all")

    tokenizer_dst = None
    if args.tokenizer:
        tokenizer_dst = out / f"{args.name}-tokenizer.json"
        shutil.copyfile(args.tokenizer, tokenizer_dst)

    manifest = write_manifest(
        out / f"{args.name}.json",
        name=args.name,
        architecture=pruner.torch_model.config(),
        files={"pt": pt_dst.name, "onnx": onnx_dst.name},
        tokenizer_file=tokenizer_dst,
        temperature=float(pruner.torch_model.temperature.item()),
        extra={
            "choice_labels": DEFAULT_CHOICE_LABELS,
            "heads": {"relevance": True, "choice": True, "score": args.score_trained},
        },
    )

    # The packaged files must load through the manifest and agree with each other.
    query = "délai de préavis en cas de démission"
    docs = ["Le salarié démissionnaire doit effectuer un préavis.", "Les chiens doivent être tenus en laisse."]
    from_pt = NanoPruner(model_path=pt_dst).score(query, docs)
    from_onnx = NanoPruner(model_path=onnx_dst).score(query, docs)
    drift = max(abs(a - b) for a, b in zip(from_pt, from_onnx))
    print(f"PyTorch vs ONNX max score difference: {drift:.2e}")
    if drift > 1e-4:
        raise SystemExit("ONNX export does not match the checkpoint")

    files = sorted(p for p in out.iterdir() if p.name.startswith(args.name))
    print(f"\nRelease files in {out}:")
    for path in files:
        print(f"  {path.name:40s} {path.stat().st_size / 1024:9.0f} KiB  sha256 {sha256(path)[:16]}…")
    print(f"\nManifest tokenizer: {manifest['tokenizer']}")
    tag = args.tag or "vX.Y.Z"
    print("\nPublish with:\n  gh release create " + tag + " " + " ".join(str(p) for p in files)
          + f" --title '{args.name}' --notes 'Model weights, tokenizer and manifest.'")


if __name__ == "__main__":
    main()
