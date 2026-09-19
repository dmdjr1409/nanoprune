import argparse
import sys
import os
from pathlib import Path

from .engine.pruner import NanoPruner
from .engine.indexer import LocalDocumentIndexer
from .engine.search import LocalSearchEngine

def main():
    parser = argparse.ArgumentParser(
        prog="nanoprune",
        description="NanoPrune: The 2.8MB System One Calibrated Decision & RAG Pruner.",
    )
    parser.add_argument("--version", action="version", version="nanoprune 0.1.0")

    subparsers = parser.add_subparsers(dest="command")

    # Command: app
    app_parser = subparsers.add_parser("app", help="Lancer l'interface locale confidentielle")
    app_parser.add_argument("--port", type=int, default=7860, help="Port du serveur local (défaut: 7860)")
    app_parser.add_argument("--dir", type=str, default=None, help="Dossier de documents à pré-charger")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Rechercher une preuve dans un dossier")
    search_parser.add_argument("query", type=str, help="Requête de recherche")
    search_parser.add_argument("--dir", type=str, default=".", help="Dossier à analyser (défaut: courant)")
    search_parser.add_argument("--threshold", type=float, default=0.50, help="Seuil de certitude minimale (0.0 à 1.0)")
    search_parser.add_argument("--top-k", type=int, default=5, help="Nombre maximal de résultats")

    # Command: prune
    prune_parser = subparsers.add_parser("prune", help="Élaguer des textes bruts par rapport à une requête")
    prune_parser.add_argument("query", type=str, help="Requête")
    prune_parser.add_argument("texts", nargs="+", help="Textes candidats à évaluer")
    prune_parser.add_argument("--threshold", type=float, default=0.70, help="Seuil d'élagage")

    args = parser.parse_args()

    if args.command == "app":
        from .app.server import run_app
        sample_dir = Path(args.dir) if args.dir else Path(__file__).parent.parent.parent / "sample_data" / "medical"
        run_app(port=args.port, sample_data_dir=sample_dir)

    elif args.command == "search":
        indexer = LocalDocumentIndexer()
        count = indexer.index_directory(args.dir)
        print(f"Indexé {count} fichiers ({len(indexer.chunks)} extraits) depuis '{args.dir}'.")
        engine = LocalSearchEngine(indexer=indexer)
        res = engine.search(args.query, top_k=args.top_k, threshold=args.threshold)
        print(f"\n⚡ Résultat en {res['latency_ms']} ms ({len(res['results'])} correspondances) :")
        for idx, r in enumerate(res["results"], 1):
            print(f"\n[{idx}] {r['file_name']} — Certitude: {r['confidence_pct']}")
            print(f"    Preuve: \"{r['highlight']}\"")

    elif args.command == "prune":
        pruner = NanoPruner.load(threshold=args.threshold)
        results = pruner.prune(args.query, args.texts)
        print(f"Conservés {len(results)}/{len(args.texts)} candidats (seuil >= {args.threshold}):")
        for text, score in results:
            print(f" - [{round(score * 100, 1)}%] {text[:80]}...")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
