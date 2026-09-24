import argparse
import json
import sys
import warnings
from pathlib import Path

from . import __version__
from .engine.pruner import NanoPruner
from .errors import HeadUnavailableError, ModelNotFoundError, NanoPruneWarning, TokenizerMismatchError

REPO_SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_data" / "medical"


def _load_pruner(args, threshold: float = 0.70) -> NanoPruner:
    """Load the model requested on the command line and report what is running."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", NanoPruneWarning)
        pruner = NanoPruner.load(
            model_path=getattr(args, "model", None),
            threshold=threshold,
            strict=getattr(args, "strict", False),
            tokenizer_path=getattr(args, "tokenizer", None),
        )
    for warning in caught:
        if issubclass(warning.category, NanoPruneWarning):
            print(f"⚠️  {warning.message}", file=sys.stderr)
        else:
            warnings.showwarning(warning.message, warning.category, warning.filename, warning.lineno)
    return pruner


def _backend_line(pruner: NanoPruner) -> str:
    info = pruner.describe()
    if info["backend"] == "heuristic":
        return "Backend : heuristique par mots-clés (aucun modèle chargé)"
    return f"Backend : {info['backend']} — {info['model_name']} (tokenizer {info['tokenizer']})"


def _score_label(pruner: NanoPruner) -> str:
    return "Score mots-clés" if pruner.backend == "heuristic" else "Pertinence"


def cmd_app(args) -> int:
    from .app.server import run_app
    if args.dir:
        sample_dir = Path(args.dir)
    else:
        sample_dir = REPO_SAMPLE_DIR if REPO_SAMPLE_DIR.is_dir() else None
    run_app(port=args.port, sample_data_dir=sample_dir, pruner=_load_pruner(args))
    return 0


def cmd_search(args) -> int:
    from .engine.indexer import LocalDocumentIndexer
    from .engine.search import LocalSearchEngine

    pruner = _load_pruner(args)
    indexer = LocalDocumentIndexer()
    count = indexer.index_directory(args.dir)
    skipped = len(indexer.last_report.get("skipped", []))
    print(f"Indexé {count} fichiers ({len(indexer.chunks)} extraits) depuis '{args.dir}'"
          + (f", {skipped} ignorés." if skipped else "."))
    print(_backend_line(pruner))
    engine = LocalSearchEngine(indexer=indexer, pruner=pruner)
    res = engine.search(args.query, top_k=args.top_k, threshold=args.threshold)
    print(f"\n⚡ Résultat en {res['latency_ms']} ms ({len(res['results'])} correspondances) :")
    for idx, r in enumerate(res["results"], 1):
        lines = f"{r['line_start']}" if r["line_start"] == r["line_end"] else f"{r['line_start']}-{r['line_end']}"
        print(f"\n[{idx}] {r['file_name']}:{lines} — {_score_label(pruner)} : {r['score_pct']}")
        print(f"    Extrait : \"{r['highlight']}\"")
    return 0


def cmd_prune(args) -> int:
    pruner = _load_pruner(args, threshold=args.threshold)
    print(_backend_line(pruner))
    results = pruner.prune(args.query, args.texts)
    print(f"Conservés {len(results)}/{len(args.texts)} candidats (seuil >= {args.threshold}) :")
    for text, score in results:
        shown = text if len(text) <= 80 else text[:77] + "..."
        print(f" - [{round(score * 100, 1)}%] {shown}")
    return 0


def cmd_choice(args) -> int:
    pruner = _load_pruner(args)
    try:
        idx, opt, conf = pruner.choice(args.text, args.options)
    except HeadUnavailableError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    print(f"Catégorie sélectionnée : {opt} (index {idx})")
    print(f"Probabilité (tête choice) : {conf * 100:.1f}%")
    return 0


def cmd_score(args) -> int:
    pruner = _load_pruner(args)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NanoPruneWarning)
            val = pruner.score_rubric(args.text)
    except HeadUnavailableError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    print(f"Score (expérimental, tête non entraînée en v0.4) : {val:.2f} / 4.0")
    return 0


def cmd_info(args) -> int:
    from .core.weights import cache_dir, discover_candidates, nanoprune_home, weight_dirs

    pruner = _load_pruner(args)
    print(f"nanoprune {__version__}")
    print(_backend_line(pruner))
    print(json.dumps(pruner.describe(), indent=2, ensure_ascii=False))
    print("\nRépertoires de poids explorés (dans l'ordre) :")
    for directory in weight_dirs():
        print(f"  {'✓' if directory.is_dir() else '·'} {directory}")
    specs, notes = discover_candidates()
    for spec in specs:
        print(f"  modèle trouvé : {spec.path} (tokenizer {spec.tokenizer_kind})")
    for note in notes:
        print(f"  note : {note}")
    print(f"\nNANOPRUNE_HOME={nanoprune_home() or '(non défini)'} ; cache={cache_dir()}")
    return 0


def cmd_download(args) -> int:
    from .download import DownloadError, download_release
    try:
        download_release(tag=args.tag, dest=args.dest, repo=args.repo, force=args.force)
    except DownloadError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    print("\nVérifiez le modèle avec : nanoprune info  puis  nanoprune eval")
    return 0


def cmd_eval(args) -> int:
    from .evaluation import format_categories, format_table, run_standard_evaluation

    pruner = None if args.no_model else _load_pruner(args)
    laya_agent = None
    if args.laya:
        try:
            import laya
        except ImportError:
            print("Erreur : le paquet 'laya' n'est pas installé.", file=sys.stderr)
            return 1
        from .engine.cascade import DEFAULT_LAYA_MODEL
        laya_agent = laya.load(args.laya_model or DEFAULT_LAYA_MODEL)

    report = run_standard_evaluation(
        suites=args.suite or ["dev", "heldout"],
        pruner=pruner,
        laya_agent=laya_agent,
        threshold=args.threshold,
        n_bootstrap=args.bootstrap,
    )
    for suite, results in report.items():
        print(f"\n## Suite : {suite} (seuil {args.threshold})\n")
        print(format_table(results))
        print()
        print(format_categories(results))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRésultats détaillés : {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoprune",
        description="NanoPrune: local System One relevance scoring and RAG pruning.",
    )
    parser.add_argument("--version", action="version", version=f"nanoprune {__version__}")

    model_args = argparse.ArgumentParser(add_help=False)
    model_args.add_argument("--model", type=str, default=None,
                            help="Fichier de poids (.onnx ou .pt) ; par défaut recherche automatique")
    model_args.add_argument("--tokenizer", type=str, default=None, help="Fichier tokenizer WordPiece (.json)")
    model_args.add_argument("--strict", action="store_true",
                            help="Échouer si aucun modèle n'est trouvé au lieu d'utiliser l'heuristique")

    subparsers = parser.add_subparsers(dest="command")

    app_parser = subparsers.add_parser("app", parents=[model_args], help="Lancer l'interface locale confidentielle")
    app_parser.add_argument("--port", type=int, default=7860, help="Port du serveur local (défaut : 7860)")
    app_parser.add_argument("--dir", type=str, default=None, help="Dossier de documents à pré-charger")
    app_parser.set_defaults(func=cmd_app)

    search_parser = subparsers.add_parser("search", parents=[model_args], help="Rechercher un passage dans un dossier")
    search_parser.add_argument("query", type=str, help="Requête de recherche")
    search_parser.add_argument("--dir", type=str, default=".", help="Dossier à analyser (défaut : courant)")
    search_parser.add_argument("--threshold", type=float, default=0.50, help="Score minimal (0.0 à 1.0)")
    search_parser.add_argument("--top-k", type=int, default=5, help="Nombre maximal de résultats")
    search_parser.set_defaults(func=cmd_search)

    prune_parser = subparsers.add_parser("prune", parents=[model_args], help="Élaguer des textes par rapport à une requête")
    prune_parser.add_argument("query", type=str, help="Requête")
    prune_parser.add_argument("texts", nargs="+", help="Textes candidats à évaluer")
    prune_parser.add_argument("--threshold", type=float, default=0.70, help="Seuil d'élagage")
    prune_parser.set_defaults(func=cmd_prune)

    choice_parser = subparsers.add_parser("choice", parents=[model_args],
                                          help="Catégoriser un texte (tête choice, modèle PyTorch requis)")
    choice_parser.add_argument("text", type=str, help="Texte à analyser")
    choice_parser.add_argument("--options", nargs="+",
                               default=["human_rights", "business_tax", "public_admin", "civil_family"],
                               help="Catégories (associées par position aux catégories d'entraînement)")
    choice_parser.set_defaults(func=cmd_choice)

    score_parser = subparsers.add_parser("score", parents=[model_args], help="Score expérimental (0 à 4)")
    score_parser.add_argument("text", type=str, help="Texte à noter")
    score_parser.set_defaults(func=cmd_score)

    info_parser = subparsers.add_parser("info", parents=[model_args], help="Afficher le modèle chargé et où il a été trouvé")
    info_parser.set_defaults(func=cmd_info)

    download_parser = subparsers.add_parser("download", help="Télécharger des poids publiés (release GitHub)")
    download_parser.add_argument("--tag", type=str, default=None, help="Tag de release (défaut : la plus récente)")
    download_parser.add_argument("--dest", type=str, default=None, help="Dossier cible (défaut : ~/.cache/nanoprune/weights)")
    download_parser.add_argument("--repo", type=str, default=None, help="Dépôt owner/nom (défaut : dmdjr1409/nanoprune)")
    download_parser.add_argument("--force", action="store_true", help="Retélécharger même si le fichier est présent")
    download_parser.set_defaults(func=cmd_download)

    eval_parser = subparsers.add_parser("eval", parents=[model_args],
                                        help="Évaluer le modèle et les baselines sur les suites de test")
    eval_parser.add_argument("--suite", action="append",
                             help="'dev', 'heldout' ou chemin JSONL (répétable ; défaut : dev et heldout)")
    eval_parser.add_argument("--no-model", action="store_true", help="N'évaluer que les baselines lexicales")
    eval_parser.add_argument("--laya", action="store_true", help="Évaluer aussi Laya et la cascade (paquet laya requis)")
    eval_parser.add_argument("--laya-model", type=str, default=None, help="Checkpoint Laya à charger")
    eval_parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de décision (défaut : 0.5)")
    eval_parser.add_argument("--bootstrap", type=int, default=1000, help="Rééchantillonnages bootstrap (0 = aucun)")
    eval_parser.add_argument("--json", type=str, default=None, help="Écrire les résultats détaillés en JSON")
    eval_parser.set_defaults(func=cmd_eval)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args) or 0
    except (ModelNotFoundError, TokenizerMismatchError, FileNotFoundError, ImportError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
