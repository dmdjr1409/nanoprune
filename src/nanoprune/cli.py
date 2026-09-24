import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from . import __version__
from .engine.pruner import NanoPruner
from .errors import HeadUnavailableError, ModelNotFoundError, NanoPruneWarning, TokenizerMismatchError

REPO_SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_data" / "medical"

QUICK_START = """
Démarrage rapide :
  nanoprune app                                   interface locale dans le navigateur
  nanoprune app --dir ~/Documents/Dossiers        ... sur un dossier précis
  nanoprune search "délai de préavis" --dir ~/Documents
  nanoprune download --dense multilingual-e5-large
                                                  recherche sémantique (550 Mo, une seule fois)
  nanoprune info                                  modèle utilisé et où il a été trouvé

Tout tourne en local : aucune donnée ne quitte l'ordinateur.
"""


def _dense_requested(args):
    """The semantic model to use: --dense, $NANOPRUNE_DENSE_MODEL or an installed one (unless --model/--no-dense)."""
    dense = getattr(args, "dense", None)
    if dense is not None or getattr(args, "model", None) is not None or getattr(args, "no_dense", False):
        return dense
    from .engine.dense import installed_dense_models
    if os.environ.get("NANOPRUNE_DENSE_MODEL") or installed_dense_models():
        return "auto"
    return None


def _load_pruner(args, threshold: float = 0.70, disk_cache: bool = False):
    """Load the scorer requested on the command line and report what is running.

    A semantic model (``--dense``, or one installed with ``nanoprune download
    --dense``) takes precedence; ``--model`` forces a NanoPrune checkpoint.
    ``disk_cache`` keeps passage embeddings on disk (folder searches).
    """
    if getattr(args, "dense", None) is not None and getattr(args, "model", None) is not None:
        raise ValueError("Use either --dense (semantic model) or --model (NanoPrune checkpoint), not both.")
    dense = _dense_requested(args)
    if dense is not None:
        from .engine.dense import SemanticScorer
        return SemanticScorer.load(dense, threshold=threshold, disk_cache=disk_cache)
    return _load_nanoprune(args, threshold)


def _load_nanoprune(args, threshold: float = 0.70) -> NanoPruner:
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


def _backend_line(pruner) -> str:
    info = pruner.describe()
    if info["backend"] == "heuristic":
        return "Backend : heuristique par mots-clés (aucun modèle chargé)"
    if info["backend"] == "semantic":
        fitted = (info.get("calibration") or {}).get("fitted_on", "?")
        return f"Backend : sémantique — {info['model_name']} (calibré sur la suite {fitted})"
    return f"Backend : {info['backend']} — {info['model_name']} (tokenizer {info['tokenizer']})"


def _score_label(pruner) -> str:
    return "Score mots-clés" if pruner.backend == "heuristic" else "Pertinence"


def cmd_app(args) -> int:
    from .app.server import run_app
    if args.dir:
        sample_dir = Path(args.dir)
    else:
        sample_dir = REPO_SAMPLE_DIR if REPO_SAMPLE_DIR.is_dir() else None
    run_app(port=args.port, sample_data_dir=sample_dir, pruner=_load_pruner(args, disk_cache=True),
            open_browser=not args.no_browser)
    return 0


class _Progress:
    """One-line progress on stderr, only when it is a terminal."""

    def __init__(self, label: str):
        self.label = label
        self.enabled = sys.stderr.isatty()
        self.shown = False

    def __call__(self, done: int, total: int) -> None:
        if self.enabled and total:
            sys.stderr.write(f"\r{self.label} : {done}/{total}")
            sys.stderr.flush()
            self.shown = True

    def close(self) -> None:
        if self.shown:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()


def cmd_search(args) -> int:
    from .engine.indexer import LocalDocumentIndexer
    from .engine.search import LocalSearchEngine

    pruner = _load_pruner(args, disk_cache=True)
    indexer = LocalDocumentIndexer()
    progress = _Progress("Lecture des fichiers")
    try:
        count = indexer.index_directory(args.dir, progress=progress)
    finally:
        progress.close()
    engine = LocalSearchEngine(indexer=indexer, pruner=pruner)
    progress = _Progress("Analyse sémantique des extraits")
    try:
        engine.prepare(progress=progress)
    finally:
        progress.close()
    res = engine.search(args.query, top_k=args.top_k, threshold=args.threshold)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    skipped = len(indexer.last_report.get("skipped", []))
    print(f"Indexé {count} fichiers ({len(indexer.chunks)} extraits) depuis '{args.dir}'"
          + (f", {skipped} ignorés." if skipped else "."))
    print(_backend_line(pruner))
    more = f" sur {res['matches_total']}" if res["more_available"] else ""
    print(f"\n⚡ Résultat en {res['latency_ms']} ms ({len(res['results'])}{more} correspondances) :")
    for idx, r in enumerate(res["results"], 1):
        lines = f"{r['line_start']}" if r["line_start"] == r["line_end"] else f"{r['line_start']}-{r['line_end']}"
        print(f"\n[{idx}] {r['rel_path']}:{lines} — {_score_label(pruner)} : {r['score_pct']}")
        print(f"    Extrait : \"{r['highlight']}\"")
    if not res["results"]:
        if res["near_misses"]:
            best = res["near_misses"][0]
            print(f"\nAucun passage au-dessus du seuil {args.threshold}. Le plus proche : "
                  f"{best['rel_path']}:{best['line_start']} ({best['score_pct']}). Essayez --threshold "
                  f"{max(0.05, round(best['score'] - 0.05, 2))}.")
        elif pruner.backend == "heuristic":
            print("\nAucun passage trouvé. En mode mots-clés, utilisez des mots présents dans les documents, ou "
                  "installez la recherche sémantique : nanoprune download --dense multilingual-e5-large")
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
    from .engine.dense import dense_models_dir, installed_dense_models
    print(f"\nModèles sémantiques installés ({dense_models_dir()}) :")
    for path in installed_dense_models():
        print(f"  {path}")
    if not installed_dense_models():
        print("  aucun (nanoprune download --dense multilingual-e5-large)")
    embeddings = cache_dir() / "embeddings"
    stores = sorted(embeddings.glob("*.sqlite3")) if embeddings.is_dir() else []
    size = sum(path.stat().st_size for path in stores)
    print(f"\nCache des vecteurs ({embeddings}) : {len(stores)} fichier(s), {size / 1e6:.1f} Mo"
          " — NANOPRUNE_DISK_CACHE=0 pour le désactiver, supprimez le dossier pour l'effacer")
    print(f"\nNANOPRUNE_HOME={nanoprune_home() or '(non défini)'} ; cache={cache_dir()}")
    return 0


def cmd_download(args) -> int:
    from .download import DownloadError, download_dense_model, download_release
    try:
        if args.dense:
            download_dense_model(args.dense, int8=not args.no_int8, keep_fp32=args.keep_fp32, dest=args.dest)
        else:
            download_release(tag=args.tag, dest=args.dest, repo=args.repo, force=args.force)
    except DownloadError as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    print("\nVérifiez le modèle avec : nanoprune info  puis  nanoprune eval")
    return 0


def cmd_calibrate(args) -> int:
    from .engine.dense import SemanticScorer
    scorer = SemanticScorer(args.dense)
    calibration = scorer.calibrate(args.suite)
    print(f"Calibration de {args.dense} sur la suite '{args.suite}' : a={calibration['a']:.3f}, b={calibration['b']:.3f}")
    print("Évaluez ensuite sur l'autre suite : nanoprune eval --dense " + str(args.dense))
    return 0


def cmd_eval(args) -> int:
    from .evaluation import format_categories, format_table, run_standard_evaluation

    models = []
    if not args.no_model:
        models.append(_load_nanoprune(args))
        dense = _dense_requested(args)
        if dense is not None:
            from .engine.dense import SemanticScorer
            models.append(SemanticScorer.load(dense))
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
        models=models,
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
        description="NanoPrune : recherche locale et élagage de contexte RAG (score de pertinence, sans LLM).",
        epilog=QUICK_START,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"nanoprune {__version__}")

    model_args = argparse.ArgumentParser(add_help=False)
    model_args.add_argument("--model", type=str, default=None,
                            help="Fichier de poids (.onnx ou .pt) ; par défaut recherche automatique")
    model_args.add_argument("--tokenizer", type=str, default=None, help="Fichier tokenizer WordPiece (.json)")
    model_args.add_argument("--strict", action="store_true",
                            help="Échouer si aucun modèle n'est trouvé au lieu d'utiliser l'heuristique")
    model_args.add_argument("--dense", type=str, default=None,
                            help="Modèle sémantique (dossier ONNX, ou 'auto') ; utilisé par défaut s'il est installé")
    model_args.add_argument("--no-dense", action="store_true", help="Ne pas utiliser le modèle sémantique installé")

    subparsers = parser.add_subparsers(dest="command")

    app_parser = subparsers.add_parser("app", parents=[model_args], help="Lancer l'interface locale confidentielle")
    app_parser.add_argument("--port", type=int, default=7860, help="Port du serveur local (défaut : 7860)")
    app_parser.add_argument("--dir", type=str, default=None, help="Dossier de documents à pré-charger")
    app_parser.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir le navigateur automatiquement")
    app_parser.set_defaults(func=cmd_app)

    search_parser = subparsers.add_parser("search", parents=[model_args], help="Rechercher un passage dans un dossier")
    search_parser.add_argument("query", type=str, help="Requête de recherche")
    search_parser.add_argument("--dir", type=str, default=".", help="Dossier à analyser (défaut : courant)")
    search_parser.add_argument("--threshold", type=float, default=0.50, help="Score minimal (0.0 à 1.0)")
    search_parser.add_argument("--top-k", type=int, default=5, help="Nombre maximal de résultats")
    search_parser.add_argument("--json", action="store_true", help="Afficher la réponse complète en JSON")
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
    download_parser.add_argument("--dense", type=str, default=None,
                                 help="Installer un modèle sémantique (ex : multilingual-e5-large) au lieu d'une release")
    download_parser.add_argument("--no-int8", action="store_true", help="Garder le modèle sémantique en float32 (4x plus gros)")
    download_parser.add_argument("--keep-fp32", action="store_true", help="Conserver aussi la version float32")
    download_parser.set_defaults(func=cmd_download)

    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrer un modèle sémantique sur une suite (défaut : dev)")
    calibrate_parser.add_argument("--dense", required=True, help="Dossier du modèle sémantique (model.onnx + tokenizer.json)")
    calibrate_parser.add_argument("--suite", default="dev", help="Suite de calibration (ne jamais utiliser 'heldout')")
    calibrate_parser.set_defaults(func=cmd_calibrate)

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
