"""NanoPrune: local relevance scoring, semantic search and RAG context pruning."""
__version__ = "0.7.0"

from .core.tokenizer import NanoTokenizer
from .core.calibration import compute_ece, TemperatureScaler
from .engine.pruner import NanoPruner
from .engine.indexer import LocalDocumentIndexer, DocumentChunk
from .engine.search import LocalSearchEngine
from .engine.lexical import BM25Index
from .engine.dense import SemanticScorer
from .errors import (
    HeadUnavailableError,
    ModelNotFoundError,
    NanoPruneWarning,
    TokenizerMismatchError,
)

__all__ = [
    "NanoTokenizer",
    "TemperatureScaler",
    "compute_ece",
    "NanoPruner",
    "LocalDocumentIndexer",
    "DocumentChunk",
    "LocalSearchEngine",
    "BM25Index",
    "SemanticScorer",
    "HeadUnavailableError",
    "ModelNotFoundError",
    "NanoPruneWarning",
    "TokenizerMismatchError",
    "__version__",
]
