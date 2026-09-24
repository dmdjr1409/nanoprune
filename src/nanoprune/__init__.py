"""NanoPrune: local System One relevance scoring and RAG pruning."""
__version__ = "0.5.0"

from .core.tokenizer import NanoTokenizer
from .core.calibration import compute_ece, TemperatureScaler
from .engine.pruner import NanoPruner
from .engine.indexer import LocalDocumentIndexer, DocumentChunk
from .engine.search import LocalSearchEngine
from .engine.lexical import BM25Index
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
    "HeadUnavailableError",
    "ModelNotFoundError",
    "NanoPruneWarning",
    "TokenizerMismatchError",
    "__version__",
]
