"""NanoPrune: local System One relevance scoring and RAG pruning."""
from .core.tokenizer import NanoTokenizer
from .core.calibration import compute_ece, TemperatureScaler
from .engine.pruner import NanoPruner
from .engine.indexer import LocalDocumentIndexer, DocumentChunk
from .engine.search import LocalSearchEngine

__version__ = "0.4.0"

__all__ = [
    "NanoTokenizer",
    "TemperatureScaler",
    "compute_ece",
    "NanoPruner",
    "LocalDocumentIndexer",
    "DocumentChunk",
    "LocalSearchEngine",
    "__version__",
]
