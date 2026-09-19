"""
NanoPrune: The 2.8MB System One Calibrated Decision & RAG Pruner.
"""
from .core.tokenizer import NanoTokenizer
from .core.calibration import compute_ece, TemperatureScaler
from .engine.pruner import NanoPruner
from .engine.indexer import LocalDocumentIndexer, DocumentChunk
from .engine.search import LocalSearchEngine

__version__ = "0.1.0"

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
