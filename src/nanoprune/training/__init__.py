"""Training utilities: leakage-free pair generation, teacher labelling, training loop.

The heavy dependencies (``torch``, ``tokenizers``) are imported lazily by the
functions that need them.
"""
from .data import (
    CATEGORY_LABELS,
    IGNORE_CATEGORY,
    Record,
    build_pairs,
    read_jsonl,
    read_legal_tsv,
    split_of,
    weak_category,
    write_jsonl,
)

__all__ = [
    "CATEGORY_LABELS",
    "IGNORE_CATEGORY",
    "Record",
    "build_pairs",
    "read_jsonl",
    "read_legal_tsv",
    "split_of",
    "weak_category",
    "write_jsonl",
]
