"""Exceptions and warnings raised by NanoPrune."""


class NanoPruneWarning(UserWarning):
    """Something works in a degraded mode (e.g. no model weights were found)."""


class ModelNotFoundError(FileNotFoundError):
    """No usable model weights were found and a model was explicitly required."""


class HeadUnavailableError(RuntimeError):
    """The loaded backend does not provide the requested primitive (choice/score)."""


class TokenizerMismatchError(ValueError):
    """The tokenizer does not produce ids from the vocabulary the model was trained on."""


class OperationCancelled(Exception):
    """A long operation (indexing, embedding) was cancelled by the user."""
