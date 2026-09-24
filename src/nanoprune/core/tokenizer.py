import hashlib
import re
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Sequence, Tuple

from ..errors import NanoPruneWarning

# WordPiece vocabulary used by the v0.4 model (trained by scripts/train_tokenizer.py).
DEFAULT_TOKENIZER_NAME = "tokenizer_legal.json"
_REPO_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class NanoTokenizer:
    """
    Compact subword tokenizer for NanoPrune.

    Two modes exist:
    - ``wordpiece``: the 8192-token WordPiece vocabulary stored in a
      ``tokenizers`` JSON file (used by the v0.4 model; needs the
      ``tokenizers`` package);
    - ``char``: the built-in character/prefix vocabulary (321 tokens) used by
      the v0.1-v0.3 checkpoints and by the keyword heuristic.
    """
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        json_path: Optional[str] = None,
        auto_load: bool = True,
    ):
        """
        Args:
            vocab: explicit char-level vocabulary.
            json_path: WordPiece JSON file. Loading it is mandatory once given:
                a missing file or a missing ``tokenizers`` package raises
                instead of silently producing ids from another vocabulary.
            auto_load: when no ``json_path`` is given, load the repository's
                ``data/tokenizer_legal.json`` if it exists.
        """
        self.fast_tokenizer = None
        self.json_path: Optional[Path] = None

        explicit = json_path is not None
        if json_path is None and auto_load and vocab is None:
            default_json = _REPO_DATA_DIR / DEFAULT_TOKENIZER_NAME
            if default_json.exists():
                json_path = str(default_json)

        if json_path is not None:
            self._load_wordpiece(Path(json_path), required=explicit)

        if self.fast_tokenizer is None:
            if vocab is not None:
                self.vocab = vocab
                self.inv_vocab = {v: k for k, v in vocab.items()}
            else:
                self.vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
                self.inv_vocab = {idx: tok for idx, tok in enumerate(self.SPECIAL_TOKENS)}
                self._init_default_subwords()

        self.pad_id = self.vocab.get(self.PAD_TOKEN, 0)
        self.unk_id = self.vocab.get(self.UNK_TOKEN, 1)
        self.cls_id = self.vocab.get(self.CLS_TOKEN, 2)
        self.sep_id = self.vocab.get(self.SEP_TOKEN, 3)

    @classmethod
    def char_level(cls) -> "NanoTokenizer":
        """The built-in char/prefix tokenizer, never the WordPiece file."""
        return cls(auto_load=False)

    @classmethod
    def from_file(cls, json_path) -> "NanoTokenizer":
        """A WordPiece tokenizer loaded from ``json_path`` (raises on failure)."""
        return cls(json_path=str(json_path))

    def _load_wordpiece(self, path: Path, required: bool) -> None:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Tokenizer file not found: {path}")
            return
        try:
            from tokenizers import Tokenizer
        except ImportError:
            message = (
                f"The WordPiece tokenizer {path} needs the 'tokenizers' package "
                "(pip install tokenizers)."
            )
            if required:
                raise ImportError(message)
            warnings.warn(message + " Using the built-in char-level vocabulary instead.", NanoPruneWarning, stacklevel=3)
            return
        self.fast_tokenizer = Tokenizer.from_file(str(path))
        self.vocab = self.fast_tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.json_path = path

    @property
    def kind(self) -> str:
        return "wordpiece" if self.fast_tokenizer is not None else "char"

    @property
    def sha256(self) -> Optional[str]:
        return file_sha256(self.json_path) if self.json_path is not None else None

    def _init_default_subwords(self):
        # Base ASCII bytes (0-127) and common UTF-8 accents/symbols
        for b in range(32, 127):
            ch = chr(b)
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inv_vocab[idx] = ch

        # Common multi-language chars (fr, es, de)
        accented = "éèêëàâäôöûüçîïÉÈÊËÀÂÄÔÖÛÜÇÎÏ«»—–’'°%€$£"
        for ch in accented:
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx
                self.inv_vocab[idx] = ch

        # Key semantic words & medical/legal stems
        keywords = [
            "oui", "non", "aucun", "aucune", "pas", "jamais", "sans", "ne",
            "allergie", "allergique", "traitement", "posologie", "patient", "dossier",
            "antécédent", "diagnostic", "ordonnance", "médicament", "chirurgie", "bilan",
            "yes", "no", "not", "none", "never", "without", "allergy", "patient",
            "date", "nom", "prix", "statut", "clause", "contrat", "article", "loi",
            "vrai", "faux", "pertinent", "trouvé", "score", "recherche", "texte"
        ]
        for word in keywords:
            for i in range(1, len(word) + 1):
                sub = word[:i]
                if sub not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[sub] = idx
                    self.inv_vocab[idx] = sub

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize_word(self, word: str) -> List[str]:
        if not word:
            return []
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            sub = None
            while end > start:
                candidate = word[start:end]
                if candidate in self.vocab:
                    sub = candidate
                    break
                end -= 1
            if sub is None:
                tokens.append(self.UNK_TOKEN)
                start += 1
            else:
                tokens.append(sub)
                start = end
        return tokens

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        norm = self.normalize(text)
        if self.fast_tokenizer is not None:
            token_ids = self.fast_tokenizer.encode(norm).ids
        else:
            words = re.findall(r"[\w']+|[^\w\s]", norm, re.UNICODE)
            token_ids = []
            for w in words:
                for t in self.tokenize_word(w):
                    token_ids.append(self.vocab.get(t, self.unk_id))
        if max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def _pair_ids(self, query_ids: List[int], context: str, max_length: int) -> List[int]:
        max_c = max(0, max_length - 3 - len(query_ids))
        c_ids = self.encode(context, max_length=max_c)
        return [self.cls_id] + query_ids + [self.sep_id] + c_ids + [self.sep_id]

    def encode_pair(
        self, query: str, context: str, max_length: int = 256
    ) -> Tuple[List[int], List[int]]:
        """
        Encodes query and context into:
        [CLS] query_tokens [SEP] context_tokens [SEP]
        padded to ``max_length``. Returns (input_ids, attention_mask).
        """
        # The query keeps at most 64 tokens; [CLS], [SEP], [SEP] are reserved.
        q_ids = self.encode(query, max_length=64)
        input_ids = self._pair_ids(q_ids, context, max_length)
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids.extend([self.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return input_ids, attention_mask

    def encode_batch(
        self,
        query: str,
        contexts: Sequence[str],
        max_length: int = 256,
        pad_to: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Encode (query, context) pairs, padding only to the longest pair.

        Produces exactly the tokens of :meth:`encode_pair`, with less padding.
        ``pad_to`` forces a fixed length (for graphs with a static sequence axis).
        """
        q_ids = self.encode(query, max_length=64)
        rows = [self._pair_ids(q_ids, context, max_length)[:max_length] for context in contexts]
        return self._pad(rows, pad_to)

    def encode_pairs(
        self,
        queries: Sequence[str],
        contexts: Sequence[str],
        max_length: int = 256,
        pad_to: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Like :meth:`encode_batch` for pairs that each have their own query (training batches)."""
        rows = [
            self._pair_ids(self.encode(query, max_length=64), context, max_length)[:max_length]
            for query, context in zip(queries, contexts)
        ]
        return self._pad(rows, pad_to)

    def _pad(self, rows: List[List[int]], pad_to: Optional[int]) -> Tuple[List[List[int]], List[List[int]]]:
        width = pad_to if pad_to is not None else max((len(r) for r in rows), default=0)
        input_ids, masks = [], []
        for row in rows:
            pad = width - len(row)
            input_ids.append(row + [self.pad_id] * pad)
            masks.append([1] * len(row) + [0] * pad)
        return input_ids, masks

    def decode(self, token_ids: List[int]) -> str:
        if self.fast_tokenizer is not None:
            return self.fast_tokenizer.decode(list(token_ids), skip_special_tokens=True)
        tokens = [self.inv_vocab.get(idx, self.UNK_TOKEN) for idx in token_ids]
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return "".join(filtered)
