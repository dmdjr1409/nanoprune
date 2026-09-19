import re
from typing import List, Dict, Optional, Tuple

class NanoTokenizer:
    """
    Compact subword tokenizer for NanoPrune.
    Designed for zero-dependency local execution and tiny model footprints.
    """
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.vocab = vocab
            self.inv_vocab = {v: k for k, v in vocab.items()}
        else:
            self.vocab = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
            self.inv_vocab = {idx: tok for idx, tok in enumerate(self.SPECIAL_TOKENS)}
            self._init_default_subwords()

        self.pad_id = self.vocab[self.PAD_TOKEN]
        self.unk_id = self.vocab[self.UNK_TOKEN]
        self.cls_id = self.vocab[self.CLS_TOKEN]
        self.sep_id = self.vocab[self.SEP_TOKEN]

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
        words = re.findall(r"[\w']+|[^\w\s]", norm, re.UNICODE)
        token_ids = []
        for w in words:
            for t in self.tokenize_word(w):
                token_ids.append(self.vocab.get(t, self.unk_id))
        if max_length is not None:
            token_ids = token_ids[:max_length]
        return token_ids

    def encode_pair(
        self, query: str, context: str, max_length: int = 256
    ) -> Tuple[List[int], List[int]]:
        """
        Encodes query and context into:
        [CLS] query_tokens [SEP] context_tokens [SEP]
        Returns (input_ids, attention_mask).
        """
        # Reserve slots for [CLS], [SEP], [SEP]
        avail = max(0, max_length - 3)
        max_q = min(len(query.split()), 64)
        max_c = avail - max_q

        q_ids = self.encode(query, max_length=max_q)
        c_ids = self.encode(context, max_length=max_c)

        input_ids = [self.cls_id] + q_ids + [self.sep_id] + c_ids + [self.sep_id]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids.extend([self.pad_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        return input_ids, attention_mask

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.inv_vocab.get(idx, self.UNK_TOKEN) for idx in token_ids]
        filtered = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        return "".join(filtered)
