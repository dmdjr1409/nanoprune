"""Dependency-free lexical matching for French and English text.

Provides the text normalisation used for keyword retrieval (accent folding,
stopword removal, light stemming), a small BM25 index used to pre-select
candidate chunks before neural re-ranking, and an IDF-weighted query coverage
score used as a transparent lexical baseline in evaluations.
"""
import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Stored accent-folded: tokens are folded before the stopword lookup.
# Negations ("aucun", "pas", "sans", "no", "not"...) are deliberately kept as terms:
# "aucune allergie" must not match like "allergie".
FRENCH_STOPWORDS = frozenset(
    """
    a ai aie aient aies ait as au aupres aussi autre autres aux avaient avais avait avec avoir avons ayant
    c ca car ce ceci cela celle celles celui ceux ces cet cette chaque chez comme comment d dans de des deja depuis donc
    dont du elle elles en encore entre es est et etaient etais etait etant ete etes etre eu eux fait fois font furent
    fut ici il ils j je jusqu jusque l la le les leur leurs lors lorsqu lorsque lui m ma mais me meme memes mes moi
    moins mon n ne nos notre nous on ont ou par parce pendant peu peut peuvent plus pour pourquoi puis qu quand
    que quel quelle quelles quels qui quoi s sa se selon ses si soi soit sommes son sont sous suis sur t ta te tes
    toi ton tous tout toute toutes tres tu un une vers voici voila vos votre vous y
    """.split()
)

ENGLISH_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below between both but
    by can could did do does doing down during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself just me more most my myself nor of off on once only or other our
    ours ourselves out over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which while who whom why will
    with would you your yours yourself yourselves
    """.split()
)

STOPWORDS = FRENCH_STOPWORDS | ENGLISH_STOPWORDS


def fold_accents(text: str) -> str:
    """Lowercase and strip diacritics ("Indemnité" -> "indemnite")."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def light_stem(word: str) -> str:
    """Conservative French/English stemmer based on plural stripping and truncation.

    Truncating to 7 characters conflates most inflections ("indemnité",
    "indemnisation") while keeping frequent lexical traps apart
    ("licenciement" vs "licence").
    """
    if len(word) <= 4 or not word.isalpha():
        return word
    if word[-1] in "sx":
        word = word[:-1]
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word[:7]


def tokenize(text: str, stem: bool = True, remove_stopwords: bool = True) -> List[str]:
    """Split text into folded (and optionally stemmed) terms."""
    terms = []
    for raw in _WORD_RE.findall(fold_accents(text)):
        if remove_stopwords and raw in STOPWORDS:
            continue
        if raw == "_":
            continue
        terms.append(light_stem(raw) if stem else raw)
    return terms


class BM25Index:
    """Okapi BM25 over an in-memory list of documents."""

    def __init__(self, documents: Optional[Sequence[str]] = None, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_freq: Dict[str, int] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        if documents is not None:
            self.build(documents)

    def build(self, documents: Sequence[str]) -> None:
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_lens = []
        for doc_id, text in enumerate(documents):
            counts = Counter(tokenize(text))
            self.doc_lens.append(sum(counts.values()))
            for term, tf in counts.items():
                postings[term].append((doc_id, tf))
        self.postings = dict(postings)
        self.doc_freq = {term: len(plist) for term, plist in self.postings.items()}
        self.doc_count = len(self.doc_lens)
        self.avg_doc_len = (sum(self.doc_lens) / self.doc_count) if self.doc_count else 0.0

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> List[float]:
        """BM25 score of every document for the query (0.0 when no term matches)."""
        result = [0.0] * self.doc_count
        if not self.doc_count:
            return result
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for doc_id, tf in plist:
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_lens[doc_id] / max(self.avg_doc_len, 1e-9))
                result[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return result

    def top_k(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Best (doc_id, score) pairs with a strictly positive score, best first."""
        scored = [(i, s) for i, s in enumerate(self.scores(query)) if s > 0.0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:k]


def idf_weighted_coverage(
    query: str,
    document: str,
    idf: Optional[Dict[str, float]] = None,
) -> float:
    """Share of the query's (IDF-weighted) terms that also occur in the document, in [0, 1].

    Without an ``idf`` table every term weighs 1, i.e. plain query-term coverage.
    """
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    d_terms = set(tokenize(document))
    weight = (lambda t: idf.get(t, max(idf.values(), default=1.0))) if idf else (lambda t: 1.0)
    total = sum(weight(t) for t in q_terms)
    if total <= 0.0:
        return 0.0
    return min(1.0, sum(weight(t) for t in q_terms & d_terms) / total)


def idf_table(documents: Iterable[str]) -> Dict[str, float]:
    """IDF of every term in a corpus (same formula as BM25)."""
    index = BM25Index(list(documents))
    return {term: index.idf(term) for term in index.doc_freq}
