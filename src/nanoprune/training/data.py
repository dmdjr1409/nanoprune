"""Build (query, passage, label) training pairs from a legal corpus.

Fixes over the v0.4 generator (``scripts/generate_contrastive_dataset.py``):

- **No leakage**: documents are assigned to train/val/test by a stable hash of
  their title *before* pairs are built, and negatives are drawn from the same
  split, so no title or passage appears in two splits.
- **Real hard negatives**: candidates are the passages BM25 ranks highest for
  the query (stopwords removed), not any passage sharing a word such as "de".
- **Consistent categories**: the choice label describes the passage itself,
  whatever the pair type; unknown categories are ignored by the loss.

Titles are still used as queries unless better queries are supplied
(``extra_queries``: human-written or generated search queries per document).
"""
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..engine.lexical import BM25Index, fold_accents

CATEGORY_LABELS = ["human_rights", "business_tax", "public_admin", "civil_family"]
IGNORE_CATEGORY = -100
SPLITS = ("train", "val", "test")

# Weak, keyword-based labels for the choice head (matched on folded text).
_CATEGORY_KEYWORDS: Dict[int, Tuple[str, ...]] = {
    0: ("droits de l'homme", "droits fondamentaux", "cedh", "convention europeenne", "liberte d'expression",
        "constitution", "discrimination", "vie privee"),
    1: ("commerce", "commercial", "societe", "fiscal", "impot", "tva", "faillite", "douane", "bancaire",
        "financier"),
    2: ("arrete", "ministre", "ministeriel", "reglement grand-ducal", "communal", "nomination",
        "fonction publique", "administratif"),
    3: ("code civil", "famille", "divorce", "succession", "mariage", "enfant", "filiation",
        "responsabilite civile", "tutelle"),
}


@dataclass
class Record:
    doc_id: str
    jurisdiction: str
    title: str
    content: str


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def read_legal_tsv(path, max_content_chars: int = 500) -> List[Record]:
    """Read ``jurisdiction<TAB>title<TAB>content`` rows (a literal "\\t" also separates)."""
    records: List[Record] = []
    seen = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t") if "\t" in line else line.strip().split(r"\t")
            if len(parts) < 3:
                continue
            jurisdiction, title, content = clean_text(parts[0]), clean_text(parts[1]), clean_text(parts[2])
            if len(title) <= 5 or len(content) <= 50:
                continue
            content = content[:max_content_chars]
            key = (title, content)
            if key in seen:
                continue
            seen.add(key)
            doc_id = hashlib.sha1(f"{title}\n{content}".encode("utf-8")).hexdigest()[:12]
            records.append(Record(doc_id, jurisdiction, title, content))
    return records


def split_of(key: str, ratios: Sequence[float] = (0.8, 0.1, 0.1)) -> str:
    """Deterministic train/val/test assignment from a key (e.g. the document title)."""
    value = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, ratio in zip(SPLITS, ratios):
        cumulative += ratio
        if value < cumulative:
            return name
    return SPLITS[-1]


def weak_category(record: Record) -> int:
    """Keyword-based category of a passage, or IGNORE_CATEGORY when unclear."""
    if "droit" in fold_accents(record.jurisdiction):
        return 0
    text = fold_accents(f"{record.title} {record.content}")
    hits = {cat: sum(text.count(k) for k in keywords) for cat, keywords in _CATEGORY_KEYWORDS.items()}
    best = max(hits, key=lambda cat: hits[cat])
    if hits[best] == 0 or list(hits.values()).count(hits[best]) > 1:
        return IGNORE_CATEGORY
    return best


def _make_query(title: str, max_chars: int = 90) -> str:
    return title[:max_chars]


def build_pairs(
    records: Sequence[Record],
    n_hard: int = 1,
    n_random: int = 1,
    hard_pool: int = 10,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    extra_queries: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[dict]]:
    """Positive, BM25 hard-negative and random-negative pairs for each split.

    Args:
        extra_queries: optional {doc_id: [queries]} used in addition to titles.
    """
    rng = random.Random(seed)
    by_split: Dict[str, List[Record]] = {name: [] for name in SPLITS}
    for record in records:
        by_split[split_of(record.title, ratios)].append(record)

    result: Dict[str, List[dict]] = {}
    for split, docs in by_split.items():
        rows: List[dict] = []
        if len(docs) < 2:
            result[split] = rows
            continue
        index = BM25Index([d.content for d in docs])
        categories = [weak_category(d) for d in docs]
        for i, doc in enumerate(docs):
            queries = [_make_query(doc.title)] + list((extra_queries or {}).get(doc.doc_id, []))
            for query in queries:
                rows.append(_row(query, doc, 1.0, "positive", categories[i], split))

                ranked = [j for j, _ in index.top_k(query, hard_pool + 1) if j != i and docs[j].title != doc.title]
                for j in rng.sample(ranked, min(n_hard, len(ranked))):
                    rows.append(_row(query, docs[j], 0.0, "hard_negative", categories[j], split))

                others = [j for j in range(len(docs)) if j != i and docs[j].title != doc.title]
                preferred = [j for j in others if docs[j].jurisdiction != doc.jurisdiction] or others
                for j in rng.sample(preferred, min(n_random, len(preferred))):
                    rows.append(_row(query, docs[j], 0.0, "random_negative", categories[j], split))
        rng.shuffle(rows)
        for number, row in enumerate(rows):
            row["pair_id"] = f"{split}-{number:06d}"
        result[split] = rows
    return result


def _row(query: str, doc: Record, label: float, kind: str, category: int, split: str) -> dict:
    return {
        "query": query,
        "content": doc.content,
        "label": label,
        "type": kind,
        "category": category,
        "doc_id": doc.doc_id,
        "split": split,
    }


def write_jsonl(rows: Iterable[dict], path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_wordpiece(texts: Iterable[str], output_json, vocab_size: int = 8192, min_frequency: int = 2) -> int:
    """Train the WordPiece vocabulary on text normalised exactly like NanoTokenizer.encode()."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from ..core.tokenizer import NanoTokenizer

    normalizer = NanoTokenizer.char_level()
    tokenizer = Tokenizer(models.WordPiece(unk_token=NanoTokenizer.UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=list(NanoTokenizer.SPECIAL_TOKENS),
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator((normalizer.normalize(t) for t in texts), trainer=trainer)
    tokenizer.decoder = decoders.WordPiece()
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_json))
    return tokenizer.get_vocab_size()
