"""Label training pairs with a teacher (Laya) for distillation.

Each output row is the input row plus ``"teacher"``: the teacher's probability
that the passage is relevant to the query. ``train(..., teacher_weight=w)``
then mixes it with the hard label. Labelling resumes where it stopped.
"""
import json
from pathlib import Path
from typing import Callable, Optional

from ..evaluation import LAYA_QUESTION
from .data import read_jsonl


def laya_relevance(agent, query: str, passage: str, question: str = LAYA_QUESTION) -> float:
    res = agent.system_one(
        f"Document: {passage}",
        {"relevance": {"type": "noul", "instructions": question.format(query=query)}},
    )
    return float(res["answers"]["relevance"]["noul"])


def label_file(
    in_path,
    out_path,
    agent,
    question: str = LAYA_QUESTION,
    limit: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> int:
    """Append teacher-labelled rows of ``in_path`` to ``out_path``; returns rows written."""
    rows = read_jsonl(in_path)
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        done = {row.get("pair_id") for row in read_jsonl(out_path)}
    todo = [row for row in rows if row.get("pair_id") not in done][:limit]
    log(f"{in_path}: {len(rows)} pairs, {len(done)} already labelled, {len(todo)} to label")
    written = 0
    with open(out_path, "a", encoding="utf-8") as handle:
        for number, row in enumerate(todo, 1):
            try:
                row = dict(row, teacher=round(laya_relevance(agent, row["query"], row["content"], question), 4))
            except Exception as exc:  # keep going; the pair can be retried later
                log(f"  skipped {row.get('pair_id')}: {exc}")
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if number % 100 == 0:
                handle.flush()
                log(f"  {number}/{len(todo)} labelled")
    return written
