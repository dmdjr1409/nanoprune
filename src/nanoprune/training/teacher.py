"""Label training pairs with a teacher for distillation.

The teacher is either Laya (``agent.system_one``) or any relevance scorer with a
``score(query, passages)`` method, e.g. a calibrated ``SemanticScorer``. Each
output row is the input row plus ``"teacher"``: the teacher's probability that
the passage is relevant to the query. ``train(..., teacher_weight=w)`` then
mixes it with the hard label. Labelling resumes where it stopped.
"""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

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
    agent=None,
    question: str = LAYA_QUESTION,
    limit: Optional[int] = None,
    log: Callable[[str], None] = print,
    scorer=None,
) -> int:
    """Append teacher-labelled rows of ``in_path`` to ``out_path``; returns rows written.

    Pass ``agent`` (Laya) or ``scorer`` (any object with ``score(query, passages)``).
    """
    if (agent is None) == (scorer is None):
        raise ValueError("Give exactly one teacher: agent (Laya) or scorer")
    rows = read_jsonl(in_path)
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        done = {row.get("pair_id") for row in read_jsonl(out_path)}
    todo = [row for row in rows if row.get("pair_id") not in done][:limit]
    log(f"{in_path}: {len(rows)} pairs, {len(done)} already labelled, {len(todo)} to label")

    written = 0
    with open(out_path, "a", encoding="utf-8") as handle:
        if scorer is not None:
            groups: Dict[str, List[dict]] = {}
            for row in todo:
                groups.setdefault(row["query"], []).append(row)
            for number, (query, group) in enumerate(groups.items(), 1):
                probs = scorer.score(query, [row["content"] for row in group])
                for row, prob in zip(group, probs):
                    handle.write(json.dumps(dict(row, teacher=round(float(prob), 4)), ensure_ascii=False) + "\n")
                    written += 1
                if number % 100 == 0:
                    handle.flush()
                    log(f"  {written}/{len(todo)} labelled")
            return written

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
