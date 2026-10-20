"""Retrieval metrics: recall@k, MRR, nDCG@k."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at_k(retrieved: Sequence[str], gold: Collection[str], k: int) -> float:
    """Fraction of gold IDs that appear in the first k retrieved IDs."""
    if k <= 0:
        raise ValueError("k must be positive")
    gold_ids = set(gold)
    if not gold_ids:
        # Recall is undefined without gold. Raising instead of returning 0.0 or 1.0
        # keeps a data bug from silently shifting the mean.
        raise ValueError("gold must not be empty")
    top_k = set(retrieved[:k])  # a set: duplicates count once, order is irrelevant
    return len(gold_ids & top_k) / len(gold_ids)


def reciprocal_rank(retrieved: Sequence[str], gold: Collection[str]) -> float:
    """1 / rank of the first gold ID in the retrieved list, or 0.0 if none.

    The mean of this value over questions is MRR.
    """
    gold_ids = set(gold)
    if not gold_ids:
        # Reciprocal rank is undefined without gold. Raising instead of returning 0.0
        # keeps a data bug from silently shifting the mean.
        raise ValueError("gold must not be empty")
    for i, pid in enumerate(retrieved, start=1):
        if pid in gold_ids:
            return 1 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Collection[str], k: int) -> float:
    """DCG of the first k retrieved IDs divided by the best possible DCG.

    Relevance is binary (gold or not), each gold ID counts once at its first
    position, and the ideal ranking places min(len(gold), k) gold IDs on top.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    gold_ids = set(gold)
    if not gold_ids:
        # nDCG is undefined without gold. Raising instead of returning 0.0 or 1.0
        # keeps a data bug from silently shifting the mean.
        raise ValueError("gold must not be empty")
    dcg = 0.0
    pid_set = set()
    for i, pid in enumerate(retrieved[:k], start=1):
        if pid in gold_ids and pid not in pid_set:
            dcg += 1 / math.log2(i + 1)
            pid_set.add(pid)
    n_ideal = min(len(gold_ids), k)
    idcg = 0.0
    for i in range(1, n_ideal + 1):
        idcg += 1 / math.log2(i + 1)
    return dcg / idcg
