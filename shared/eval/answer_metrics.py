"""Answer-quality metrics against a gold answer: exact match and token-level F1."""

from __future__ import annotations

import string
from collections import Counter

ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, and collapse whitespace."""
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    words = [word for word in text.split() if word not in ARTICLES]
    return " ".join(words)


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if prediction and gold are identical after normalization, else 0.0."""
    if normalize_answer(prediction) == normalize_answer(gold):
        return 1.0
    else:
        return 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Harmonic mean of token precision and recall after normalization.

    Overlap is counted as a multiset, so a repeated word matches at most as
    many times as it appears in both answers.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
