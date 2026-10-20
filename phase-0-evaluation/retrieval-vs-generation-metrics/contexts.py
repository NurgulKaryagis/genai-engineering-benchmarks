"""Build the C1-C3 contexts (2, 1 and 0 gold paragraphs) for each question."""

from __future__ import annotations

import random

# Condition name -> number of gold paragraphs kept in the context.
CONDITIONS = {"C1": 2, "C2": 1, "C3": 0}


def build_contexts(question: dict, seed: int) -> dict[str, list[dict]]:
    """Return the 5-paragraph context for each condition, keyed by condition name.

    All conditions share the C1 layout: a removed gold paragraph is replaced in
    its own slot by a distractor fixed in advance, so paragraphs that stay keep
    their position and the distractor sets are nested (C1 within C2 within C3).
    The result depends only on the seed and the question ID.
    """
    rng = random.Random(f"{seed}:{question['id']}")
    gold = []
    distractors = []
    for paragraph in question["paragraphs"]:
        if paragraph["is_gold"]:
            gold.append(paragraph)
        else:
            distractors.append(paragraph)
    rng.shuffle(gold)
    rng.shuffle(distractors)
    c1_layout = gold + distractors[:3]
    rng.shuffle(c1_layout)
    replacement = {
        gold[0]["id"]: distractors[3],
        gold[1]["id"]: distractors[4],
    }
    contexts = {}
    for name, n_gold in CONDITIONS.items():
        context = list(c1_layout)
        for removed in gold[n_gold:]:
            # Look the slot up in the untouched C1 layout, not in the copy.
            context[c1_layout.index(removed)] = replacement[removed["id"]]
        contexts[name] = context
    return contexts
