"""Exploratory (post-hoc): what kind of questions get answered in C3?

C3 has no gold paragraph, yet the generator answered some questions. This script
slices those rows by question type, yes/no answers and whether the answer text
appears in the distractor paragraphs. It was written after the main results, so
it is labeled exploratory in the README.

Usage:
    python explore_c3_answered.py results/full
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared" / "eval"))

from answer_metrics import normalize_answer
from contexts import build_contexts
from experiment import SAMPLE_PATH, SEED

YES_NO = {"yes", "no"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    questions = {q["id"]: q for q in json.loads(SAMPLE_PATH.read_text())}
    rows = [
        json.loads(line)
        for line in (args.out_dir / "items.jsonl").read_text().splitlines()
    ]
    c3 = [r for r in rows if r["condition"] == "C3"]
    answered = [r for r in c3 if not r["abstained"]]

    print(f"C3 rows: {len(c3)}, answered rows: {len(answered)}")
    unique = Counter(r["question_id"] for r in answered)
    print(f"unique questions answered in at least one run: {len(unique)}")
    print(f"times answered (out of 3 runs): {dict(Counter(unique.values()))}")

    sample_types = Counter(q["type"] for q in questions.values())
    print(f"\nquestion type in the 150-question sample: {dict(sample_types)}")
    print(
        "question type of answered rows: "
        f"{dict(Counter(questions[r['question_id']]['type'] for r in answered))}"
    )
    print(
        "answered unique questions by type: "
        f"{dict(Counter(questions[q]['type'] for q in unique))}"
    )

    yes_no = [r for r in answered if normalize_answer(r["answer"]) in YES_NO]
    print(f"\nanswers that are yes/no: {len(yes_no)} of {len(answered)}")

    in_distractors = 0
    checked = 0
    for row in answered:
        answer = normalize_answer(row["answer"])
        if answer in YES_NO or not answer:
            continue
        question = questions[row["question_id"]]
        context = build_contexts(question, SEED)["C3"]
        text = " ".join(normalize_answer(p["text"]) for p in context)
        checked += 1
        in_distractors += answer in text
    print(
        f"non yes/no answers found verbatim in the C3 distractor text: "
        f"{in_distractors} of {checked}"
    )

    by_type = defaultdict(list)
    for row in answered:
        by_type[questions[row["question_id"]]["type"]].append(row)
    print()
    for question_type, group in sorted(by_type.items()):
        print(
            f"{question_type}: {len(group)} answered rows, "
            f"mean EM {statistics.fmean(r['em'] for r in group):.3f}, "
            f"mean F1 {statistics.fmean(r['f1'] for r in group):.3f}"
        )


if __name__ == "__main__":
    main()
