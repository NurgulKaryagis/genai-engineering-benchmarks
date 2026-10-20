"""Download HotpotQA (distractor setting, validation split) and draw a seeded sample.

Source: https://huggingface.co/datasets/hotpotqa/hotpot_qa (CC BY-SA 4.0).
Raw files are written to shared/datasets/raw/ (gitignored); rerun this script to
recreate them.

Usage:
    python shared/datasets/hotpotqa/fetch.py [--n 150] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
PARQUET_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    f"{REVISION}/distractor/validation-00000-of-00001.parquet"
)
RAW_DIR = Path(__file__).resolve().parents[1] / "raw"
PARQUET_PATH = RAW_DIR / "hotpotqa_distractor_validation.parquet"

# The experiment design assumes 2 gold paragraphs and 8 distractors per question.
N_GOLD = 2
N_DISTRACTORS = 8


def download(path: Path = PARQUET_PATH) -> str:
    """Download the parquet file if missing and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {PARQUET_URL}")
        urllib.request.urlretrieve(PARQUET_URL, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(row: dict) -> dict:
    """Turn a raw parquet row into one question with labeled paragraphs.

    Paragraph IDs are the Wikipedia titles, which are unique within a question.
    """
    gold_titles = set(row["supporting_facts"]["title"])
    paragraphs = [
        {
            "id": title,
            "text": " ".join(sentences).strip(),
            "is_gold": title in gold_titles,
        }
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"], strict=True
        )
    ]
    return {
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "paragraphs": paragraphs,
    }


def is_eligible(question: dict) -> bool:
    """Keep questions that match the experiment's 2 gold + 8 distractor layout."""
    n_gold = sum(p["is_gold"] for p in question["paragraphs"])
    n_distractors = len(question["paragraphs"]) - n_gold
    return n_gold == N_GOLD and n_distractors == N_DISTRACTORS


def load_questions(path: Path = PARQUET_PATH) -> list[dict]:
    table = pq.read_table(path)
    return [normalize(row) for row in table.to_pylist()]


def sample(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Seeded sample of eligible questions, independent of file order."""
    eligible = sorted((q for q in questions if is_eligible(q)), key=lambda q: q["id"])
    return random.Random(seed).sample(eligible, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    digest = download()
    questions = load_questions()
    eligible = [q for q in questions if is_eligible(q)]
    print(f"parquet sha256: {digest}")
    print(f"revision: {REVISION}")
    print(
        f"questions: {len(questions)}, eligible (2 gold + 8 distractors): {len(eligible)}"
    )
    print(f"types: {dict(Counter(q['type'] for q in questions))}")
    print(f"levels: {dict(Counter(q['level'] for q in questions))}")

    picked = sample(questions, args.n, args.seed)
    out_path = RAW_DIR / f"hotpotqa_sample_n{args.n}_seed{args.seed}.json"
    out_path.write_text(json.dumps(picked, ensure_ascii=False, indent=2))
    print(f"wrote {len(picked)} questions to {out_path}")


if __name__ == "__main__":
    main()
