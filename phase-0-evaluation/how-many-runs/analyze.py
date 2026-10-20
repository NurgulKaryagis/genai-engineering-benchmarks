"""Turn how-many-runs results into tables: run noise vs runs and questions.

Usage:
    python analyze.py results/full
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

CONDITIONS = ("C1", "C2")
METRICS = (("EM", "em"), ("F1", "f1"), ("abstention", "abstained"))
GROUP_SIZES = (1, 2, 3, 5)
QUESTION_COUNTS = (25, 50, 100, 150)
N_SUBSETS = 200
N_BOOTSTRAP = 1000
SEED = 42

Matrix = dict[str, dict[int, float]]  # question ID -> run -> score


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def score_matrix(items: list[dict], condition: str, key: str) -> Matrix:
    matrix = defaultdict(dict)
    for row in items:
        if row["condition"] == condition:
            matrix[row["question_id"]][row["run"]] = float(row[key])
    return matrix


def run_means(matrix: Matrix, questions: list[str], runs: list[int]) -> list[float]:
    return [statistics.fmean(matrix[q][r] for q in questions) for r in runs]


def split_differences(means: list[float], k: int) -> list[float]:
    """|mean(A) - mean(B)| over every split into two disjoint groups of k runs."""
    indices = range(len(means))
    differences = []
    for group_a in itertools.combinations(indices, k):
        rest = [i for i in indices if i not in group_a]
        for group_b in itertools.combinations(rest, k):
            if group_a < group_b:  # each unordered pair once
                mean_a = statistics.fmean(means[i] for i in group_a)
                mean_b = statistics.fmean(means[i] for i in group_b)
                differences.append(abs(mean_a - mean_b))
    return differences


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round(q * (len(ordered) - 1))]


def bootstrap_half_width(values: list[float], rng: random.Random) -> float:
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(N_BOOTSTRAP)
    )
    low = means[int(0.025 * N_BOOTSTRAP)]
    high = means[int(0.975 * N_BOOTSTRAP) - 1]
    return (high - low) / 2


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def per_run_table(items: list[dict], runs: list[int]) -> str:
    body = []
    for condition in CONDITIONS:
        for label, key in METRICS:
            matrix = score_matrix(items, condition, key)
            means = run_means(matrix, sorted(matrix), runs)
            body.append(
                [
                    condition,
                    label,
                    f"{statistics.fmean(means):.3f}",
                    f"{statistics.stdev(means):.4f}",
                    f"{min(means):.3f}",
                    f"{max(means):.3f}",
                ]
            )
    header = ["Condition", "Metric", "mean of runs", "std across runs", "min", "max"]
    return markdown_table(header, body)


def aa_runs_table(items: list[dict], runs: list[int]) -> str:
    """A/A difference vs number of runs per group, all 150 questions."""
    body = []
    for condition in CONDITIONS:
        for label, key in METRICS:
            matrix = score_matrix(items, condition, key)
            means = run_means(matrix, sorted(matrix), runs)
            row = [condition, label]
            for k in GROUP_SIZES:
                differences = split_differences(means, k)
                row.append(
                    f"{statistics.median(differences):.4f} / "
                    f"{percentile(differences, 0.95):.4f}"
                )
            body.append(row)
    header = ["Condition", "Metric"] + [f"k = {k}" for k in GROUP_SIZES]
    return markdown_table(header, body)


def questions_table(items: list[dict], runs: list[int], k: int = 3) -> str:
    """EM: run noise (A/A, k runs) next to question-sampling uncertainty, by n."""
    rng = random.Random(SEED)
    body = []
    for condition in CONDITIONS:
        matrix = score_matrix(items, condition, "em")
        all_questions = sorted(matrix)
        for n in QUESTION_COUNTS:
            n_subsets = 1 if n == len(all_questions) else N_SUBSETS
            differences, half_widths = [], []
            for _ in range(n_subsets):
                subset = rng.sample(all_questions, n)
                differences.extend(
                    split_differences(run_means(matrix, subset, runs), k)
                )
                if len(half_widths) < 20:
                    question_scores = [
                        statistics.fmean(matrix[q][r] for r in runs) for q in subset
                    ]
                    half_widths.append(bootstrap_half_width(question_scores, rng))
            body.append(
                [
                    condition,
                    str(n),
                    f"{percentile(differences, 0.95):.4f}",
                    f"{statistics.fmean(half_widths):.3f}",
                ]
            )
    header = [
        "Condition",
        "questions",
        f"run noise: 95th pct A/A difference, k = {k}",
        "question sampling: 95% CI half-width",
    ]
    return markdown_table(header, body)


def stability_table(items: list[dict]) -> str:
    body = []
    for condition in CONDITIONS:
        em = score_matrix(items, condition, "em")
        answers = defaultdict(set)
        for row in items:
            if row["condition"] == condition:
                answers[row["question_id"]].add(row["answer"])
        unstable = [q for q in em if len(set(em[q].values())) > 1]
        varied = [q for q in answers if len(answers[q]) > 1]
        body.append(
            [
                condition,
                f"{len(unstable)} of {len(em)}",
                f"{len(varied)} of {len(answers)}",
            ]
        )
    header = [
        "Condition",
        "questions whose EM differs across runs",
        "questions with more than one distinct answer",
    ]
    return markdown_table(header, body)


def operations_table(items: list[dict], calls: list[dict]) -> str:
    failed = [c for c in calls if "error" in c]
    body = []
    for condition in CONDITIONS:
        rows = [r for r in items if r["condition"] == condition]
        body.append(
            [
                condition,
                str(len(rows)),
                f"{sum(r['cost'] for r in rows):.4f}",
                f"{statistics.fmean(r['latency'] for r in rows):.2f}",
                str(sum(r["finish_reason"] != "stop" for r in rows)),
                str(sum(c["tag"]["condition"] == condition for c in failed)),
            ]
        )
    header = [
        "Condition",
        "generator calls",
        "cost (USD)",
        "latency, mean (s)",
        "finish_reason != stop",
        "failed attempts",
    ]
    return markdown_table(header, body)


def summarize(out_dir: Path) -> str:
    items = load_jsonl(out_dir / "items.jsonl")
    calls_path = out_dir / "calls.jsonl"
    calls = load_jsonl(calls_path) if calls_path.exists() else []
    runs = sorted({row["run"] for row in items})
    model_ids = sorted({row["model_id"] for row in items})
    total_cost = sum(row["cost"] for row in items)
    header = (
        f"Generator {', '.join(f'`{m}`' for m in model_ids)}, {len(runs)} runs, "
        f"{len({r['question_id'] for r in items})} questions. "
        f"Total API spend: ${total_cost:.4f}."
    )
    sections = [
        header,
        "Per-run means (each run averaged over all questions):",
        per_run_table(items, runs),
        (
            "A/A difference between two disjoint groups of k identical runs, all "
            "questions, every possible split (median / 95th percentile):"
        ),
        aa_runs_table(items, runs),
        (
            "EM, run noise vs question-sampling uncertainty by number of questions "
            f"({N_SUBSETS} random subsets per size below 150):"
        ),
        questions_table(items, runs),
        "Stability of answers across runs:",
        stability_table(items),
        operations_table(items, calls),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    text = summarize(args.out_dir)
    (args.out_dir / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
