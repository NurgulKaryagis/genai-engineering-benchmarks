"""Summarize items.jsonl and calls.jsonl into the results tables.

Usage:
    python summarize.py results/pilot

Each quality cell reads: mean over runs of the per-run mean, then
"± standard deviation across runs" (only with 2+ runs), then a 95% bootstrap
confidence interval over questions in brackets.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42
CONDITIONS = ["C1", "C2", "C3"]

# (label, key in items.jsonl, only over answered questions)
QUALITY_METRICS = [
    ("recall@5", "recall", False),
    ("MRR", "rr", False),
    ("nDCG@5", "ndcg", False),
    ("EM (all questions)", "em", False),
    ("F1 (all questions)", "f1", False),
    ("EM (answered only)", "em", True),
    ("F1 (answered only)", "f1", True),
    ("faithfulness (1-5)", "faithfulness", True),
    ("answer relevance (1-5)", "relevance", True),
    ("abstention rate", "abstained", False),
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def select(rows: list[dict], key: str, only_answered: bool) -> list[tuple]:
    """Return (row, value) pairs, skipping missing values."""
    pairs = []
    for row in rows:
        if only_answered and row["abstained"]:
            continue
        if row[key] is not None:
            pairs.append((row, float(row[key])))
    return pairs


def grouped_means(pairs: list[tuple], group_key: str) -> list[float]:
    groups = defaultdict(list)
    for row, value in pairs:
        groups[row[group_key]].append(value)
    return [statistics.fmean(values) for values in groups.values()]


def bootstrap_ci(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(N_BOOTSTRAP)
    )
    return means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP) - 1]


def quality_cell(rows: list[dict], key: str, only_answered: bool) -> str:
    pairs = select(rows, key, only_answered)
    run_means = grouped_means(pairs, "run")
    if not run_means:
        return "n/a"
    cell = f"{statistics.fmean(run_means):.3f}"
    if len(run_means) > 1:
        cell += f" ± {statistics.stdev(run_means):.3f}"
    ci = bootstrap_ci(grouped_means(pairs, "question_id"))
    if ci is not None:
        cell += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return cell


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def quality_table(items: list[dict]) -> str:
    by_condition = {c: [r for r in items if r["condition"] == c] for c in CONDITIONS}
    body = [
        [label] + [quality_cell(by_condition[c], key, only) for c in CONDITIONS]
        for label, key, only in QUALITY_METRICS
    ]
    n_questions = len({r["question_id"] for r in items})
    n_runs = len({r["run"] for r in items})
    body.insert(0, ["questions × runs"] + [f"{n_questions} × {n_runs}"] * 3)
    return markdown_table(["Metric"] + CONDITIONS, body)


def operations_table(items: list[dict], calls: list[dict]) -> str:
    ok = [c for c in calls if "error" not in c]
    failed = [c for c in calls if "error" in c]

    def calls_of(condition: str, role: str | None = None) -> list[dict]:
        return [
            c
            for c in ok
            if c["tag"]["condition"] == condition
            and (role is None or c["tag"]["role"] == role)
        ]

    def mean_latency(rows: list[dict]) -> str:
        return (
            f"{statistics.fmean(c['latency_s'] for c in rows):.2f}" if rows else "n/a"
        )

    def per_condition(function) -> list[str]:
        return [function(c) for c in CONDITIONS]

    rows = {
        "generator calls": lambda c: str(len(calls_of(c, "generator"))),
        "judge calls": lambda c: str(len(calls_of(c, "judge"))),
        "input tokens": lambda c: str(sum(x["input_tokens"] for x in calls_of(c))),
        "output tokens": lambda c: str(sum(x["output_tokens"] for x in calls_of(c))),
        "cost, total (USD)": lambda c: f"{sum(x['cost_usd'] for x in calls_of(c)):.4f}",
        "generator latency, mean (s)": lambda c: mean_latency(calls_of(c, "generator")),
        "judge latency, mean (s)": lambda c: mean_latency(calls_of(c, "judge")),
        "failed attempts": lambda c: str(
            sum(1 for x in failed if x["tag"]["condition"] == c)
        ),
        "generator finish_reason != stop": lambda c: str(
            sum(
                1
                for r in items
                if r["condition"] == c and r["gen_finish_reason"] != "stop"
            )
        ),
        "judge parse errors": lambda c: str(
            sum(1 for r in items if r["condition"] == c and r["judge_parse_error"])
        ),
    }
    body = [[label] + per_condition(function) for label, function in rows.items()]
    return markdown_table(["Operations"] + CONDITIONS, body)


OUTCOMES = [
    "exact match",
    "partial overlap (EM 0, F1 > 0)",
    "answered, no overlap (F1 0)",
    "abstained",
]


def outcome(row: dict) -> str:
    if row["abstained"]:
        return "abstained"
    if row["em"] == 1.0:
        return "exact match"
    return "partial overlap (EM 0, F1 > 0)" if row["f1"] > 0 else OUTCOMES[2]


def outcome_table(items: list[dict]) -> str:
    """Share of rows (all runs pooled) in each mutually exclusive answer outcome."""
    counts = {
        c: Counter(outcome(r) for r in items if r["condition"] == c) for c in CONDITIONS
    }
    totals = {c: sum(counts[c].values()) for c in CONDITIONS}
    body = [
        [name]
        + [f"{counts[c][name] / totals[c]:.3f} ({counts[c][name]})" for c in CONDITIONS]
        for name in OUTCOMES
    ]
    body.append(["rows"] + [str(totals[c]) for c in CONDITIONS])
    return markdown_table(["Answer outcome, share of rows (count)"] + CONDITIONS, body)


def summarize(out_dir: Path) -> str:
    items = load_jsonl(out_dir / "items.jsonl")
    calls = load_jsonl(out_dir / "calls.jsonl")
    meta = json.loads((out_dir / "meta.json").read_text())
    total_cost = sum(c["cost_usd"] for c in calls if "error" not in c)
    header = (
        f"Generator `{meta['generator']}`, judge `{meta['judge']}`, seed {meta['seed']}, "
        f"k={meta['k']}. Total API spend: ${total_cost:.4f}."
    )
    return (
        "\n\n".join(
            [
                header,
                quality_table(items),
                outcome_table(items),
                operations_table(items, calls),
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    text = summarize(args.out_dir)
    (args.out_dir / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
