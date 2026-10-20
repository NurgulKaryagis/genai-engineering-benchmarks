"""Run the how-many-runs experiment: many generator-only runs of one setup.

Reuses the context builder and prompts of retrieval-vs-generation-metrics
unchanged, so both experiments measure the same thing.

Usage:
    python experiment.py --runs 10 --out results/full
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIRST_EXPERIMENT = HERE.parent / "retrieval-vs-generation-metrics"
sys.path.insert(0, str(ROOT / "shared" / "eval"))
sys.path.insert(0, str(FIRST_EXPERIMENT))

from answer_metrics import exact_match, normalize_answer, token_f1
from contexts import build_contexts
from llm import call_llm
from prompts import REFUSAL, build_generator_messages

GENERATOR_MODEL = "gpt-4o-mini"
SEED = 42
CONDITIONS = ("C1", "C2")
SAMPLE_PATH = ROOT / "shared" / "datasets" / "raw" / "hotpotqa_sample_n150_seed42.json"


def is_refusal(answer: str) -> bool:
    return normalize_answer(answer) == normalize_answer(REFUSAL)


def run_item(
    question: dict,
    condition: str,
    paragraphs: list[dict],
    run: int,
    calls_path: Path,
    llm=call_llm,
) -> dict:
    tag = {"question_id": question["id"], "condition": condition, "run": run}
    result = llm(
        GENERATOR_MODEL,
        build_generator_messages(question["question"], paragraphs),
        temperature=0.0,
        seed=SEED,
        max_tokens=64,
        log_path=calls_path,
        tag={**tag, "role": "generator"},
    )
    answer = result.text.strip()
    return {
        **tag,
        "answer": answer,
        "gold_answer": question["answer"],
        "abstained": is_refusal(answer),
        "em": exact_match(answer, question["answer"]),
        "f1": token_f1(answer, question["answer"]),
        "model_id": result.model_id,
        "cost": result.cost_usd,
        "latency": result.latency_s,
        "finish_reason": result.finish_reason,
    }


def write_meta(out_dir: Path, n: int, runs: int) -> None:
    meta = {
        "generator": GENERATOR_MODEL,
        "seed": SEED,
        "conditions": list(CONDITIONS),
        "n_questions": n,
        "runs": runs,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "openai": importlib.metadata.version("openai"),
        "sample_path": str(SAMPLE_PATH.relative_to(ROOT)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def run_experiment(
    questions: list[dict],
    runs: int,
    out_dir: Path,
    llm=call_llm,
    max_cost: float = 1.0,
) -> None:
    """Write one row per (run, question, condition) to items.jsonl.

    Stops if the spend of this invocation passes max_cost (a safety stop).
    """
    items_path = out_dir / "items.jsonl"
    if items_path.exists():
        raise SystemExit(f"{items_path} already exists; use a new --out directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_meta(out_dir, len(questions), runs)
    calls_path = out_dir / "calls.jsonl"
    total_cost = 0.0
    for run in range(1, runs + 1):
        for question in questions:
            contexts = build_contexts(question, SEED)
            for condition in CONDITIONS:
                row = run_item(
                    question, condition, contexts[condition], run, calls_path, llm
                )
                with items_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_cost += row["cost"]
                if total_cost > max_cost:
                    raise SystemExit(f"stopped: spend ${total_cost:.2f} > ${max_cost}")
        print(f"run {run}/{runs} done, spend ${total_cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=150, help="number of questions")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "full")
    parser.add_argument(
        "--max-cost", type=float, default=1.0, help="safety stop, in USD"
    )
    args = parser.parse_args()
    questions = json.loads(SAMPLE_PATH.read_text())[: args.n]
    run_experiment(questions, args.runs, args.out, max_cost=args.max_cost)


if __name__ == "__main__":
    main()
