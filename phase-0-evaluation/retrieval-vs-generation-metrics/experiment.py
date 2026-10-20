"""Run the retrieval-vs-generation-metrics experiment.

Usage:
    python experiment.py --n 10 --runs 1 --out results/pilot
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
sys.path.insert(0, str(ROOT / "shared" / "eval"))

from answer_metrics import exact_match, normalize_answer, token_f1
from contexts import CONDITIONS, build_contexts
from llm import LLMResult, call_llm
from prompts import (
    REFUSAL,
    build_generator_messages,
    build_judge_messages,
)
from retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank

GENERATOR_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4.1"
SEED = 42
K = 5
SAMPLE_PATH = ROOT / "shared" / "datasets" / "raw" / "hotpotqa_sample_n150_seed42.json"


def is_refusal(answer: str) -> bool:
    return normalize_answer(answer) == normalize_answer(REFUSAL)


def parse_scores(text: str) -> dict | None:
    """Return {"faithfulness": int, "relevance": int}, or None if malformed."""
    try:
        data = json.loads(text)
        scores = {key: data[key] for key in ("faithfulness", "relevance")}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if all(isinstance(v, int) and 1 <= v <= 5 for v in scores.values()):
        return scores
    return None


def evaluate_item(
    question: dict,
    condition: str,
    paragraphs: list[dict],
    run: int,
    calls_path: Path,
    llm=call_llm,
) -> dict:
    ids = [p["id"] for p in paragraphs]
    gold = {p["id"] for p in question["paragraphs"] if p["is_gold"]}
    tag = {"question_id": question["id"], "condition": condition, "run": run}

    generation: LLMResult = llm(
        GENERATOR_MODEL,
        build_generator_messages(question["question"], paragraphs),
        temperature=0.0,
        seed=SEED,
        max_tokens=64,
        log_path=calls_path,
        tag={**tag, "role": "generator"},
    )
    answer = generation.text.strip()
    abstained = is_refusal(answer)

    row = {
        **tag,
        "recall": recall_at_k(ids, gold, K),
        "rr": reciprocal_rank(ids, gold),
        "ndcg": ndcg_at_k(ids, gold, K),
        "answer": answer,
        "gold_answer": question["answer"],
        "abstained": abstained,
        "em": exact_match(answer, question["answer"]),
        "f1": token_f1(answer, question["answer"]),
        "faithfulness": None,
        "relevance": None,
        "judge_parse_error": False,
        "gen_cost": generation.cost_usd,
        "gen_latency": generation.latency_s,
        "gen_finish_reason": generation.finish_reason,
        "judge_cost": 0.0,
        "judge_latency": None,
    }

    if not abstained:
        judgement: LLMResult = llm(
            JUDGE_MODEL,
            build_judge_messages(question["question"], paragraphs, answer),
            temperature=0.0,
            seed=SEED,
            max_tokens=32,
            response_format={"type": "json_object"},
            log_path=calls_path,
            tag={**tag, "role": "judge"},
        )
        scores = parse_scores(judgement.text)
        row["judge_cost"] = judgement.cost_usd
        row["judge_latency"] = judgement.latency_s
        if scores is None:
            row["judge_parse_error"] = True
        else:
            row.update(scores)
    return row


def write_meta(out_dir: Path, n: int, runs: int) -> None:
    meta = {
        "generator": GENERATOR_MODEL,
        "judge": JUDGE_MODEL,
        "seed": SEED,
        "k": K,
        "n_questions": n,
        "runs": runs,
        "conditions": CONDITIONS,
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
    max_cost: float = 5.0,
) -> None:
    """Write one row per (run, question, condition) to items.jsonl.

    Stops if the spend of this invocation passes max_cost. This is a safety stop
    only; the phase budget is tracked separately, across all runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    write_meta(out_dir, len(questions), runs)
    items_path = out_dir / "items.jsonl"
    calls_path = out_dir / "calls.jsonl"
    total_cost = 0.0
    for run in range(1, runs + 1):
        for i, question in enumerate(questions, start=1):
            contexts = build_contexts(question, SEED)
            for condition, paragraphs in contexts.items():
                row = evaluate_item(
                    question, condition, paragraphs, run, calls_path, llm
                )
                with items_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_cost += row["gen_cost"] + row["judge_cost"]
                if total_cost > max_cost:
                    raise SystemExit(f"stopped: spend ${total_cost:.2f} > ${max_cost}")
            print(f"run {run} question {i}/{len(questions)} spend ${total_cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10, help="number of questions")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "pilot")
    parser.add_argument(
        "--max-cost", type=float, default=5.0, help="safety stop, in USD"
    )
    args = parser.parse_args()
    questions = json.loads(SAMPLE_PATH.read_text())[: args.n]
    run_experiment(questions, args.runs, args.out, max_cost=args.max_cost)


if __name__ == "__main__":
    main()
