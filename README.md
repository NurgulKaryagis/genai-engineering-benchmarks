# GenAI Engineering Benchmarks

A structured program of controlled experiments covering the core decisions of GenAI engineering: evaluation, document processing, embeddings, retrieval, generation, prompting, agents, fine-tuning, and serving. Each experiment answers one real engineering question with measurements on my own setup.

## Results Matrix

| Phase | Question | Result | Link |
| ----- | -------- | ------ | ---- |
| 0 - Evaluation | Does a high retrieval score imply a high-quality answer? | Retrieval is necessary but not sufficient: removing gold paragraphs cut EM from 0.56 to 0.19 to 0.03, yet with both gold paragraphs present only 56% of answers matched exactly (F1 0.67). Tested on HotpotQA hard questions with `gpt-4o-mini` and degraded (not real) retrieval. | [retrieval-vs-generation-metrics](phase-0-evaluation/retrieval-vs-generation-metrics/README.md) |

## Methodology Notes

Shared harness changes are recorded here, and flagged when they could affect earlier results.

- **2026-09-30, harness created** (`shared/eval/`): `retrieval_metrics.py` (recall@k, reciprocal rank, nDCG@k), `answer_metrics.py` (normalization, exact match, token-level F1) and `llm.py` (OpenAI call wrapper that logs tokens, latency and cost; prices in `PRICES`, read 2026-09-30). First used by `retrieval-vs-generation-metrics`. No earlier results exist, so nothing is affected.
- **Dataset:** HotpotQA distractor validation (CC BY-SA 4.0), see `shared/datasets/hotpotqa/DATASET_CARD.md`.
