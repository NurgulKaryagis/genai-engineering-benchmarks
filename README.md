# GenAI Engineering Benchmarks

Controlled experiments on the everyday decisions of GenAI engineering: evaluation, document processing, embeddings, retrieval, generation, prompting, agents, fine-tuning and serving. Each experiment answers one engineering question with measurements on my own setup, instead of taking a blog post's or a vendor's word for it.

## Results

| Phase | Question | Result | Write-up |
| ----- | -------- | ------ | -------- |
| 0 | Does a high retrieval score imply a high-quality answer? | Necessary but not sufficient: with every gold paragraph in the context, only 56% of answers matched exactly. | [retrieval-vs-generation-metrics](phase-0-evaluation/retrieval-vs-generation-metrics/README.md) |
| 0 | How many runs do we need before a difference between two setups can be trusted? | Within one session 2-3 runs are enough (about 1 point of EM noise), but the same setup shifted by about 2 points between two dates. | [how-many-runs](phase-0-evaluation/how-many-runs/README.md) |

Each write-up lists the setup, the full results with variance, cost and latency, the caveats, and what is still open. Total API spend so far: $1.58.

## How experiments are run

- **Hypothesis first.** The prediction is written before anything runs and kept even when it turns out wrong.
- **One variable at a time.** Seeds, datasets and the evaluation harness stay fixed; only the variable under test changes.
- **Multiple runs.** Results are reported as mean, run-to-run spread and a confidence interval over questions. Differences inside the noise are not reported as findings.
- **Quality never alone.** Cost and latency are in the same tables as the quality metrics.
- **Nothing hidden.** Failed and anomalous runs are kept and reported, and "no meaningful difference" is a valid result.
- **Reproducible.** Model IDs, library versions and dataset checksums are recorded, and every write-up has the commands to rerun it.

## Roadmap

| Phase | Topic | Status |
| ----- | ----- | ------ |
| 0 | Evaluation: metrics, run-to-run variance, LLM-as-judge | In progress (2 of 3 core experiments) |
| 1 | Document processing: parsers, OCR, tables | Planned |
| 2 | Embeddings and indexing: model choice, quantization, HNSW | Planned |
| 3 | Retrieval: chunking, hybrid search, rerankers | Planned |
| 4 | Generation and RAG: number of chunks, long context vs RAG, abstention | Planned |
| 5 | Prompting and structured output | Planned |
| 6 | Agents: planning styles, multi-agent, tool selection | Planned |
| 7 | Fine-tuning: LoRA and variants, data size, forgetting | Planned |
| 8 | Serving and cost: quantization, self-hosting, routing | Planned |

## Repository layout

```text
shared/eval/          evaluation harness used by every experiment (metrics, LLM call logging)
shared/datasets/      fetch scripts and dataset cards (raw data is downloaded, not committed)
phase-N-<topic>/      one folder per experiment: README, code, results
```

## Reproduce

Python 3.12 and an OpenAI API key:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "OPENAI_API_KEY=..." > .env
.venv/bin/python shared/datasets/hotpotqa/fetch.py --n 150 --seed 42
```

Then follow the "Reproduce" section of the experiment's README.

## Methodology notes

Changes to the shared harness are recorded here, and flagged when they could affect earlier results.

- **2026-09-30, harness created** (`shared/eval/`): `retrieval_metrics.py` (recall@k, reciprocal rank, nDCG@k), `answer_metrics.py` (normalization, exact match, token-level F1) and `llm.py` (OpenAI call wrapper that logs tokens, latency and cost; prices in `PRICES`, read 2026-09-30). First used by `retrieval-vs-generation-metrics`. No earlier results exist, so nothing is affected.
- **Dataset:** HotpotQA distractor validation (CC BY-SA 4.0), see `shared/datasets/hotpotqa/DATASET_CARD.md`.

## License

[MIT](LICENSE)
