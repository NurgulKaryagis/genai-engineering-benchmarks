# HotpotQA (distractor setting)

Multi-hop question answering over Wikipedia paragraphs. Each question comes with 10 paragraphs: 2 gold paragraphs (the supporting facts annotators marked as necessary) and 8 distractors.

| Field | Value |
| ----- | ----- |
| Source | https://huggingface.co/datasets/hotpotqa/hotpot_qa |
| Paper | Yang et al., 2018, "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering" (arXiv:1809.09600) |
| License | CC BY-SA 4.0 |
| Personal data | None. Questions were written by crowdworkers about Wikipedia articles. |
| Config and split used | `distractor`, `validation` (7,405 questions) |
| Pinned revision | `1908d6afbbead072334abe2965f91bd2709910ab` |
| File SHA-256 | `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6` |

## How it is fetched

`python shared/datasets/hotpotqa/fetch.py --n 150 --seed 42`

The script downloads the pinned parquet file into `shared/datasets/raw/` (gitignored), normalizes each question into labeled paragraphs (`id` = Wikipedia title, `is_gold` from the supporting facts), and draws a seeded sample. Raw data is never committed.

## Filtering

Questions are kept only if they have exactly 2 gold paragraphs and 8 distractors: 7,345 of 7,405 (60 excluded). The sample is drawn from the eligible questions, sorted by ID before sampling so the result does not depend on file order.

## Known properties and caveats

- Every validation question is labeled `hard`. Results say nothing about easier HotpotQA questions.
- Question types in the full validation split: 5,918 bridge, 1,487 comparison. The 150-question sample (seed 42) has 127 bridge and 23 comparison.
- Distractors were selected by similarity to the question, so they are harder negatives than random paragraphs.
- Paragraphs are Wikipedia text; an LLM may have seen them during pretraining, so a question can sometimes be answered without any context.
- Mean paragraph length is about 570 characters.
