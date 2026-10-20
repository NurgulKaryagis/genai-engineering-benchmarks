# Retrieval metrics vs generation metrics

## Question

Does a high retrieval score imply a high-quality answer?

## Hypothesis

_Written before the experiment ran (2026-09-29)._

I hypothesize that a high retrieval score will be accompanied by a high-quality answer: systems that retrieve well will also generate well.

## Setup

### Variable under test

Retrieval quality, manipulated directly through controlled degradation rather than through different retrievers. Each question comes with 2 gold paragraphs (the supporting facts marked by annotators) and 8 distractors. The generator receives 5 paragraphs, composed as follows:

| Condition | Context (5 paragraphs) | recall@5 |
| --------- | ---------------------- | -------- |
| C1 | 2 gold + 3 distractors | 1.0 |
| C2 | 1 gold + 4 distractors | 0.5 |
| C3 | 0 gold + 5 distractors | 0.0 |

How the contexts are built (`contexts.py`), per question and with a seed derived from the experiment seed and the question ID:

1. The 2 gold paragraphs and the 8 distractors are each shuffled; the order decides which gold paragraph is kept in C2 and which distractors are used first.
2. The C1 context is the 2 gold paragraphs plus the first 3 distractors, shuffled once. This fixes the position (slot) of every paragraph.
3. C2 and C3 start from the C1 layout. Each gold paragraph has its own replacement distractor, fixed in advance (the 4th or 5th distractor). A removed gold paragraph is replaced by it in the same slot: C2 removes one gold paragraph, C3 removes both, so C3 contains C2's replacement.

As a result, only the gold/distractor status of the changed slots differs between conditions: paragraphs that stay keep their position, and the distractor sets are nested (C1 ⊂ C2 ⊂ C3). Context length is roughly constant (always 5 paragraphs). Context ordering is not under test here. A second arm with real retrievers (BM25, an embedding model) is out of scope for this experiment.

### Fixed conditions

- **Dataset:** HotpotQA, distractor setting, validation split (CC BY-SA 4.0; see the [dataset card](../../shared/datasets/hotpotqa/DATASET_CARD.md)). Only questions with exactly 2 gold paragraphs and 8 distractors are eligible (7,345 of 7,405). 150 questions, sampled with seed 42: 127 bridge, 23 comparison.
- **Scope note:** every validation question is labeled `hard`, so results say nothing about easier questions.
- **Generator:** `gpt-4o-mini` (OpenAI), temperature 0, identical prompt across conditions.
- **Generator prompt:** the model is told to answer only from the provided paragraphs and not to use its own knowledge. When the paragraphs do not contain the answer it should say so instead of guessing.
- **Judge:** `gpt-4.1` (OpenAI), a different model from the generator; one call returns both faithfulness and answer relevance. Frozen for the whole experiment.
- **Known limitation:** generator and judge come from the same vendor, so self-preference bias is not controlled. The judge has also not been validated against human labels (separate Phase 0 experiments).
- **Eval harness:** metric code lives in `shared/eval/` and is reused by later experiments.

### Metrics

- **Retrieval (deterministic, against gold):** recall@5, MRR, nDCG@5.
- **Generation (LLM judge):** faithfulness (is the answer supported by the paragraphs?) and answer relevance (does it address the question, regardless of correctness?), each an integer from 1 to 5, returned together as JSON (`prompts.py`). The judge sees the paragraphs, the question and the answer, but not the gold answer. It is not called when the generator abstains, so both scores exist only for answered rows.
- **Reference for answer quality (deterministic):** exact match and token-level F1 against the gold answer. This defines what "high-quality answer" means in the Question. Reported two ways: over all questions (an abstention scores 0) and over answered questions only.
- **Abstention rate (deterministic):** share of answers equal to the fixed refusal sentence `I cannot answer from the provided paragraphs.`, which the generator prompt asks for when the paragraphs do not contain the answer.
- **Design consequence:** because the prompt forbids using the model's own knowledge, low EM/F1 in C3 is partly forced by the prompt and is not by itself a finding. Abstention rate and the answered-only numbers exist to separate "did not answer" from "answered wrongly".

### Runs

3 runs per condition (generator and judge re-called, seed fixed). Reported as mean ± standard deviation, plus a bootstrap confidence interval over questions.

### Cost tracking

- Every call logs input tokens, output tokens, latency and price. The results table includes total cost and mean latency per condition.
- A 10-question pilot ran first to measure real cost ($0.0292). Its results are not used and not committed.

### Versions

- HotpotQA revision `1908d6afbbead072334abe2965f91bd2709910ab`, parquet SHA-256 `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
- Run date 2026-09-30. Python 3.12.14, `openai` 3.22.1, macOS on Apple silicon (API calls only, no local compute).
- Generator `gpt-4o-mini-2024-07-18`, judge `gpt-4.1-2025-04-14`, seed 42, temperature 0.

### Reproduce

From the repository root, with Python 3.12 and an OpenAI API key:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "OPENAI_API_KEY=..." > .env                      # never committed
.venv/bin/python shared/datasets/hotpotqa/fetch.py --n 150 --seed 42
cd phase-0-evaluation/retrieval-vs-generation-metrics
../../.venv/bin/python experiment.py --n 150 --runs 3 --out results/full   # about $1.20
../../.venv/bin/python summarize.py results/full
../../.venv/bin/python explore_c3_answered.py results/full
```

`experiment.py` appends to `items.jsonl`, so use a new `--out` directory for each run. The replies will not match the committed ones exactly, because the API is not fully deterministic.

## Results

Full run: 150 questions × 3 conditions × 3 runs = 1,350 rows (`results/full/`). Each quality cell reads: mean over runs of the per-run mean, ± standard deviation across the 3 runs, and a 95% bootstrap confidence interval over questions in brackets. Faithfulness and relevance are only defined for answered questions (the judge is not called on abstentions). Generated by `summarize.py`.

| Metric | C1 | C2 | C3 |
| --- | --- | --- | --- |
| questions × runs | 150 × 3 | 150 × 3 | 150 × 3 |
| recall@5 | 1.000 ± 0.000 [1.00, 1.00] | 0.500 ± 0.000 [0.50, 0.50] | 0.000 ± 0.000 [0.00, 0.00] |
| MRR | 0.623 ± 0.000 [0.57, 0.68] | 0.412 ± 0.000 [0.37, 0.46] | 0.000 ± 0.000 [0.00, 0.00] |
| nDCG@5 | 0.706 ± 0.000 [0.68, 0.73] | 0.340 ± 0.000 [0.32, 0.36] | 0.000 ± 0.000 [0.00, 0.00] |
| EM (all questions) | 0.558 ± 0.004 [0.48, 0.64] | 0.187 ± 0.012 [0.13, 0.25] | 0.033 ± 0.000 [0.01, 0.06] |
| F1 (all questions) | 0.670 ± 0.006 [0.60, 0.73] | 0.212 ± 0.009 [0.15, 0.28] | 0.039 ± 0.000 [0.01, 0.07] |
| EM (answered only) | 0.662 ± 0.004 [0.58, 0.74] | 0.567 ± 0.019 [0.45, 0.72] | 0.395 ± 0.019 [0.15, 0.69] |
| F1 (answered only) | 0.795 ± 0.004 [0.73, 0.85] | 0.643 ± 0.011 [0.54, 0.78] | 0.448 ± 0.021 [0.15, 0.69] |
| faithfulness (1-5) | 4.763 ± 0.031 [4.58, 4.86] | 4.256 ± 0.066 [3.92, 4.55] | 4.368 ± 0.030 [3.92, 4.85] |
| answer relevance (1-5) | 4.926 ± 0.009 [4.83, 4.99] | 4.487 ± 0.057 [4.21, 4.76] | 4.526 ± 0.022 [4.08, 5.00] |
| abstention rate | 0.158 ± 0.004 [0.10, 0.22] | 0.671 ± 0.010 [0.59, 0.74] | 0.916 ± 0.004 [0.87, 0.96] |

Every row falls into exactly one of four answer outcomes (all 3 runs pooled, 450 rows per condition):

| Answer outcome, share of rows (count) | C1 | C2 | C3 |
| --- | --- | --- | --- |
| exact match | 0.558 (251) | 0.187 (84) | 0.033 (15) |
| partial overlap (EM 0, F1 > 0) | 0.173 (78) | 0.038 (17) | 0.007 (3) |
| answered, no overlap (F1 0) | 0.111 (50) | 0.104 (47) | 0.044 (20) |
| abstained | 0.158 (71) | 0.671 (302) | 0.916 (412) |
| rows | 450 | 450 | 450 |

| Operations | C1 | C2 | C3 |
| --- | --- | --- | --- |
| generator calls | 450 | 450 | 450 |
| judge calls | 379 | 148 | 38 |
| input tokens | 699983 | 516429 | 423149 |
| output tokens | 7380 | 5044 | 3955 |
| cost, total (USD) | 0.7533 | 0.3269 | 0.1229 |
| generator latency, mean (s) | 0.82 | 0.81 | 0.85 |
| judge latency, mean (s) | 0.87 | 0.90 | 0.86 |
| failed attempts | 0 | 0 | 0 |
| generator finish_reason != stop | 0 | 0 | 0 |
| judge parse errors | 0 | 0 | 0 |

Total API spend: $1.2031 for 1,915 calls (1,350 generator, 565 judge), computed from logged token counts and the prices in `shared/eval/llm.py`. Checked against the OpenAI usage dashboard: 1,957 requests and about $1.23 in total, which matches this run plus the 42-request pilot ($1.2031 + $0.0292). No failed attempts, no truncated generations, no malformed judge output. The raw call log (`calls.jsonl`, 8 MB with every prompt and reply) is not committed; `items.jsonl` (one row per run, question and condition), `meta.json` and `summary.md` are. Rerunning `experiment.py` regenerates the log but costs about $1.20 and will not reproduce the replies exactly (see the variance note below). The retrieval metrics have zero variance across runs because the contexts are built deterministically from the seed; only generator and judge output can vary between runs.

## Interpretation

In this setup (HotpotQA hard questions, `gpt-4o-mini`, a prompt that restricts the model to the provided paragraphs), removing the gold paragraphs from the context sharply lowers the share of exact answers and raises abstention, and even with both gold paragraphs present only 56% of answers match the gold answer exactly. Retrieval is necessary but not sufficient for a good answer.

### What was measured, beyond noise

- _Answer quality falls as gold evidence is removed._ Over all questions, EM falls from 0.558 (C1) to 0.187 (C2) to 0.033 (C3), and F1 from 0.670 to 0.212 to 0.039. The 95% intervals of the three conditions do not overlap and the standard deviation across runs is at most 0.012, so these differences are explained neither by run-to-run variation nor by which questions were sampled. Halving recall (C1 to C2) cuts EM to about a third.
- _Abstention rises as gold evidence is removed:_ 0.158, 0.671, 0.916 (intervals do not overlap). Much of this is the model following the prompt, which tells it to refuse when the paragraphs lack the answer. The low all-questions EM in C3 is therefore partly forced by the design and is not a finding on its own.
- _Perfect retrieval does not give perfect answers._ In C1, both gold paragraphs are in the context, yet 55.8% of rows are exact matches, 15.8% are abstentions, 17.3% overlap the gold answer partially and 11.1% share no word with it.

### What the data shows but cannot settle

The slices below were made after seeing the main results, so they are exploratory. The C3 slice is reproducible with `explore_c3_answered.py`; the partial-overlap inspection was done by hand on a seeded sample.

- _Partial overlap._ A seeded hand inspection of 12 of the 25 C1 partial-overlap rows from run 1 found mostly the same answer with extra or missing words (for example gold `Riviera`, answer `Riviera Hotel`; gold `Kylie Ann Minogue`, answer `Kylie Minogue`), plus a few genuinely wrong or over-complete answers (gold `Rudolf Höss`, answer `Kurt Gerstein and Rudolf Höss.`). This suggests that part of the EM gap is the strictness of EM rather than wrong answers, and F1 should be read next to EM. The share was not quantified.
- _Answered-only quality._ Among answered questions EM goes 0.662, 0.567, 0.395 and F1 0.795, 0.643, 0.448, but the intervals of C1 and C2 overlap and each condition answers a different subset of questions (84%, 33% and 8% of rows), so the three columns are not directly comparable. This trend is not established.
- _Judge scores._ Faithfulness (4.76, 4.26, 4.37) and answer relevance (4.93, 4.49, 4.53) are lower in C2 than in C1 (intervals barely separate for faithfulness, separate for relevance), but C2 and C3 cannot be told apart: their intervals overlap and the C3 mean rests on 13 distinct questions. The small C3 increase is not read as real. Scores sit close to the top of the 1-5 scale, which leaves the judge little room to discriminate. The judge sees the paragraphs, not the gold answer, so it measures grounding in the context, not correctness.
- _The 13 questions answered in C3_ (38 rows over 3 runs; 12 questions answered in all three runs, one in two). For all 35 answers that are not yes/no, the answer text appears verbatim in the C3 distractor paragraphs; only 3 of the 38 answers are yes/no. Candidate explanations are that a distractor carried enough information, that the model used its own knowledge despite the prompt, or that it guessed. The data does not separate them (a verbatim match can also be chance).

**Cost and latency.** Total spend was $1.20 for 1,915 calls. Cost per condition falls from $0.75 (C1) to $0.33 (C2) to $0.12 (C3). The generator cost is flat at about $0.05 per condition; the difference comes from the judge ($0.70, $0.27, $0.07), which is called only on answered rows (379, 148 and 38 calls) and is priced higher per token. Latency is about 0.8 s per generator call and 0.9 s per judge call in every condition, so the conditions do not differ in latency.

**Operations.** No failed attempts, no truncated generations and no malformed judge output. Temperature was 0 with a fixed seed, yet the standard deviation across runs is not zero (EM up to 0.019 among answered rows), so the API output is not fully deterministic; the effect is small compared with the differences between conditions.

**Retrieval metrics.** recall@5 (1.0, 0.5, 0.0), MRR and nDCG@5 are manipulated by construction, not measured outcomes. Because the paragraph order is shuffled, MRR and nDCG@5 only reflect where the shuffle happened to place the gold paragraphs (C1 MRR 0.623, close to the 0.642 expected for random placement), and in the three conditions all three metrics fall together. They validate the shared harness; they do not show what each metric tells apart.

**Limits.** Results are specific to HotpotQA validation questions (all labeled hard), one generator, one prompt and one judge. Generator and judge are both OpenAI models, so self-preference is not controlled, and the judge has not been checked against human labels. Retrieval quality was degraded by construction (2, 1 or 0 gold paragraphs), so the results show the effect of missing evidence, not the behavior of BM25 or embedding retrievers. The faithfulness and relevance numbers are computed on different subsets of questions in each condition.

## Theory

Retrieval is necessary because the generator builds its answer from the retrieved text: with the evidence removed, the prompted model mostly refuses (91.6% in C3). It is not sufficient because the generator must still do work once the evidence is in the context. In this task, that work is plausibly three separate things: selecting the relevant paragraphs among hard distractors, combining facts across two paragraphs, and following the output instructions (short answer, refuse only when the answer is absent). The three kinds of loss seen in C1 (abstention, partial overlap, no overlap) may map onto these differently, for example partial overlap mostly onto instruction following and EM strictness, and no overlap onto selection or combination. This mapping, and the effect of model capability (for instance its reasoning ability), are plausible mechanisms and were not tested: the experiment used one model and did not record which paragraphs the model used or why it refused.

## Takeaway

In this setup, a retrieval score alone is not enough to judge a RAG system: removing gold paragraphs lowers answer quality sharply, but with every gold paragraph present only 56% of answers matched exactly (67% by F1), so the generated answers need their own evaluation next to the retrieval metrics. Read EM together with F1 and the abstention rate, since EM penalizes harmless wording differences and a refusal is not the same as a wrong answer. These conclusions hold for HotpotQA hard questions with `gpt-4o-mini` and a context-only prompt; they were not tested with real retrievers, other models or other prompts.

## Open

- Why did the model answer in C3 at all? Distractor content, the model's own knowledge and guessing are all consistent with the data. Not separated.
- Why does the model refuse in C1 even with both gold paragraphs (15.8%), and why 67.1% in C2? Selection, combination and over-cautious instruction following are candidates; the data cannot separate them. In C2 only one of the two gold paragraphs is present, so missing evidence may explain much of it.
- What share of the partial-overlap answers is correct but worded differently? An answer-correctness judge that sees the gold answer would measure this. It is a new metric and belongs with the Phase 0 experiment on judge agreement with human labels.
- Does a stronger or weaker generator change the gap between C1 and the other conditions? Only one model was used.
- How sensitive are abstention and answer format to the prompt? Prompt variants (few-shot format examples, step-by-step answers, allowing the model's own knowledge) are not covered here; they belong to Phase 5.
- What does each retrieval metric tell apart with real retrievers? Here recall, MRR and nDCG fall together. Real retrievers (Phases 2 and 3) will give a natural spread, and a dated `Revisit` section can be added to this README then.
- Would real retrievers show the same effect? Here the gold paragraphs were removed at random, independently of the question. With a real retriever, retrieval failures likely correlate with question difficulty, so the link between recall and answer quality could look different. Not tested.
- Does context order matter (Phase 4)? Order was shuffled randomly and not studied.
- Is the judge reliable, and is self-preference present? Not validated against human labels (Phase 0).
