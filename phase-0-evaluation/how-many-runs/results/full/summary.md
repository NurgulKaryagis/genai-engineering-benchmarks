Generator `gpt-4o-mini-2024-07-18`, 10 runs, 150 questions. Total API spend: $0.3464.

Per-run means (each run averaged over all questions):

| Condition | Metric | mean of runs | std across runs | min | max |
| --- | --- | --- | --- | --- | --- |
| C1 | EM | 0.570 | 0.0057 | 0.560 | 0.580 |
| C1 | F1 | 0.682 | 0.0062 | 0.673 | 0.694 |
| C1 | abstention | 0.137 | 0.0072 | 0.127 | 0.147 |
| C2 | EM | 0.203 | 0.0072 | 0.193 | 0.213 |
| C2 | F1 | 0.226 | 0.0066 | 0.217 | 0.235 |
| C2 | abstention | 0.663 | 0.0064 | 0.653 | 0.673 |

A/A difference between two disjoint groups of k identical runs, all questions, every possible split (median / 95th percentile):

| Condition | Metric | k = 1 | k = 2 | k = 3 | k = 5 |
| --- | --- | --- | --- | --- | --- |
| C1 | EM | 0.0067 / 0.0133 | 0.0033 / 0.0100 | 0.0022 / 0.0089 | 0.0013 / 0.0067 |
| C1 | F1 | 0.0067 / 0.0162 | 0.0046 / 0.0115 | 0.0036 / 0.0095 | 0.0029 / 0.0069 |
| C1 | abstention | 0.0067 / 0.0200 | 0.0067 / 0.0133 | 0.0044 / 0.0111 | 0.0040 / 0.0093 |
| C2 | EM | 0.0067 / 0.0200 | 0.0067 / 0.0133 | 0.0044 / 0.0111 | 0.0040 / 0.0093 |
| C2 | F1 | 0.0067 / 0.0181 | 0.0048 / 0.0124 | 0.0038 / 0.0103 | 0.0032 / 0.0084 |
| C2 | abstention | 0.0067 / 0.0133 | 0.0033 / 0.0133 | 0.0044 / 0.0089 | 0.0027 / 0.0080 |

EM, run noise vs question-sampling uncertainty by number of questions (200 random subsets per size below 150):

| Condition | questions | run noise: 95th pct A/A difference, k = 3 | question sampling: 95% CI half-width |
| --- | --- | --- | --- |
| C1 | 25 | 0.0267 | 0.188 |
| C1 | 50 | 0.0200 | 0.135 |
| C1 | 100 | 0.0100 | 0.095 |
| C1 | 150 | 0.0089 | 0.080 |
| C2 | 25 | 0.0267 | 0.153 |
| C2 | 50 | 0.0200 | 0.107 |
| C2 | 100 | 0.0133 | 0.076 |
| C2 | 150 | 0.0111 | 0.065 |

Stability of answers across runs:

| Condition | questions whose EM differs across runs | questions with more than one distinct answer |
| --- | --- | --- |
| C1 | 6 of 150 | 21 of 150 |
| C2 | 6 of 150 | 19 of 150 |

| Condition | generator calls | cost (USD) | latency, mean (s) | finish_reason != stop | failed attempts |
| --- | --- | --- | --- | --- | --- |
| C1 | 1500 | 0.1700 | 0.67 | 0 | 0 |
| C2 | 1500 | 0.1764 | 0.68 | 0 | 0 |
