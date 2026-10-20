"""Generator and judge prompts. They are fixed for the whole experiment."""

from __future__ import annotations

REFUSAL = "I cannot answer from the provided paragraphs."

GENERATOR_SYSTEM = f"""\
You answer questions using only the paragraphs provided by the user. \
Do not use any outside knowledge.

Reply with the shortest phrase that answers the question, such as a name, \
a date, a number, "yes" or "no". Do not explain your answer.

If the paragraphs do not contain the information needed to answer, reply \
exactly: {REFUSAL}"""

JUDGE_SYSTEM = """\
You grade an answer to a question. You are given the paragraphs the answerer \
was allowed to use, the question, and the answer. Grade two properties \
independently, each on a scale from 1 to 5.

Faithfulness: is the answer supported by the paragraphs? Judge only against \
the paragraphs, not against what you know about the world.
5 = every claim in the answer is directly supported by the paragraphs.
3 = the answer is partly supported, or supported only by a loose inference.
1 = the answer is not supported by the paragraphs or contradicts them.

Answer relevance: does the answer address the question that was asked? Do not \
judge whether the answer is correct.
5 = the answer directly and completely addresses the question.
3 = the answer addresses the question only partly or is vague.
1 = the answer does not address the question.

Reply with a JSON object with exactly two keys: "faithfulness" (integer 1-5) \
and "relevance" (integer 1-5). Output nothing else."""


def format_paragraphs(paragraphs: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {paragraph['id']}\n{paragraph['text']}"
        for i, paragraph in enumerate(paragraphs, start=1)
    )


def build_generator_messages(question: str, paragraphs: list[dict]) -> list[dict]:
    user = f"Paragraphs:\n\n{format_paragraphs(paragraphs)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_judge_messages(
    question: str, paragraphs: list[dict], answer: str
) -> list[dict]:
    user = (
        f"Paragraphs:\n\n{format_paragraphs(paragraphs)}\n\n"
        f"Question: {question}\n\nAnswer: {answer}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
