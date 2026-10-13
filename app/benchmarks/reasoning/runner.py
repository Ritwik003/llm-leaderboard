import re
import random
import asyncio
from datasets import load_dataset

from app.utils.settings import settings
from app.benchmarks.base import run_benchmark

SYSTEM_PROMPT = (
    "You are an expert scientist. Think step by step, then end with your final "
    "answer as a single letter. Format: 'Final Answer: X'"
)


def _load():
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)
    qs = random.sample(list(ds), min(settings.REASONING_SAMPLE_SIZE, len(ds)))
    # Shuffle the four options per question so the correct answer is NOT always
    # at position A. Fixing it at A (and scoring == "A") let any model that
    # blindly answers "A" score 100% — the documented first-option bias. We
    # store the shuffled order and the correct letter on each question so the
    # prompt and scorer stay in sync.
    for q in qs:
        options = [
            (q["Correct Answer"], True),
            (q["Incorrect Answer 1"], False),
            (q["Incorrect Answer 2"], False),
            (q["Incorrect Answer 3"], False),
        ]
        random.shuffle(options)
        q["_options"] = [text for text, _ in options]
        q["_correct_letter"] = "ABCD"[next(i for i, (_, ok) in enumerate(options) if ok)]
    return qs


def _prompt(q):
    opts = "\n".join(f"{'ABCD'[i]}. {text}" for i, text in enumerate(q["_options"]))
    return (
        f"Question: {q['Question']}\n\n"
        f"{opts}\n\n"
        f"Reason step by step, then give your final answer."
    )


def _extract(text):
    m = re.search(r"final\s+answer\s*[:\-]\s*([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    all_letters = re.findall(r"\b([A-D])\b", (text or "").upper())
    return all_letters[-1] if all_letters else ""


def _score(text, q):
    return _extract(text) == q["_correct_letter"]


async def run(provider: str, model: str) -> dict:
    questions = await asyncio.to_thread(_load)
    # 2048 (not 512): reasoning asks for step-by-step output, and "thinking"
    # models (Mistral's magistral, etc.) spend the budget on internal reasoning
    # before emitting the final letter — 512 truncated them to an empty answer.
    # Non-thinking models stop early, so the larger ceiling costs them nothing.
    return await run_benchmark(provider, model, questions, _prompt, _score,
                               SYSTEM_PROMPT, max_tokens=2048)
