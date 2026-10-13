"""
Shared loop that every category runner uses. Each runner just supplies:
  - how to load its questions
  - how to build a prompt
  - how to score a response
This removes the ~80% duplicated code that used to live in all 7 runners.
"""

import asyncio
from app.utils.provider_clients import call_model
from app.utils.logger import get_logger

logger = get_logger("benchmark")


async def run_benchmark(
    provider: str,
    model: str,
    questions: list,
    build_prompt,          # (question) -> str
    score_fn,              # (response_text, question) -> bool
    system_prompt: str,
    max_tokens: int = 512,
) -> dict:
    """Ask the model each question, score it, return {score, sample_count}."""

    correct = 0
    total = len(questions)

    for q in questions:
        prompt = build_prompt(q)
        try:
            text, _ = await call_model(
                provider=provider, model=model, prompt=prompt,
                system_prompt=system_prompt, max_tokens=max_tokens, temperature=0.0,
            )
            if score_fn(text, q):
                correct += 1
        except Exception:
            # A failed question just scores 0. We do NOT retry here — the retry
            # policy lives in provider_clients and permanent errors (bad model)
            # already failed fast. This keeps one bad question from stalling.
            pass

    score = round((correct / total) * 100, 2) if total else 0.0
    return {"score": score, "sample_count": total}
