import re
import random
import asyncio
from datasets import load_dataset

from app.utils.settings import settings
from app.benchmarks.base import run_benchmark

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the multiple choice question with "
    "ONLY the letter of the correct answer (A-J). No explanation. Just the letter."
)


def _load():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
    qs = list(ds)
    return random.sample(qs, min(settings.KNOWLEDGE_SAMPLE_SIZE, len(qs)))


def _prompt(q):
    letters = "ABCDEFGHIJ"
    opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(q.get("options", [])))
    return f"Question: {q['question']}\n\nOptions:\n{opts}\n\nAnswer:"


def _extract(text):
    text = (text or "").strip().upper()
    if not text:
        return ""
    # Prefer an explicit "answer is X" / "answer: X" statement — verbose
    # reasoning models wrap the letter in prose rather than emitting it alone.
    m = re.search(r"ANSWER\s*(?:IS|:|=|-)?\s*\(?([A-J])\)?", text)
    if m:
        return m.group(1)
    # A lone bracketed/parenthesised letter, e.g. "(C)" or "[C]".
    m = re.search(r"[\(\[]\s*([A-J])\s*[\)\]]", text)
    if m:
        return m.group(1)
    # Fall back to the last standalone letter in the response (the conclusion).
    letters = re.findall(r"\b([A-J])\b", text)
    if letters:
        return letters[-1]
    return text[0] if text[0] in "ABCDEFGHIJ" else ""


def _score(text, q):
    return _extract(text) == str(q.get("answer", "")).upper()


async def run(provider: str, model: str) -> dict:
    questions = await asyncio.to_thread(_load)
    # 1024 gives "thinking" models (magistral, etc.) room to finish their
    # internal reasoning before emitting the letter on hard questions; a
    # non-thinking model still answers in a few tokens and stops early.
    return await run_benchmark(provider, model, questions, _prompt, _score,
                               SYSTEM_PROMPT, max_tokens=1024)
