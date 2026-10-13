import re
import random
import asyncio
from datasets import load_dataset

from app.utils.settings import settings
from app.benchmarks.base import run_benchmark

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve step by step and put your final "
    "answer inside \\boxed{} at the end."
)


def _load():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", trust_remote_code=True)
    qs = list(ds)
    return random.sample(qs, min(settings.MATH_SAMPLE_SIZE, len(qs)))


def _boxed(text):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1).strip() if m else ""


def _norm(a):
    return a.strip().replace(" ", "").replace(",", "").lower()


def _prompt(q):
    return f"Problem: {q['problem']}\n\nSolve step by step, final answer in \\boxed{{}}."


def _score(text, q):
    # Gold comes from MATH-500's own `answer` field, not from re-parsing
    # \boxed{} out of the solution text. If the model's output was truncated
    # before it emitted \boxed{}, fall back to the last number in the response
    # so a correct-but-unboxed answer still scores.
    got = _boxed(text)
    if not got:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
        got = nums[-1] if nums else ""
    gold = str(q.get("answer") or _boxed(q.get("solution", "")))
    if not got or not gold:
        return False
    if _norm(got) == _norm(gold):
        return True
    try:
        return abs(float(re.sub(r"[\\{}]|frac", "", got)) -
                   float(re.sub(r"[\\{}]|frac", "", gold))) < 1e-6
    except (ValueError, TypeError):
        return False


async def run(provider: str, model: str) -> dict:
    questions = await asyncio.to_thread(_load)
    return await run_benchmark(provider, model, questions, _prompt, _score,
                               SYSTEM_PROMPT, max_tokens=1024)
