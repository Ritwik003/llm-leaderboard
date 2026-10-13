import re
import os
import random
import asyncio
import tempfile
import subprocess
from datasets import load_dataset

from app.utils.settings import settings
from app.utils.provider_clients import call_model
from app.utils.logger import get_logger

logger = get_logger("benchmark")

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the given function. Return ONLY "
    "the complete Python function code — no explanations, no markdown fences."
)


def _load():
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    qs = list(ds)
    return random.sample(qs, min(settings.CODING_SAMPLE_SIZE, len(qs)))


def _prompt(q):
    return f"Complete this Python function:\n\n{q['prompt']}\n\nReturn only the implementation."


def _extract_code(response, original_prompt):
    code = re.sub(r"```python\n?", "", response)
    code = re.sub(r"```\n?", "", code).strip()
    if not code.startswith("def "):
        code = original_prompt.strip() + "\n" + code
    return code


def _run_test(code, test_code, entry_point, timeout=10):
    # HumanEval's `test` blob defines `def check(candidate): ...`. It must be
    # called with the ACTUAL function under test, whose name is `entry_point`
    # (e.g. check(has_close_elements)). The old code passed the literal name
    # `candidate`, which is undefined at call time → NameError on every problem,
    # forcing coding to score 0 for every model.
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run(["python", path], capture_output=True, timeout=timeout, text=True)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(path)


async def run(provider: str, model: str) -> dict:
    """Coding needs a per-question code-execution step, so it doesn't use the
    shared base loop — but it follows the same fail-fast, no-retry pattern."""

    questions = await asyncio.to_thread(_load)
    correct = 0
    total = len(questions)

    for q in questions:
        try:
            text, _ = await call_model(
                provider=provider, model=model, prompt=_prompt(q),
                system_prompt=SYSTEM_PROMPT, max_tokens=1024, temperature=0.0,
            )
            code = _extract_code(text, q["prompt"])
            if await asyncio.to_thread(_run_test, code, q["test"], q["entry_point"], 10):
                correct += 1
        except Exception:
            pass

    score = round((correct / total) * 100, 2) if total else 0.0
    return {"score": score, "sample_count": total}
