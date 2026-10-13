"""
ESTIMATE — how long will a benchmark run take, and will it fit in the free tier?

Run this BEFORE benchmarking so there are no surprises:

    python app\\scripts\\estimate.py
    python app\\scripts\\estimate.py --provider mistral
    python app\\scripts\\estimate.py --questions 12

It reads the same rate_limits.json the benchmark uses, applies the same model
selection, and reports per-provider time and request cost.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv()
os.environ.setdefault("DB_Configuration", "postgresql://u:p@localhost/db")

from app.utils.rate_limits import limiter          # noqa: E402
from app.utils.provider_clients import (            # noqa: E402
    _fetch_openai_style_models, _fetch_gemini_models, _dedupe, close_client,
)
from app.utils.selection import pick_flagships      # noqa: E402
from app.utils.settings import settings, PROVIDERS  # noqa: E402


def fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


async def gather(provider: str):
    urls = {
        "groq": "https://api.groq.com/openai/v1/models",
        "nvidia": "https://integrate.api.nvidia.com/v1/models",
        "mistral": "https://api.mistral.ai/v1/models",
    }
    keys = {
        "groq": settings.GROQ_API_KEY,
        "nvidia": settings.NVIDIA_API_KEY,
        "mistral": settings.MISTRAL_API_KEY,
    }
    if provider == "gemini":
        models = await _fetch_gemini_models()
    else:
        models = await _fetch_openai_style_models(urls[provider], keys[provider], provider)

    models = _dedupe(models)
    chosen, _ = pick_flagships(models, settings.MODELS_PER_PUBLISHER,
                               settings.MAX_MODELS_PER_PROVIDER, settings.MIN_MODEL_SCORE)
    return chosen


async def main(args):
    q = args.questions or (settings.KNOWLEDGE_SAMPLE_SIZE + settings.REASONING_SAMPLE_SIZE
                           + settings.MATH_SAMPLE_SIZE + settings.CODING_SAMPLE_SIZE)

    print()
    print("=" * 78)
    print(f" BENCHMARK ESTIMATE   ({q} questions per model)")
    print("=" * 78)

    grand_time, grand_calls = 0.0, 0

    for provider in PROVIDERS:
        if args.provider and provider != args.provider:
            continue

        models = await gather(provider)
        if not models:
            print(f"\n  {provider.upper()}: no models (missing API key?)")
            continue

        budget = limiter.budget_for(provider)
        total_calls = len(models) * q + len(models)      # +1 preflight probe each
        p_time = 0.0

        print(f"\n  {provider.upper()}   {len(models)} models   "
              f"{total_calls} calls   budget {budget}")
        print("  " + "-" * 74)

        rows = []
        for m in models:
            rps = limiter.rps_for(provider, m["model_id"])
            t = q / rps
            p_time += t
            rows.append((t, rps, m["model_id"]))

        for t, rps, mid in sorted(rows, reverse=True)[:6]:
            flag = "  <-- SLOW" if t > 300 else ""
            print(f"    {fmt(t):>6}  {rps:6.2f} rps  {mid}{flag}")
        if len(rows) > 6:
            print(f"    ... and {len(rows) - 6} more")

        over = total_calls > budget
        print(f"    TOTAL: {fmt(p_time)}   {total_calls} calls"
              f"{'   ** OVER BUDGET **' if over else ''}")

        grand_time += p_time
        grand_calls += total_calls

    print()
    print("=" * 78)
    print(f" ALL PROVIDERS: {fmt(grand_time)} of pacing, {grand_calls} API calls")
    print("=" * 78)
    print(" Note: this counts REQUIRED WAITING only. Add model response time")
    print(" (roughly 1-3s per call) for the real total.")
    print()
    print(" To go faster: lower the *_SAMPLE_SIZE values in .env")
    print(" (each question removed saves one call per model).")
    print()

    await close_client()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=PROVIDERS)
    ap.add_argument("--questions", type=int, help="override questions per model")
    asyncio.run(main(ap.parse_args()))
