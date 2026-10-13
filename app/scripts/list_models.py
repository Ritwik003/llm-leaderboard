"""
MODEL DISCOVERY — see everything each provider actually offers.

Run:  python list_models.py

Fetches the COMPLETE model list from all 4 providers' own /models endpoints.
Nothing is hardcoded, nothing is capped. Every model id comes live from the API.

Outputs:
  - a readable summary in the console
  - models_report.json   (full raw detail per model)
  - models_report.csv    (open in Excel to sort/filter)

Flags:
  python list_models.py --chat-only     hide non-chat models (audio, embeddings, vision-only, etc.)
  python list_models.py --provider groq only one provider
"""

import os
import sys
import csv
import json
import asyncio
import argparse

import httpx
from dotenv import load_dotenv

load_dotenv()

TIMEOUT = 30.0

KEYS = {
    "groq": os.getenv("GROQ_API_KEY", ""),
    "nvidia": os.getenv("NVIDIA_API_KEY", ""),
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    "mistral": os.getenv("MISTRAL_API_KEY", ""),
}

# Substrings that indicate a model is NOT a plain text-chat model.
# Used ONLY to set the "is_chat" flag — nothing is removed from the report
# unless you pass --chat-only.
NON_CHAT_HINTS = (
    "whisper", "tts", "speech", "audio", "voice", "orpheus",
    "embed", "embedding", "rerank", "retrieval",
    "guard", "moderation", "safety", "shield", "content-safety",
    "ocr", "diffusion", "stable-", "sdxl", "image", "imagen", "clip",
    "video", "riva", "nemoretriever", "parakeet", "canary",
    "veo", "aqa",
)


def looks_like_chat(model_id: str, extra: str = "") -> bool:
    blob = f"{model_id} {extra}".lower()
    return not any(h in blob for h in NON_CHAT_HINTS)


# ── Fetchers ────────────────────────────────────────────────────────────────

async def fetch_openai_style(client, url, key, provider):
    """Groq / NVIDIA / Mistral all expose an OpenAI-compatible /models."""
    if not key:
        return [], f"no API key set for {provider}"

    try:
        r = await client.get(url, headers={"Authorization": f"Bearer {key}"})
        if r.status_code != 200:
            return [], f"HTTP {r.status_code}: {r.text[:120]}"

        out = []
        for m in r.json().get("data", []):
            mid = m.get("id", "")

            # Mistral reports capabilities explicitly — trust it over guessing.
            caps = m.get("capabilities") or {}
            if caps:
                is_chat = bool(caps.get("completion_chat", False))
            else:
                is_chat = looks_like_chat(mid, m.get("description", ""))

            out.append({
                "provider": provider,
                "model_id": mid,
                "display_name": m.get("name") or mid,
                "is_chat": is_chat,
                "active": m.get("active", True),
                "context": (m.get("context_window")
                            or m.get("max_context_length")
                            or m.get("max_model_len") or ""),
                "owned_by": m.get("owned_by", ""),
                "description": (m.get("description", "") or "")[:100],
            })
        return out, None

    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


async def fetch_gemini(client, key):
    """Gemini has its own list endpoint and DOES paginate — follow pageToken."""
    if not key:
        return [], "no API key set for gemini"

    try:
        out = []
        token = None

        while True:
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token

            r = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": key},
                params=params,
            )
            if r.status_code != 200:
                return [], f"HTTP {r.status_code}: {r.text[:120]}"

            body = r.json()
            for m in body.get("models", []):
                mid = m.get("name", "").replace("models/", "")
                methods = m.get("supportedGenerationMethods", [])
                # Gemini tells us directly whether it can do chat completions
                is_chat = "generateContent" in methods and looks_like_chat(mid)

                out.append({
                    "provider": "gemini",
                    "model_id": mid,
                    "display_name": m.get("displayName", mid),
                    "is_chat": is_chat,
                    "active": True,
                    "context": m.get("inputTokenLimit", ""),
                    "owned_by": "google",
                    "description": (m.get("description", "") or "")[:100],
                })

            token = body.get("nextPageToken")
            if not token:
                break

        return out, None

    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


async def gather_all(only_provider=None):
    endpoints = {
        "groq": "https://api.groq.com/openai/v1/models",
        "nvidia": "https://integrate.api.nvidia.com/v1/models",
        "mistral": "https://api.mistral.ai/v1/models",
    }

    results, errors = {}, {}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        tasks = {}

        for provider, url in endpoints.items():
            if only_provider and provider != only_provider:
                continue
            tasks[provider] = fetch_openai_style(client, url, KEYS[provider], provider)

        if not only_provider or only_provider == "gemini":
            tasks["gemini"] = fetch_gemini(client, KEYS["gemini"])

        done = await asyncio.gather(*tasks.values())

        for provider, (models, err) in zip(tasks.keys(), done):
            results[provider] = models
            if err:
                errors[provider] = err

    return results, errors


# ── Reporting ───────────────────────────────────────────────────────────────

def report(results, errors, chat_only):
    all_rows = []

    print()
    print("=" * 78)
    print(" AVAILABLE MODELS PER PROVIDER  (fetched live — nothing hardcoded)")
    print("=" * 78)

    for provider in ("groq", "nvidia", "gemini", "mistral"):
        if provider not in results:
            continue

        models = results[provider]

        if provider in errors:
            print(f"\n### {provider.upper()} — COULD NOT FETCH")
            print(f"    {errors[provider]}")
            continue

        chat = [m for m in models if m["is_chat"]]
        shown = chat if chat_only else models
        shown = sorted(shown, key=lambda m: m["model_id"])

        print(f"\n### {provider.upper()} — {len(models)} total, {len(chat)} chat-capable")
        print("-" * 78)

        for m in shown:
            tag = "" if m["is_chat"] else "  [non-chat]"
            ctx = f"  ctx={m['context']}" if m["context"] else ""
            print(f"  {m['model_id']}{ctx}{tag}")

        all_rows.extend(models)

    # ── summary ──
    print()
    print("=" * 78)
    print(" SUMMARY")
    print("=" * 78)
    total = len(all_rows)
    total_chat = len([m for m in all_rows if m["is_chat"]])
    for provider in ("groq", "nvidia", "gemini", "mistral"):
        if provider in results and provider not in errors:
            ms = results[provider]
            print(f"  {provider:9} {len(ms):4} total   {len([m for m in ms if m['is_chat']]):4} chat")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':9} {total:4} total   {total_chat:4} chat")

    if errors:
        print("\n  Providers that failed:")
        for p, e in errors.items():
            print(f"    {p}: {e}")

    # ── files ──
    with open("models_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if all_rows:
        with open("models_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    print()
    print("  Written: models_report.json  and  models_report.csv")
    print("  Open the CSV in Excel to sort and pick what you want to benchmark.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-only", action="store_true",
                    help="hide non-chat models (audio, embeddings, image, safety)")
    ap.add_argument("--provider", choices=["groq", "nvidia", "gemini", "mistral"],
                    help="only fetch one provider")
    args = ap.parse_args()

    missing = [p for p, k in KEYS.items() if not k]
    if missing:
        print(f"\n  NOTE: no API key found for: {', '.join(missing)}")
        print("  Those providers will be skipped. Add keys to your .env to include them.")

    results, errors = asyncio.run(gather_all(args.provider))
    report(results, errors, args.chat_only)


if __name__ == "__main__":
    main()
