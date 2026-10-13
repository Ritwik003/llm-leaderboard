"""
PICK MODELS — choose which models are worth benchmarking, and show the reasoning.

Standalone. Needs only httpx + python-dotenv. Does not import the app.

    python app\\scripts\\pick_models.py
    python app\\scripts\\pick_models.py --from-file models_report.json   # no API calls
    python app\\scripts\\pick_models.py --show-all                        # score breakdown
    python app\\scripts\\pick_models.py --per-publisher 2
    python app\\scripts\\pick_models.py --save picks.json

PHILOSOPHY
    Take the BEST MODEL FROM EVERY PUBLISHER — Meta's best, Google's best,
    Qwen's best, DeepSeek's best, Moonshot's best, Z-AI's best, and so on.
    Then let the BENCHMARK decide the top 10, not a heuristic.

    An earlier version capped this at 10 models per provider, which cut strong
    models like kimi-k2.6, glm-5.2 and minimax-m3 before they ever ran. That
    was backwards: pre-judging quality is the benchmark's job. Now there is no
    cap by default (--max 0), so every publisher's flagship gets a fair run.

NOTHING IS HARDCODED. No model name appears anywhere in this file. Ranking
reads signals already encoded in the model id:

    meta/llama-3.3-70b-instruct
     |      |    |    |
     |      |    |    +-- instruction-tuned
     |      |    +------- 70B parameters
     |      +------------ version 3.3
     +--------------------publisher -> the grouping key

CAVEAT: a heuristic, not ground truth. It only has to pick the best model
WITHIN one publisher — a much easier job than ranking across publishers,
which is what the benchmark itself measures.
"""

import os
import re
import sys
import json
import asyncio
import argparse

import httpx
from dotenv import load_dotenv

load_dotenv()

KEYS = {
    "groq": os.getenv("GROQ_API_KEY", ""),
    "nvidia": os.getenv("NVIDIA_API_KEY", ""),
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    "mistral": os.getenv("MISTRAL_API_KEY", ""),
}
PROVIDERS = ["groq", "nvidia", "gemini", "mistral"]


# ══════════════════════════════════════════════════════════════════════════
# 1. NON-CHAT FILTER
# ══════════════════════════════════════════════════════════════════════════

NON_CHAT = (
    # speech / audio
    "whisper", "tts", "speech", "audio", "voice", "orpheus", "voxtral",
    "realtime", "transcribe", "live", "omni", "canary", "parakeet",
    # embeddings / retrieval
    "embed", "rerank", "retrieval", "bge-", "nemoretriever", "arctic-embed",
    # safety / moderation / reward
    "guard", "moderation", "safety", "shield", "topic-control", "reward",
    # image / video / music
    "image", "imagen", "diffusion", "stable-", "sdxl", "clip", "nvclip",
    "veo", "lyria", "nano-banana", "vision", "-vl-", "vlm",
    # documents / specialised
    "ocr", "parse", "deplot", "fuyu", "kosmos", "neva", "vila", "gliner",
    "pii", "robotics", "computer-use", "-fim", "riva", "translate",
    # agentic systems, not plain chat models — these run multi-step tool loops
    # and a single call can take minutes, which would stall a benchmark run
    "vibe-cli", "code-agent", "compound", "deep-research", "antigravity",
)


def is_chat(model_id: str) -> bool:
    low = model_id.lower()
    return not any(s in low for s in NON_CHAT)


# ══════════════════════════════════════════════════════════════════════════
# 2. DEDUPE
# ══════════════════════════════════════════════════════════════════════════

DATED = re.compile(r"^(?P<base>.+)-\d{3,6}$")


def dedupe(models):
    seen = {}
    for m in models:
        seen.setdefault(m["model_id"], m)
    models = list(seen.values())

    latest_bases = {
        m["model_id"][: -len("-latest")]
        for m in models if m["model_id"].endswith("-latest")
    }

    out = []
    for m in models:
        d = DATED.match(m["model_id"])
        if d and d.group("base") in latest_bases:
            continue
        out.append(m)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 3. RANKING SIGNALS
# ══════════════════════════════════════════════════════════════════════════

SIZE = re.compile(r"(?<![a-z])(\d+(?:\.\d+)?)\s*x?\s*(\d+(?:\.\d+)?)?\s*b(?![a-z])")

# Tier words are matched as whole TOKENS, not substrings. Substring matching
# had a real bug: "minimax" contains "max", so minimax-m3 was wrongly awarded
# the +40 flagship bonus.
TIER_TOP = {"large", "pro", "ultra", "max", "opus", "xl", "huge", "premier", "flagship"}
TIER_MID = {"medium", "super", "plus", "standard"}
TIER_LOW = {"mini", "nano", "tiny", "lite", "xs", "small", "micro", "edge", "instant"}

TUNED = {"instruct", "it", "chat", "versatile", "instant", "latest",
         "preview", "pro", "flash", "medium", "small", "large"}


def tokens(model_id):
    return set(re.split(r"[-_/.]+", model_id.lower()))


def parse_size(model_id):
    """Largest parameter count in billions. '8x22b' -> 176."""
    best = 0.0
    for m in SIZE.finditer(model_id.lower()):
        a, b = m.group(1), m.group(2)
        best = max(best, float(a) * float(b) if b else float(a))
    return best


def parse_version(model_id):
    """
    Version number from the id.

    Two patterns are recognised, because vendors name models differently:
      A) version attached to the family   llama-3.3, qwen3.5, gemini-3.1
      B) short letter-prefixed version    kimi-k2.6, minimax-m3, deepseek-v4

    Pattern B matters a lot. Without it, kimi-k2.6 and glm-5.2 parse as
    version 0 and score near-zero despite being strong modern models.

    Parameter counts and dates are stripped FIRST — otherwise 'gpt-oss-20b'
    reads as version 20 and beats 'gpt-oss-120b', which is backwards.
    """
    name = model_id.split("/")[-1].lower()
    name = re.sub(r"\d+(?:\.\d+)?\s*x?\s*\d*\s*b(?![a-z])", " ", name)  # 70b, 8x22b
    name = re.sub(r"a\d+b", " ", name)                                  # MoE active
    name = re.sub(r"\b\d{3,6}\b", " ", name)                            # dates

    candidates = []

    m = re.match(r"[a-z]+-?(\d+(?:\.\d+)?)", name)                      # pattern A
    if m:
        candidates.append(m.group(1))

    for m in re.finditer(r"\b[a-z]{1,4}-?(\d+(?:\.\d+)?)\b", name):     # pattern B
        candidates.append(m.group(1))

    best = 0.0
    for c in candidates:
        try:
            v = float(c)
        except ValueError:
            continue
        if v <= 100:
            best = max(best, v)
    return best


def tier_bonus(model_id):
    tk = tokens(model_id)
    if tk & TIER_TOP:
        return 40.0
    if tk & TIER_MID:
        return 18.0
    if tk & TIER_LOW:
        return -30.0
    return 0.0


def is_tuned(model_id):
    return bool(tokens(model_id) & TUNED)


def publisher(model_id):
    """Grouping key: publisher if namespaced, else model family."""
    low = model_id.lower()
    if "/" in low:
        org = low.split("/")[0]
        return org.split("-")[0] if ("-" in org and not org[0].isdigit()) else org
    if low.startswith("open-"):
        low = low[5:]
    m = re.match(r"[a-z]+", low)
    return m.group(0) if m else low


def score_parts(model):
    """Score split into components so every pick is explainable."""
    mid = model["model_id"]
    ctx = float(model.get("context") or 0)

    # Instruction tuning is a positive signal but NOT dominant. Many strong
    # modern models (kimi-k2.6, glm-5.2, yi-large) simply don't put "instruct"
    # in the name, and a large bonus here unfairly buried them.
    tuned = 25.0 if is_tuned(mid) else 0.0
    tier = tier_bonus(mid)
    ver = parse_version(mid) * 10.0
    size_b = parse_size(mid)
    size = (min(size_b, 1000.0) ** 0.5) * 3.0 if size_b else 0.0
    ctxs = min(ctx / 100_000.0, 10.0) if ctx else 0.0

    return {
        "tuned": tuned, "tier": tier, "version": ver,
        "size": size, "size_b": size_b, "context": ctxs,
        "total": tuned + tier + ver + size + ctxs,
    }


def score(model):
    return score_parts(model)["total"]


# ══════════════════════════════════════════════════════════════════════════
# 4. SELECTION
# ══════════════════════════════════════════════════════════════════════════

def pick(models, per_publisher=1, max_total=0, min_score=10.0):
    """
    Best `per_publisher` model(s) from each publisher.

    min_score drops sole-member junk: a publisher with exactly one weak entry
    (an experimental or non-flagship model) would otherwise be auto-promoted
    simply because it had no competition in its own group.
    """
    groups = {}
    for m in models:
        groups.setdefault(publisher(m["model_id"]), []).append(m)

    for members in groups.values():
        members.sort(key=score, reverse=True)

    chosen = []
    for members in groups.values():
        for m in members[:per_publisher]:
            if score(m) >= min_score:
                chosen.append(m)

    chosen.sort(key=score, reverse=True)
    if max_total and len(chosen) > max_total:
        chosen = chosen[:max_total]
    return chosen, groups


# ══════════════════════════════════════════════════════════════════════════
# 5. FETCHING
# ══════════════════════════════════════════════════════════════════════════

async def fetch_openai_style(client, url, key, provider):
    if not key:
        print(f"  ! no API key for {provider}, skipping")
        return []
    try:
        r = await client.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        out = []
        for m in r.json().get("data", []):
            caps = m.get("capabilities") or {}
            if caps and not caps.get("completion_chat", False):
                continue
            if not m.get("active", True):
                continue
            out.append({
                "model_id": m.get("id", ""),
                "provider": provider,
                "context": m.get("context_window") or m.get("max_context_length") or 0,
            })
        return out
    except Exception as e:
        print(f"  ! {provider} fetch failed: {e}")
        return []


async def fetch_gemini(client, key):
    if not key:
        print("  ! no API key for gemini, skipping")
        return []
    try:
        out, token = [], None
        while True:
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            r = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": key}, params=params, timeout=30)
            r.raise_for_status()
            body = r.json()
            for m in body.get("models", []):
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                out.append({
                    "model_id": m.get("name", "").replace("models/", ""),
                    "provider": "gemini",
                    "context": m.get("inputTokenLimit", 0),
                })
            token = body.get("nextPageToken")
            if not token:
                break
        return out
    except Exception as e:
        print(f"  ! gemini fetch failed: {e}")
        return []


async def fetch_all(only=None):
    urls = {
        "groq": "https://api.groq.com/openai/v1/models",
        "nvidia": "https://integrate.api.nvidia.com/v1/models",
        "mistral": "https://api.mistral.ai/v1/models",
    }
    res = {}
    async with httpx.AsyncClient() as client:
        for p in PROVIDERS:
            if only and p != only:
                continue
            res[p] = (await fetch_gemini(client, KEYS[p]) if p == "gemini"
                      else await fetch_openai_style(client, urls[p], KEYS[p], p))
    return res


def load_from_file(path, only=None):
    """Reuse models_report.json from list_models.py — instant, no API calls."""
    raw = json.load(open(path, encoding="utf-8"))
    res = {}
    for prov, models in raw.items():
        if only and prov != only:
            continue
        res[prov] = [
            {"model_id": m["model_id"], "provider": prov, "context": m.get("context") or 0}
            for m in models
        ]
    return res


# ══════════════════════════════════════════════════════════════════════════
# 6. REPORT
# ══════════════════════════════════════════════════════════════════════════

def report(fetched, args):
    final, grand = {}, 0

    for prov in PROVIDERS:
        if prov not in fetched:
            continue

        raw = fetched[prov]
        chat = [m for m in raw if is_chat(m["model_id"])]
        clean = dedupe(chat)
        chosen, groups = pick(clean, args.per_publisher, args.max, args.min_score)

        final[prov] = [m["model_id"] for m in chosen]
        grand += len(chosen)

        print()
        print("=" * 78)
        print(f" {prov.upper()}   {len(raw)} fetched -> {len(chat)} chat -> "
              f"{len(clean)} deduped -> {len(groups)} publishers -> {len(chosen)} PICKED")
        print("=" * 78)

        if args.show_all:
            picked_ids = {m["model_id"] for m in chosen}
            for pub in sorted(groups):
                members = groups[pub]
                print(f"\n  [{pub}]  {len(members)} candidate(s)")
                for m in members:
                    sp = score_parts(m)
                    mark = ">>" if m["model_id"] in picked_ids else "  "
                    print(f"   {mark} {sp['total']:7.1f}  {m['model_id']}")
                    print(f"          tuned={sp['tuned']:.0f} tier={sp['tier']:+.0f} "
                          f"ver={sp['version']:.0f} size={sp['size']:.1f}"
                          f"({sp['size_b']:g}B) ctx={sp['context']:.1f}")
        else:
            for m in chosen:
                sp = score_parts(m)
                print(f"   {sp['total']:7.1f}  {m['model_id']:54} [{publisher(m['model_id'])}]")

    print()
    print("=" * 78)
    print(f" TOTAL PICKED: {grand} models")
    print(f" Approx benchmark cost: {grand} x 34 questions = ~{grand * 34:,} API calls")
    print("=" * 78)
    print(f" per-publisher={args.per_publisher}   "
          f"max-per-provider={args.max or 'none'}   min-score={args.min_score}")
    print()

    if args.save:
        json.dump(final, open(args.save, "w", encoding="utf-8"), indent=2)
        print(f" Saved -> {args.save}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-publisher", type=int, default=1,
                    help="models kept per publisher (default 1 = its flagship)")
    ap.add_argument("--max", type=int, default=0,
                    help="cap per provider AFTER selection (default 0 = no cap; "
                         "let the benchmark decide the top 10)")
    ap.add_argument("--min-score", type=float, default=10.0,
                    help="drop weak sole-member picks below this score")
    ap.add_argument("--provider", choices=PROVIDERS)
    ap.add_argument("--show-all", action="store_true",
                    help="show every candidate with its score breakdown")
    ap.add_argument("--from-file", help="read models_report.json instead of calling APIs")
    ap.add_argument("--save", help="write chosen model ids to a JSON file")
    args = ap.parse_args()

    if args.from_file:
        if not os.path.exists(args.from_file):
            print(f"File not found: {args.from_file}")
            sys.exit(1)
        fetched = load_from_file(args.from_file, args.provider)
    else:
        fetched = asyncio.run(fetch_all(args.provider))

    report(fetched, args)


if __name__ == "__main__":
    main()
