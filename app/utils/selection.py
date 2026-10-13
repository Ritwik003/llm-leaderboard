"""
FLAGSHIP SELECTION — pick only the top model(s) per publisher/family.

The goal: a top-10 style leaderboard, not a 120-model marathon. Instead of
benchmarking every Llama variant NVIDIA hosts, take Meta's best one, Google's
best one, Qwen's best one, and so on.

NOTHING IS HARDCODED. No model names appear anywhere in this file. Selection
works by parsing signals that are already encoded in the model id itself:

    meta/llama-3.3-70b-instruct
     |        |    |    |
     |        |    |    +-- instruction-tuned  -> big bonus (base models score badly)
     |        |    +------- 70 billion params  -> bigger is usually stronger
     |        +------------ version 3.3        -> newer is usually stronger
     +--------------------- publisher "meta"   -> the grouping key

Models are grouped by publisher (the part before "/") or, when there's no
prefix, by model family (the leading word: llama, gemini, mistral, gemma...).
Within each group they're ranked and only the top N survive.

HONEST CAVEAT: this is a heuristic, not ground truth. Parameter count is a
rough proxy for quality and a newer small model can beat an older large one.
It's here to cut 120 candidates down to a sensible shortlist — the benchmark
itself is what actually decides the ranking.
"""

import re

from app.utils.logger import get_logger

logger = get_logger("selection")


# ── Signal parsing ──────────────────────────────────────────────────────────

# Matches a parameter count like "70b", "405b", "8x22b", "1.5b".
# The negative lookbehind for "a" skips MoE *active* counts ("...-a17b"), so
# "qwen3.5-397b-a17b" reads as 397B total rather than 17B active.
_SIZE = re.compile(r"(?<![a-z])(\d+(?:\.\d+)?)\s*x?\s*(\d+(?:\.\d+)?)?\s*b(?![a-z])")

def parse_size(model_id: str) -> float:
    """Largest parameter count in billions found in the id. 0 if none."""
    best = 0.0
    for m in _SIZE.finditer(model_id.lower()):
        a, b = m.group(1), m.group(2)
        val = float(a) * float(b) if b else float(a)   # "8x22b" -> 8 * 22
        best = max(best, val)
    return best


# Tier words matched as whole TOKENS, not substrings. Substring matching was a
# real bug: "minimax" contains "max", so minimax-m3 wrongly got the flagship bonus.
_TIER_TOP = {"large", "pro", "ultra", "max", "opus", "xl", "huge", "premier", "flagship"}
_TIER_MID = {"medium", "super", "plus", "standard"}
_TIER_LOW = {"mini", "nano", "tiny", "lite", "xs", "small", "micro", "edge", "instant"}

_TUNED = {"instruct", "it", "chat", "versatile", "instant", "latest",
          "preview", "pro", "flash", "medium", "small", "large"}


def _tokens(model_id: str) -> set:
    return set(re.split(r"[-_/.]+", model_id.lower()))


def parse_version(model_id: str) -> float:
    """
    Version from the id. Two patterns, because vendors name models differently:
      A) attached to family   llama-3.3, qwen3.5, gemini-3.1
      B) letter-prefixed      kimi-k2.6, minimax-m3, deepseek-v4

    Pattern B matters: without it kimi-k2.6 and glm-5.2 parse as version 0 and
    score near-zero despite being strong models.

    Parameter counts and dates are stripped FIRST, otherwise "gpt-oss-20b"
    reads as version 20 and beats "gpt-oss-120b" — backwards.
    """
    name = model_id.split("/")[-1].lower()
    name = re.sub(r"\d+(?:\.\d+)?\s*x?\s*\d*\s*b(?![a-z])", " ", name)
    name = re.sub(r"a\d+b", " ", name)
    name = re.sub(r"\b\d{3,6}\b", " ", name)

    cands = []
    m = re.match(r"[a-z]+-?(\d+(?:\.\d+)?)", name)
    if m:
        cands.append(m.group(1))
    for m in re.finditer(r"\b[a-z]{1,4}-?(\d+(?:\.\d+)?)\b", name):
        cands.append(m.group(1))

    best = 0.0
    for c in cands:
        try:
            v = float(c)
        except ValueError:
            continue
        if v <= 100:
            best = max(best, v)
    return best


def tier_bonus(model_id: str) -> float:
    tk = _tokens(model_id)
    if tk & _TIER_TOP:
        return 40.0
    if tk & _TIER_MID:
        return 18.0
    if tk & _TIER_LOW:
        return -30.0
    return 0.0


def is_tuned(model_id: str) -> bool:
    return bool(_tokens(model_id) & _TUNED)


def publisher(model_id: str) -> str:
    """Grouping key: publisher if namespaced, else model family."""
    low = model_id.lower()
    if "/" in low:
        org = low.split("/")[0]
        return org.split("-")[0] if ("-" in org and not org[0].isdigit()) else org
    if low.startswith("open-"):
        low = low[5:]
    m = re.match(r"[a-z]+", low)
    return m.group(0) if m else low


def rank_score(model: dict) -> float:
    """Higher = more likely this publisher's flagship."""
    mid = model["model_id"]
    ctx = float(model.get("context") or 0)

    # Tuning is positive but NOT dominant: kimi-k2.6, glm-5.2 and yi-large
    # don't say "instruct" and a large bonus unfairly buried them.
    score = 25.0 if is_tuned(mid) else 0.0
    score += tier_bonus(mid)
    score += parse_version(mid) * 10.0

    size_b = parse_size(mid)
    if size_b:
        score += (min(size_b, 1000.0) ** 0.5) * 3.0
    if ctx:
        score += min(ctx / 100_000.0, 10.0)
    return score


def pick_flagships(models: list[dict], per_group: int = 1, max_total: int = 0,
                   min_score: float = 10.0):
    """
    Best `per_group` model(s) per publisher.

    Returns (chosen, groups). `groups` is kept so preflight can BACKFILL from
    the same publisher when the top pick turns out to be unavailable.

    min_score drops sole-member junk: a publisher with one weak entry would
    otherwise be auto-promoted just because nothing competed with it.
    """
    if not models:
        return [], {}

    groups: dict[str, list[dict]] = {}
    for m in models:
        groups.setdefault(publisher(m["model_id"]), []).append(m)

    for members in groups.values():
        members.sort(key=rank_score, reverse=True)

    chosen = []
    for key, members in groups.items():
        for m in members[:per_group]:
            if rank_score(m) >= min_score:
                chosen.append(m)
                logger.info(f"  {key}: {m['model_id']} ({rank_score(m):.0f})")

    chosen.sort(key=rank_score, reverse=True)
    if max_total and len(chosen) > max_total:
        chosen = chosen[:max_total]
    return chosen, groups
