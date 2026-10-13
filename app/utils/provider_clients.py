"""
All LLM provider communication lives here.

THE BIG FIX (this is what caused the endless 404 loop before):
  - A 404 / 400 / 401 / 403 means "this will NEVER work" (bad model id, bad key).
    We now FAIL IMMEDIATELY on any 4xx. We do NOT retry it. Retrying a permanent
    error is what flooded the logs with thousands of identical 404s.
  - Only transient errors (timeout, network drop, 5xx server error, 429 rate
    limit) get retried, and only ONCE (settings.API_MAX_RETRIES = 1).

Model lists are fetched LIVE from each provider's /models endpoint wherever one
exists, so we never hardcode a model id that has since been deleted (the other
thing that caused the 404s). A small verified fallback is used only if the live
fetch fails.
"""

import re
import time
import asyncio
import httpx

from app.utils.settings import settings
from app.utils.logger import get_logger
from app.utils.rate_limits import limiter

logger = get_logger("providers")


# A 4xx we should give up on instantly. 429 is deliberately excluded — that one
# is transient and worth a short wait + single retry.
class PermanentAPIError(Exception):
    """Raised on a 4xx (not 429). Never retried."""


# ── One HTTP client per event loop ──────────────────────────────────────────
# Celery (--pool=threads) calls asyncio.run() per task, creating a new event
# loop each time. httpx clients bind to the loop they were made on, so we key
# one client per loop and close it when the task's loop ends.
_clients: dict[int, httpx.AsyncClient] = {}


def _client() -> httpx.AsyncClient:
    key = id(asyncio.get_event_loop())
    if key not in _clients:
        _clients[key] = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.API_TIMEOUT_SECONDS),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        )
    return _clients[key]


async def close_client():
    """Close the HTTP client tied to the current event loop."""
    key = id(asyncio.get_event_loop())
    client = _clients.pop(key, None)
    if client is not None:
        await client.aclose()


# ── Core call with a SANE retry policy ──────────────────────────────────────

class BudgetExhausted(Exception):
    """Provider request budget spent. Stop calling this provider."""


async def _post(url: str, headers: dict, payload: dict, provider: str, model: str,
                timeout: float | None = None) -> dict:
    client = _client()
    attempts = settings.API_MAX_RETRIES + 1
    # Per-request timeout override (the shared client's default is used when None).
    # Slow "thinking" providers like Gemini need longer than the 45s default or a
    # hard, thinking-heavy question ReadTimeouts and scores 0.
    post_kwargs = {"headers": headers, "json": payload}
    if timeout is not None:
        post_kwargs["timeout"] = timeout

    for attempt in range(1, attempts + 1):

        # GATE: wait for this model's pacing interval, and stop entirely if the
        # provider's request budget is gone. Without this the free tiers 429 and
        # every score comes back 0.
        if not await limiter.acquire(provider, model):
            raise BudgetExhausted(f"{provider} request budget exhausted")

        try:
            resp = await client.post(url, **post_kwargs)

            # 429 → transient, but the published limit was too optimistic.
            # Honour retry-after and permanently slow this model down.
            if resp.status_code == 429:
                retry_after = None
                for h in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
                    v = resp.headers.get(h)
                    if v:
                        try:
                            retry_after = float(re.sub(r"[^0-9.]", "", v) or 0)
                            break
                        except ValueError:
                            pass
                limiter.penalise(provider, model, retry_after)

                if attempt >= attempts:
                    raise PermanentAPIError(f"{provider}:{model} rate limited (429)")
                await asyncio.sleep(min(retry_after or settings.API_RETRY_DELAY, 30.0))
                continue

            # Any other 4xx → PERMANENT. Never retried.
            if 400 <= resp.status_code < 500:
                detail = resp.text[:150].replace("\n", " ")
                logger.error(f"{resp.status_code} {provider} {model} — permanent, not retrying. {detail}")
                raise PermanentAPIError(f"{provider}:{model} returned {resp.status_code}")

            resp.raise_for_status()
            return resp.json()

        except (PermanentAPIError, BudgetExhausted):
            raise

        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as e:
            if attempt >= attempts:
                logger.error(f"{provider} {model} failed after {attempt} attempts: {type(e).__name__}")
                raise
            logger.warning(f"{provider} {model} transient ({type(e).__name__}), retrying once")
            await asyncio.sleep(settings.API_RETRY_DELAY)

    return {}


# ── Per-provider call wrappers (all OpenAI-compatible except Gemini) ─────────

def _messages(prompt: str, system_prompt: str | None):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _content_to_text(content) -> str:
    """Normalise an OpenAI-style `message.content` to a plain string.

    Most providers return a string. But Mistral's reasoning models (magistral)
    return a LIST of typed blocks — a `thinking` block plus a `text` block:
        [ {"type":"thinking","thinking":[{"type":"text","text":"..."}]},
          {"type":"text","text":"the answer"} ]
    The old code assumed a string, so a list scored as empty → magistral got
    0 in every category. We pull the answer from the `text` block(s), skipping
    the chain-of-thought, and fall back to the thinking text only if that's all
    there is (e.g. output truncated mid-thought)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        answer, thinking = [], []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                answer.append(block["text"])
            elif btype == "thinking":
                for t in block.get("thinking", []) or []:
                    if isinstance(t, dict) and isinstance(t.get("text"), str):
                        thinking.append(t["text"])
        return "".join(answer) if answer else "".join(thinking)
    return ""


async def _openai_style(base_url: str, api_key: str, provider: str, model: str,
                        prompt: str, system_prompt: str | None,
                        max_tokens: int, temperature: float,
                        timeout: float | None = None) -> tuple[str, int]:
    start = time.time()
    data = await _post(
        url=base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload={
            "model": model,
            "messages": _messages(prompt, system_prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        provider=provider,
        model=model,
        timeout=timeout,
    )
    latency = int((time.time() - start) * 1000)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _content_to_text(content), latency


async def call_gemini(model: str, prompt: str, system_prompt: str | None,
                      max_tokens: int, temperature: float,
                      timeout: float | None = None) -> tuple[str, int]:
    name = model.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent"
    text_in = (system_prompt + "\n\n" + prompt) if system_prompt else prompt

    # Gemini/Gemma are THINKING models and thinking can't be disabled on Gemma
    # (the API rejects thinkingConfig). With a small budget the thinking gets
    # truncated at the cap (finishReason=MAX_TOKENS) BEFORE the model ever emits
    # an answer part — that's what zeroed knowledge/reasoning/coding. Given ample
    # headroom, thinking finishes on its own (~200 tokens) and the answer follows,
    # and the model STOPS when done, so a high ceiling costs nothing extra. We
    # therefore floor Gemini's output budget; the per-category caps (512/1024)
    # are far too low for a reasoning model.
    effective_max = max(max_tokens, settings.GEMINI_MIN_OUTPUT_TOKENS)

    start = time.time()
    data = await _post(
        url=url,
        headers={"x-goog-api-key": settings.GEMINI_API_KEY, "Content-Type": "application/json"},
        payload={
            "contents": [{"role": "user", "parts": [{"text": text_in}]}],
            "generationConfig": {"maxOutputTokens": effective_max, "temperature": temperature},
        },
        provider="gemini",
        model=model,
        timeout=timeout or settings.GEMINI_TIMEOUT_SECONDS,
    )
    latency = int((time.time() - start) * 1000)
    text = ""
    cands = data.get("candidates", [])
    if cands:
        parts = cands[0].get("content", {}).get("parts", [])
        # Gemini/Gemma "thinking" models return the chain-of-thought as a part
        # flagged `thought: true` and the ACTUAL answer as a SEPARATE non-thought
        # part. The old code read parts[0] only, which is the thought — so the
        # real answer (the \boxed{} value, the code, the final letter) in a later
        # part was thrown away, zeroing math/coding and corrupting the rest.
        # Take every non-thought text part; fall back to the thought text only if
        # that's all there is (e.g. output truncated mid-thinking).
        answer = [p.get("text", "") for p in parts if p.get("text") and not p.get("thought")]
        text = "".join(answer) if answer else "".join(p.get("text", "") for p in parts if p.get("text"))
    return text, latency


async def call_model(provider: str, model: str, prompt: str,
                     system_prompt: str | None = None,
                     max_tokens: int = 512, temperature: float = 0.0,
                     timeout: float | None = None) -> tuple[str, int]:
    """Unified entry point — routes to the right provider. Returns (text, latency_ms).

    `timeout` overrides the per-provider default (used by preflight to fail fast
    on slow models). When None, each provider uses its own configured timeout."""

    if provider == "groq":
        return await _openai_style("https://api.groq.com/openai/v1/chat/completions",
                                   settings.GROQ_API_KEY, "groq", model,
                                   prompt, system_prompt, max_tokens, temperature,
                                   timeout=timeout)
    if provider == "nvidia":
        return await _openai_style("https://integrate.api.nvidia.com/v1/chat/completions",
                                   settings.NVIDIA_API_KEY, "nvidia", model,
                                   prompt, system_prompt, max_tokens, temperature,
                                   timeout=timeout or settings.NVIDIA_TIMEOUT_SECONDS)
    if provider == "mistral":
        return await _openai_style("https://api.mistral.ai/v1/chat/completions",
                                   settings.MISTRAL_API_KEY, "mistral", model,
                                   prompt, system_prompt, max_tokens, temperature,
                                   timeout=timeout)
    if provider == "gemini":
        return await call_gemini(model, prompt, system_prompt, max_tokens, temperature,
                                 timeout=timeout)

    raise ValueError(f"Unknown provider: {provider}")


# ── Model discovery ─────────────────────────────────────────────────────────
#
# NOTHING IS HARDCODED. Every model id comes live from the provider's own
# /models endpoint. We then apply three rule-based passes:
#
#   1. PATTERN FILTER  — drop models that clearly aren't text chat
#                        (audio, embeddings, image/video, safety, OCR, reward,
#                        agentic CLI, fill-in-middle, robotics, etc.)
#   2. DEDUPE          — providers return literal duplicates (Mistral does),
#                        and "-latest" aliases duplicate their dated twin
#                        (codestral-latest == codestral-2508).
#   3. PREFLIGHT PROBE — one tiny request per model. Anything that errors
#                        (dead id, no access, wrong modality) is dropped before
#                        the benchmark starts. This is what stops a bad model
#                        from generating hundreds of failures mid-run.

# Substrings marking a model as NOT plain text chat. Rules, not model names.
_NON_CHAT = (
    # speech / audio
    "whisper", "tts", "speech", "audio", "voice", "orpheus", "voxtral",
    "realtime", "transcribe", "live", "omni", "canary", "parakeet",
    # embeddings / retrieval / rerank
    "embed", "rerank", "retrieval", "bge-", "nemoretriever", "arctic-embed",
    # safety / moderation / reward
    "guard", "moderation", "safety", "shield", "topic-control", "reward",
    # image / video / music
    "image", "imagen", "diffusion", "stable-", "sdxl", "clip", "nvclip",
    "veo", "lyria", "nano-banana", "vision", "-vl-", "vlm",
    # document / parsing / specialised
    "ocr", "parse", "deplot", "fuyu", "kosmos", "neva", "vila", "gliner",
    "pii", "robotics", "computer-use", "-fim", "vibe-cli", "code-agent",
    "riva", "translate",
    # Gemini models that only support the Interactions API, not generateContent —
    # they return 400 on every call, so selecting them just wastes preflight and
    # backfill time. (Observed: deep-research-*, antigravity-*.)
    "deep-research", "antigravity",
)


def _is_chat_model(model_id: str) -> bool:
    low = model_id.lower()
    return not any(s in low for s in _NON_CHAT)


_DATED = re.compile(r"^(?P<base>.+)-\d{3,6}$")


def _dedupe(models: list[dict]) -> list[dict]:
    """Remove exact duplicates, then drop dated variants when a '-latest'
    alias for the same base exists (codestral-2508 vs codestral-latest)."""

    seen: dict[str, dict] = {}
    for m in models:
        seen.setdefault(m["model_id"], m)      # exact duplicate removal
    models = list(seen.values())

    latest_bases = {
        m["model_id"][: -len("-latest")]
        for m in models if m["model_id"].endswith("-latest")
    }

    out = []
    for m in models:
        mid = m["model_id"]
        match = _DATED.match(mid)
        if match and match.group("base") in latest_bases:
            continue                            # superseded by the -latest alias
        out.append(m)
    return out


async def _probe(provider: str, model: str) -> bool:
    """One tiny call to confirm the model is reachable. Cheap insurance.

    We give the probe 64 tokens (not 5): reasoning models spend their token
    budget on internal reasoning and can return an EMPTY content field even
    though the call succeeded — 5 tokens guaranteed empty output and wrongly
    dropped them (this is what killed the gpt-oss preflight). A successful HTTP
    call now counts as alive even if the body came back empty; only an actual
    exception (dead id, no access, wrong modality) drops the model.
    """
    try:
        await call_model(
            provider=provider, model=model, prompt="Reply with the word OK.",
            max_tokens=64, temperature=0.0,
            timeout=settings.PREFLIGHT_TIMEOUT_SECONDS,
        )
        return True
    except Exception as e:
        logger.warning(f"preflight drop {provider}/{model}: {type(e).__name__}")
        return False


async def _preflight(provider: str, models: list[dict]) -> list[dict]:
    """Probe models concurrently (bounded) and keep only the ones that answer."""
    sem = asyncio.Semaphore(settings.PREFLIGHT_CONCURRENCY)

    async def check(m):
        async with sem:
            return m if await _probe(provider, m["model_id"]) else None

    results = await asyncio.gather(*(check(m) for m in models))
    keep = [m for m in results if m]
    logger.info(f"{provider}: preflight kept {len(keep)}, dropped {len(models) - len(keep)}")
    return keep


async def _preflight_with_backfill(provider: str, chosen: list[dict],
                                   groups: dict[str, list[dict]]) -> list[dict]:
    """
    Preflight the selected flagships. When one fails, promote the NEXT best
    model from that same publisher and probe that instead.

    This matters concretely: Google removed Pro models from the Gemini free
    tier in 2026, so gemini-*-pro will fail on a free key. Without backfill
    the whole Gemini publisher group would vanish from the leaderboard.
    Instead we fall back to that publisher's next model (a Flash variant).
    """
    from app.utils.selection import publisher

    working = await _preflight(provider, chosen)
    ok_ids = {m["model_id"] for m in working}
    failed = [m for m in chosen if m["model_id"] not in ok_ids]

    for bad in failed:
        pub = publisher(bad["model_id"])
        alternatives = [m for m in groups.get(pub, [])
                        if m["model_id"] != bad["model_id"] and m["model_id"] not in ok_ids]

        for alt in alternatives[:settings.BACKFILL_ATTEMPTS]:
            logger.info(f"{provider}: {bad['model_id']} failed — trying {alt['model_id']}")
            if await _probe(provider, alt["model_id"]):
                working.append(alt)
                ok_ids.add(alt["model_id"])
                break

    return working


async def _fetch_openai_style_models(base_url: str, api_key: str, provider: str) -> list[dict]:
    if not api_key:
        logger.warning(f"No API key for {provider} — skipping")
        return []
    try:
        resp = await _client().get(base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0)
        resp.raise_for_status()
        out = []
        for m in resp.json().get("data", []):
            mid = m.get("id", "")
            caps = m.get("capabilities") or {}
            # Mistral states capabilities explicitly — trust it over guessing
            if caps and not caps.get("completion_chat", False):
                continue
            if not _is_chat_model(mid) or not m.get("active", True):
                continue
            out.append({"model_id": mid, "name": mid, "provider": provider})
        return out
    except Exception as e:
        logger.error(f"{provider} model fetch failed: {e}")
        return []


async def _fetch_gemini_models() -> list[dict]:
    if not settings.GEMINI_API_KEY:
        return []
    try:
        out, token = [], None
        while True:                                    # Gemini paginates
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            resp = await _client().get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": settings.GEMINI_API_KEY},
                params=params, timeout=30.0,
            )
            resp.raise_for_status()
            body = resp.json()
            for m in body.get("models", []):
                mid = m.get("name", "").replace("models/", "")
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                if not _is_chat_model(mid):
                    continue
                out.append({"model_id": mid, "name": m.get("displayName", mid), "provider": "gemini"})
            token = body.get("nextPageToken")
            if not token:
                break
        return out
    except Exception as e:
        logger.error(f"gemini model fetch failed: {e}")
        return []


async def get_models_for_provider(provider: str, preflight: bool = True) -> list[dict]:
    """ALL usable chat models for one provider. No cap, no hardcoded ids."""

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

    if provider in urls:
        models = await _fetch_openai_style_models(urls[provider], keys[provider], provider)
    elif provider == "gemini":
        models = await _fetch_gemini_models()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    before = len(models)
    models = _dedupe(models)
    logger.info(f"{provider}: {before} chat models, {len(models)} after dedupe")

    # NVIDIA: keep only "good and big" models. Drop the small 7-15B entries and
    # keep models with >= NVIDIA_MIN_PARAM_B billion params OR a top-tier name
    # (-large/-ultra/-pro/-max) that doesn't put a number in the id. This keeps
    # gpt-oss-120b, llama-70b, nemotron-550b, qwen-397b, deepseek-pro, etc. and
    # drops sea-lion-7b, granite-8b, solar-10.7b, starcoder-15b, and the like.
    if provider == "nvidia" and settings.NVIDIA_MIN_PARAM_B > 0 and models:
        from app.utils.selection import parse_size, tier_bonus
        kept = [m for m in models
                if parse_size(m["model_id"]) >= settings.NVIDIA_MIN_PARAM_B
                or tier_bonus(m["model_id"]) >= 40]
        if kept:
            logger.info(f"nvidia: {len(kept)}/{len(models)} models kept after "
                        f"'big & good' filter (>= {settings.NVIDIA_MIN_PARAM_B:.0f}B or top-tier)")
            models = kept

    # Gemini free-tier override: the heuristic keeps picking gemini-*-pro-preview
    # (rate-limited to 429 on every call) and gemma (a THINKING model so slow it
    # ReadTimeouts on hard questions, 60-95s+ each). Flash models answer the same
    # questions correctly in ~8s. So for Gemini we prefer a curated fast list;
    # preflight keeps whichever the key actually has, and we benchmark the best
    # one. Empty list or none available → fall through to the normal heuristic.
    if provider == "gemini" and settings.GEMINI_PREFERRED_MODELS:
        order = {mid: i for i, mid in enumerate(settings.GEMINI_PREFERRED_MODELS)}
        preferred = sorted((m for m in models if m["model_id"] in order),
                           key=lambda m: order[m["model_id"]])
        if preferred:
            logger.info(f"gemini: preferring fast models "
                        f"{[m['model_id'] for m in preferred]}")
            if preflight:
                preferred = await _preflight("gemini", preferred)
            if preferred:
                chosen = preferred[:max(1, settings.MODELS_PER_PUBLISHER)]
                logger.info(f"gemini: using {[m['model_id'] for m in chosen]}")
                return chosen
            logger.warning("gemini: no preferred fast model passed preflight — "
                           "falling back to the normal heuristic")

    # Flagship selection: best model per publisher, so we benchmark strong
    # distinct models instead of many near-duplicates.
    groups = {}
    if settings.FLAGSHIPS_ONLY and models:
        from app.utils.selection import pick_flagships
        models, groups = pick_flagships(
            models,
            per_group=settings.MODELS_PER_PUBLISHER,
            max_total=settings.MAX_MODELS_PER_PROVIDER,
            min_score=settings.MIN_MODEL_SCORE,
        )
        logger.info(f"{provider}: {len(models)} flagships selected")
    else:
        cap = settings.MAX_MODELS_PER_PROVIDER
        if cap and len(models) > cap:
            models = models[:cap]

    if preflight and models:
        models = await _preflight_with_backfill(provider, models, groups)

    return models


async def get_all_models() -> list[dict]:
    """Every usable model across all 4 providers (for the /models endpoint).
    Preflight is skipped here — this is a fast listing, not a benchmark."""
    from app.utils.settings import PROVIDERS
    out = []
    for p in PROVIDERS:
        out.extend(await get_models_for_provider(p, preflight=False))
    return out
