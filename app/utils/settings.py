from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration in one place. Loaded from .env.
    DB_Configuration is a standard postgresql:// URL — database.py converts it
    to the async (asyncpg) form internally.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Database ────────────────────────────────────────────────────────────
    DB_Configuration: str

    # ── Redis (used as Celery broker/backend AND app cache) ─────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Provider API Keys ───────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""

    # ── Benchmark sample sizes (per category) ───────────────────────────────
    # Small on purpose so a provider run finishes fast and stays inside free
    # tiers. Raise once a clean run completes.
    # Deliberately SMALL. With 35 models and free-tier pacing (some Mistral
    # models allow one request every 33 seconds), large samples make a full
    # run take days. 5 per category = 20 questions per model = ~700 calls
    # total, which finishes in about an hour and still ranks models sensibly.
    # Raise these once you have a complete leaderboard to show.
    KNOWLEDGE_SAMPLE_SIZE: int = 5
    REASONING_SAMPLE_SIZE: int = 5
    MATH_SAMPLE_SIZE: int = 5
    CODING_SAMPLE_SIZE: int = 5

    # ── Retry / timeout ─────────────────────────────────────────────────────
    # ONE place controls retries now. Kept LOW to avoid the retry-storm problem.
    # 4xx errors (like a 404 "model not found") are NEVER retried — see
    # provider_clients.py. Only timeouts / 5xx / 429 get a single retry.
    API_MAX_RETRIES: int = 1          # 1 retry = 2 attempts total, max
    API_RETRY_DELAY: float = 2.0
    API_TIMEOUT_SECONDS: float = 45.0

    # Minimum output-token budget for Gemini/Gemma. They are THINKING models and
    # thinking can't be disabled on Gemma, so a small cap truncates the thinking
    # before the answer is emitted (finishReason=MAX_TOKENS, empty answer). With
    # this floor thinking finishes and the model answers; it STOPS when done, so
    # the ceiling doesn't waste tokens. 512/1024 was starving it → zero scores.
    GEMINI_MIN_OUTPUT_TOKENS: int = 8192

    # Gemini needs a LONGER per-request timeout than the 45s default. It generates
    # at ~30 tokens/sec and a hard, thinking-heavy question can run well past 45s,
    # which ReadTimeouts and scores 0. 120s covers ~3600 thinking tokens — enough
    # for the hardest benchmark questions. Only Gemini uses this; fast providers
    # keep the 45s default.
    GEMINI_TIMEOUT_SECONDS: float = 120.0

    # Gemini model preference (free tier). The scoring heuristic keeps choosing
    # gemini-*-pro-preview (rate-limited → 429 on every call) and gemma-4 (a
    # thinking model so slow it ReadTimeouts, 60-95s+ per hard question). Flash
    # models answer correctly in ~8s, so we prefer them for Gemini. Ordered by
    # preference; preflight keeps whichever your key actually has, and the best
    # one is benchmarked. Empty list → use the normal per-publisher heuristic.
    GEMINI_PREFERRED_MODELS: list[str] = [
        "gemini-3.5-flash",
        "gemini-flash-latest",
        "gemini-3-flash-preview",
        "gemini-2.0-flash",
    ]

    # NVIDIA hosts 550B-675B models that take 20-45s+ per answer and ReadTimeout
    # at the 45s default. Give NVIDIA a longer per-request timeout (like Gemini)
    # so the big, strong models actually finish instead of scoring 0.
    NVIDIA_TIMEOUT_SECONDS: float = 120.0

    # "Only good and big models." NVIDIA's catalog is full of small 7-15B models
    # that aren't competitive. Keep only models with at least this many billion
    # parameters (or a top-tier name like -large/-ultra/-pro that doesn't expose
    # a number). 0 disables the filter. This is what drops the weak small models
    # while keeping gpt-oss-120b, llama-70b, nemotron-550b, qwen-397b, etc.
    NVIDIA_MIN_PARAM_B: float = 30.0

    # Preflight uses a SHORT timeout so a chronically-slow model (some NVIDIA
    # free-tier models take >120s even to say "OK" due to server-side queueing)
    # is dropped in seconds instead of stalling the whole run. Models that pass
    # preflight still get the full provider timeout for real benchmark questions.
    PREFLIGHT_TIMEOUT_SECONDS: float = 30.0

    # ── Model selection ─────────────────────────────────────────────────────
    # FLAGSHIP MODE (default ON): instead of benchmarking every model, keep
    # only the best model per publisher/family. Turns ~120 candidates into a
    # top-10 style shortlist. Set False to benchmark everything.
    FLAGSHIPS_ONLY: bool = True

    # How many models to keep per publisher (meta, google, qwen, deepseek...).
    # 1 = just their flagship. 2 = flagship + runner-up.
    MODELS_PER_PUBLISHER: int = 1

    # Overall cap per provider AFTER flagship selection. 0 = no cap.
    # 10 gives you a clean top-10 leaderboard per provider.
    # 0 = no cap. The BENCHMARK decides the top 10, not a heuristic.
    MAX_MODELS_PER_PROVIDER: int = 0

    # Drop weak sole-member picks (a publisher with one non-flagship entry).
    MIN_MODEL_SCORE: float = 10.0

    # When a flagship fails preflight (e.g. Gemini Pro is no longer free),
    # try this many alternatives from the same publisher before giving up.
    BACKFILL_ATTEMPTS: int = 2

    # Preflight sends 1 tiny request per model to drop dead/unusable ids
    # BEFORE the benchmark starts. This is what prevents failure storms.
    PREFLIGHT_CONCURRENCY: int = 2   # low: preflight is rate-limited too

    # Hard ceiling on a single provider run, in seconds.
    # NVIDIA has ~97 models; at ~34 questions each this needs hours, so the
    # old 30-minute limit would kill the task mid-run. 6h default.
    RUN_TIME_LIMIT: int = 21600

    # A second benchmark for the same provider is rejected while an existing run
    # started within this many seconds is still 'running'. Stops double-clicking
    # "Execute" from launching duplicate runs that fight over the rate limit and
    # deadlock. After this window a run orphaned by a killed worker no longer
    # blocks new runs. 15 min comfortably covers a normal single-provider run.
    DUPLICATE_RUN_WINDOW_SECONDS: int = 900

    # Number of entries the /leaderboard endpoint returns. 0 = all.
    LEADERBOARD_TOP_N: int = 10

    # ── Cache TTL (seconds) ─────────────────────────────────────────────────
    LEADERBOARD_CACHE_TTL: int = 120

    # ── Category weights (must sum to 1.0) ──────────────────────────────────
    WEIGHT_KNOWLEDGE: float = 0.30
    WEIGHT_REASONING: float = 0.25
    WEIGHT_MATH: float = 0.20
    WEIGHT_CODING: float = 0.25


settings = Settings()

# The 4 providers this app benchmarks, in a fixed order.
PROVIDERS = ["groq", "nvidia", "gemini", "mistral"]
