# LLM Benchmark Leaderboard

Benchmarks free-tier LLMs from **Groq, NVIDIA, Gemini and Mistral**, one
provider at a time, and ranks them on a single leaderboard.

Built around one hard constraint: **free tiers are much tighter than they look.**
Mistral allows `magistral-small` one request every 33 seconds. Google removed
Pro models from the Gemini free tier in 2026. Firing requests without pacing
produces 429s that score 0 and look exactly like "this model is bad".

---

## Quick start

```bash
# 0. one-time
cp .env.example .env          # fill in your 4 API keys
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 1. infra (Redis only - use your own Postgres)
docker compose up -d redis

# 2. see what a run will cost BEFORE running it
python app\scripts\estimate.py

# 3. start API + worker (2 terminals)
uvicorn app.main:app --reload
celery -A app.celery.celery_app worker --pool=threads --concurrency=2 --loglevel=info

# 4. benchmark
curl -X POST http://localhost:8000/benchmark/groq     # start small (~3 min)
curl -X POST http://localhost:8000/benchmark/all      # everything (~55 min)
curl http://localhost:8000/leaderboard
```

---

## The API (10 endpoints)

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health` | service + Redis check |
| POST | `/sync/models` | discover models from all providers |
| GET  | `/models` | list known models |
| POST | `/benchmark/groq` | benchmark Groq |
| POST | `/benchmark/nvidia` | benchmark NVIDIA |
| POST | `/benchmark/gemini` | benchmark Gemini |
| POST | `/benchmark/mistral` | benchmark Mistral |
| POST | `/benchmark/all` | all four, sequentially |
| GET  | `/runs/{id}` | run progress |
| GET  | `/leaderboard` | **top 10 across all providers** |

Docs: `http://localhost:8000/docs`

---

## How models are chosen

No model name is hardcoded anywhere. Every id is fetched live, then:

1. **Filter** non-chat models (audio, embeddings, image, OCR, safety, reward,
   agentic CLI, deep-research).
2. **Dedupe** exact duplicates and dated variants superseded by `-latest`
   (Mistral returns several models twice).
3. **Rank within each publisher** using signals encoded in the id itself:
   parameter size, version number, tier word (`large`/`pro` vs `mini`/`lite`),
   and instruction tuning.
4. **Keep the best model per publisher** — Meta's best, Google's best, Qwen's
   best, DeepSeek's best. Roughly 35 models total.
5. **Preflight probe**: one tiny request each. Anything that fails is dropped
   *before* the benchmark starts.
6. **Backfill**: if a publisher's flagship fails (Gemini Pro is no longer free),
   the next-best model from that same publisher is promoted automatically.

There is deliberately **no cap at 10**. The benchmark decides the top 10, not a
heuristic. `/leaderboard` shows the top `LEADERBOARD_TOP_N` (default 10).

Inspect any of this without running a benchmark:

```bash
python app\scripts\list_models.py            # everything each provider offers
python app\scripts\pick_models.py --show-all # which models win, and why
python app\scripts\estimate.py               # time + quota cost
```

---

## Rate limits and budgets

`rate_limits.json` holds the real published limits. **Edit it freely — it is
read at runtime, no code change needed.**

```json
"mistral": {
  "default_rps": 0.3,
  "daily_request_budget": 3000,
  "models": { "magistral-small": 0.03, "mistral-large": 0.07 }
}
```

Three protections:

- **Pacing** — waits the required gap before every call, per model. Model keys
  match by prefix, so `mistral-large` covers `mistral-large-latest`.
- **Adaptive backoff** — on a 429 it reads `retry-after` and *permanently*
  slows that model for the rest of the session.
- **Budget cap** — a hard ceiling on requests per provider. When it's hit the
  run stops cleanly and keeps everything already scored.

Results are saved **after each model**, not at the end, so a crash or a spent
budget never loses completed work. A partial leaderboard beats none.

---

## Tuning

In `.env`:

```dotenv
KNOWLEDGE_SAMPLE_SIZE=5      # questions per category (4 categories = 20/model)
REASONING_SAMPLE_SIZE=5
MATH_SAMPLE_SIZE=5
CODING_SAMPLE_SIZE=5
MODELS_PER_PUBLISHER=1       # 2 = flagship + runner-up
MAX_MODELS_PER_PROVIDER=0    # 0 = no cap
LEADERBOARD_TOP_N=10
```

Fewer questions = faster run, noisier scores. 5 per category is enough to rank
models; raise it once you have a full leaderboard.

---

## If something goes wrong

```bash
python stop.py     # kills workers and flushes the queue
```

**Everything scoring 0** → rate limits. Lower `default_rps` for that provider in
`rate_limits.json`.
**A provider returns no models** → missing API key, or all its models failed
preflight. Check the worker log.
**Run seems stuck** → it is probably pacing. `magistral-small` legitimately waits
33 seconds between calls; the log prints `pacing ... waiting Xs`.

---

## Benchmarks

| Category | Dataset | Weight |
|----------|---------|--------|
| knowledge | MMLU-Pro | 0.30 |
| reasoning | GPQA Diamond | 0.25 |
| math | MATH-500 | 0.20 |
| coding | HumanEval | 0.25 |

Composite = weighted average, 0-100.
