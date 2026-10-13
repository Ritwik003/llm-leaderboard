from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.database import create_all_tables
from app.controllers.routes import router
from app.utils.provider_clients import close_client
from app.utils.cache import close_redis
from app.utils.logger import get_logger

logger = get_logger("main")

app = FastAPI(
    title="LLM Benchmark Leaderboard",
    description="Benchmark free LLMs from Groq, NVIDIA, Gemini, and Mistral — one provider at a time.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def on_startup():
    await create_all_tables()
    logger.info("Tables ready. API up. Trigger a run via POST /benchmark/<provider>.")


@app.on_event("shutdown")
async def on_shutdown():
    await close_client()
    await close_redis()
    logger.info("Shutdown clean.")
