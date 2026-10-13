"""
Database tables. Kept minimal:
  Model              — one row per discovered LLM
  BenchmarkRun       — one row per "run this provider" click
  Score              — one row per (run, model, category)
  LeaderboardSnapshot— one row per (run, model): composite + per-category scores

The old detailed per-question ModelResponse table was dropped — it stored
thousands of rows per run and nothing in the simplified API surfaced it.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON,
)
from app.config.database import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(200), unique=True, index=True)  # e.g. meta/llama-3.1-70b-instruct
    name = Column(String(200))
    provider = Column(String(50))                            # groq | nvidia | gemini | mistral
    is_active = Column(Boolean, default=True)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id = Column(Integer, primary_key=True, index=True)
    provider = Column(String(50))                            # which provider this run targeted
    status = Column(String(20), default="running")           # running | completed | failed
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    total_models = Column(Integer, default=0)
    completed_models = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)


class Score(Base):
    __tablename__ = "scores"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("benchmark_runs.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    category = Column(String(50))                            # knowledge | reasoning | math | coding
    score = Column(Float)                                   # 0-100
    sample_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class LeaderboardSnapshot(Base):
    __tablename__ = "leaderboard_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("benchmark_runs.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    provider = Column(String(50))
    composite_score = Column(Float)
    scores_json = Column(JSON)                              # {category: score, ...}
    created_at = Column(DateTime, default=datetime.utcnow)
