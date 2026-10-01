from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="DSRec Recommendation API",
    version="0.3.0",
    description="Inference service for the DSRec sequential recommender.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "DSRec Recommendation API", "docs": "/docs", "health": "/v1/health"}


app.include_router(router, prefix="/v1")
