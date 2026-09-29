from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.db import init_db


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="DSRec Recommendation API",
    version="0.2.0",
    description="Inference service for the DSRec sequential recommender.",
    lifespan=lifespan,
)
app.include_router(router, prefix="/v1")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
