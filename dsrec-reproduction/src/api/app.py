from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="DSRec Recommendation API",
    version="0.1.0",
    description="Inference service for the DSRec sequential recommender.",
)
app.include_router(router, prefix="/v1")
