from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    top_k: int = Field(default=10, ge=1, le=100)


class Recommendation(BaseModel):
    item_id: int
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[Recommendation]
    model_version: str
