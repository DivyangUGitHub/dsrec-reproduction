from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_recommender
from src.inference.recommender import Recommender
from src.inference.schemas import Recommendation, RecommendationRequest, RecommendationResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(recommender: Recommender = Depends(get_recommender)) -> dict[str, str]:
    return {"status": "ready", "model_version": recommender.predictor.model_version}


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest, recommender: Recommender = Depends(get_recommender)) -> RecommendationResponse:
    try:
        rows = recommender.recommend(request.user_id, request.top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=[Recommendation(item_id=i, score=s) for i, s in rows],
        model_version=recommender.predictor.model_version,
    )
