from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import get_recommender
from src.api.interaction_schemas import (
    InteractionListResponse,
    InteractionRequest,
    InteractionResponse,
)
from src.db import InteractionEvent, SessionLocal, User
from src.inference.recommender import Recommender
from src.inference.schemas import (
    Recommendation,
    RecommendationRequest,
    RecommendationResponse,
)

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(
    recommender: Recommender = Depends(get_recommender),  # noqa: B008
) -> dict[str, str]:
    return {"status": "ready", "model_version": recommender.predictor.model_version}


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(
    request: RecommendationRequest,
    recommender: Recommender = Depends(get_recommender),  # noqa: B008
    db: Session = Depends(get_db),  # noqa: B008
) -> RecommendationResponse:
    try:
        rows = recommender.recommend(request.user_id, request.top_k)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    user = db.get(User, request.user_id)
    if user is None:
        db.add(User(model_user_id=request.user_id))

    db.add(
        InteractionEvent(
            model_user_id=request.user_id,
            event_type="recommendation_requested",
            metadata_json={
                "top_k": request.top_k,
                "model_version": recommender.predictor.model_version,
            },
        )
    )
    db.commit()

    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=[Recommendation(item_id=i, score=s) for i, s in rows],
        model_version=recommender.predictor.model_version,
    )


@router.post("/interactions", response_model=InteractionResponse, status_code=201)
def record_interaction(
    request: InteractionRequest,
    db: Session = Depends(get_db),  # noqa: B008
) -> InteractionResponse:
    user = db.get(User, request.user_id)
    if user is None:
        user = User(model_user_id=request.user_id)
        db.add(user)
        db.flush()

    event = InteractionEvent(
        model_user_id=request.user_id,
        event_type=request.event_type,
        item_id=request.item_id,
        recommendation_rank=request.recommendation_rank,
        recommendation_score=request.recommendation_score,
        session_id=request.session_id,
        metadata_json=request.metadata,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    return InteractionResponse(
        id=event.id,
        user_id=event.model_user_id,
        event_type=event.event_type,
        item_id=event.item_id,
        created_at=event.created_at,
    )


@router.get("/users/{user_id}/interactions", response_model=InteractionListResponse)
def list_interactions(
    user_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),  # noqa: B008
) -> InteractionListResponse:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

    events = db.scalars(
        select(InteractionEvent)
        .where(InteractionEvent.model_user_id == user_id)
        .order_by(InteractionEvent.created_at.desc())
        .limit(limit)
    ).all()

    return InteractionListResponse(
        user_id=user_id,
        events=[
            InteractionResponse(
                id=event.id,
                user_id=event.model_user_id,
                event_type=event.event_type,
                item_id=event.item_id,
                created_at=event.created_at,
            )
            for event in events
        ],
    )
