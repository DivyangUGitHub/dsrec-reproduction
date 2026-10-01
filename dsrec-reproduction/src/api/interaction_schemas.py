# ruff: noqa: I001
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


InteractionType = Literal[
    "recommendation_requested",
    "impression",
    "click",
    "like",
    "dislike",
    "hide",
]


class InteractionRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    event_type: InteractionType
    item_id: int | None = Field(default=None, ge=1)
    recommendation_rank: int | None = Field(default=None, ge=1, le=100)
    recommendation_score: float | None = None
    session_id: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InteractionResponse(BaseModel):
    id: UUID
    user_id: int
    event_type: InteractionType
    item_id: int | None
    created_at: datetime


class InteractionListResponse(BaseModel):
    user_id: int
    events: list[InteractionResponse]
