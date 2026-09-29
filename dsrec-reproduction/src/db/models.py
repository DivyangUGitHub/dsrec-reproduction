from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.database import Base


class User(Base):
    __tablename__ = "users"

    model_user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    events: Mapped[list[InteractionEvent]] = relationship(back_populates="user")


class InteractionEvent(Base):
    __tablename__ = "interaction_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    model_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.model_user_id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    item_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    recommendation_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    recommendation_score: Mapped[float | None] = mapped_column(nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )

    user: Mapped[User] = relationship(back_populates="events")
