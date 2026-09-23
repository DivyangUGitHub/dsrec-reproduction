from __future__ import annotations

from src.inference.predictor import Predictor


class Recommender:
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def recommend(self, user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        return self.predictor.recommend(user_id, top_k)
