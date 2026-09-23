from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from src.inference.predictor import Predictor
from src.inference.recommender import Recommender


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    root = Path(os.getenv("DSREC_ROOT", "."))
    predictor = Predictor(
        checkpoint=root / os.getenv("DSREC_CHECKPOINT", "data/checkpoints/best.pt"),
        interactions=root / os.getenv("DSREC_INTERACTIONS", "data/processed/interactions.pkl"),
        time_bucketizer=root / os.getenv("DSREC_TIME_BUCKETIZER", "data/processed/time_bucketizer.pkl"),
        user_mapping=root / os.getenv("DSREC_USER_MAPPING", "data/processed/user_mapping.pkl"),
        max_len=int(os.getenv("DSREC_MAX_LEN", "50")),
    )
    return Recommender(predictor)
