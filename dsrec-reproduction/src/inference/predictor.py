from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import IdMapping
from src.data.time_features import load_time_bucketizer
from src.models.dsrec import DSRec


class Predictor:
    """Load a trained DSRec checkpoint and produce deterministic top-k scores."""

    def __init__(
        self,
        checkpoint: Path,
        interactions: Path,
        time_bucketizer: Path,
        user_mapping: Path,
        item_mapping: Path,
        max_len: int | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.interactions = pd.read_pickle(interactions)
        self.bucketizer = load_time_bucketizer(time_bucketizer)

        with user_mapping.open("rb") as f:
            self.user_mapping: IdMapping = pickle.load(f)
        with item_mapping.open("rb") as f:
            self.item_mapping: IdMapping = pickle.load(f)

        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = ckpt["model_state_dict"]
        saved_config = ckpt.get("model_config", {})
        n_items = state["item_embedding.weight"].shape[0] - 1

        # Recreate the exact architecture/backend used to train this checkpoint.
        # Older checkpoints lack metadata and fall back to the portable baseline.
        self.max_len = max_len or int(saved_config.get("max_sequence_length", 50))
        self.model = DSRec(
            n_items=n_items,
            d_model=int(saved_config.get("d_model", 64)),
            n_time_buckets=int(saved_config.get("n_time_buckets", self.bucketizer.n_buckets)),
            n_blocks=int(saved_config.get("n_blocks", 2)),
            d_state=int(saved_config.get("d_state", 32)),
            conv_width=int(saved_config.get("conv_width", 4)),
            expansion=int(saved_config.get("expansion", 2)),
            dropout=float(saved_config.get("dropout", 0.2)),
            cross_fusion=bool(saved_config.get("cross_fusion", True)),
            dual_interest=bool(saved_config.get("dual_interest", True)),
            short_ssm=bool(saved_config.get("short_ssm", True)),
            long_branch=str(saved_config.get("long_branch", "mamba")),
            short_branch=str(saved_config.get("short_branch", "time_aware_ssm")),
            ssm_backend=str(saved_config.get("ssm_backend", "torch")),
            paper_norms=bool(saved_config.get("paper_norms", False)),
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model_version = f"epoch-{ckpt.get('epoch', 'unknown')}"

    def _internal_user_id(self, raw_user_id: int) -> int:
        return int(self.user_mapping.raw_to_internal.get(raw_user_id, 0))

    def recommend(self, raw_user_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        internal_user = self._internal_user_id(raw_user_id)
        if internal_user == 0:
            raise KeyError(f"Unknown user_id: {raw_user_id}")

        hist = self.interactions[self.interactions["user_id"] == internal_user].tail(self.max_len)
        if hist.empty:
            raise KeyError(f"User has no interaction history: {raw_user_id}")

        item_ids = hist["item_id"].to_numpy(dtype=np.int64)
        timestamps = hist["timestamp"].to_numpy(dtype=np.int64)
        gaps = np.zeros_like(timestamps)
        if len(timestamps) > 1:
            gaps[1:] = np.maximum(np.diff(timestamps), 0)
        time_bucket_ids = self.bucketizer.transform(gaps).astype(np.int64)

        ids = torch.from_numpy(item_ids[None, :]).to(self.device)
        tb = torch.from_numpy(time_bucket_ids[None, :]).to(self.device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.inference_mode():
            logits = self.model(ids, tb, mask)[0]
            values, indices = torch.topk(logits, k=min(top_k, logits.numel() - 1))

        recommendations: list[tuple[int, float]] = []
        for internal_item_id, score in zip(indices.tolist(), values.tolist(), strict=True):
            if internal_item_id == 0:
                continue
            raw_item_id = self.item_mapping.internal_to_raw.get(internal_item_id)
            if raw_item_id is None:
                continue
            recommendations.append((int(raw_item_id), float(score)))
        return recommendations
