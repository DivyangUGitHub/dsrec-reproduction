from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    processed_dir: str = "data/processed"
    max_sequence_length: int = 50
    n_time_buckets: int = 10


@dataclass
class ModelConfig:
    d_model: int = 64
    n_blocks: int = 2
    d_state: int = 32
    conv_width: int = 4
    expansion: int = 2
    dropout: float = 0.2
    # Ablation switches. Defaults preserve the original DSRec behavior.
    cross_fusion: bool = True
    dual_interest: bool = True
    short_ssm: bool = True
    long_branch: str = "mamba"
    short_branch: str = "time_aware_ssm"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    model_raw = raw.get("model", {})

    return Config(
        seed=int(raw.get("seed", 42)),
        data=DataConfig(
            **{
                k: v
                for k, v in raw.get("data", {}).items()
                if k in DataConfig.__annotations__
            }
        ),
        model=ModelConfig(
            **{
                k: v
                for k, v in model_raw.items()
                if k in ModelConfig.__annotations__
            }
        ),
        training=TrainingConfig(
            **{
                k: v
                for k, v in raw.get("training", {}).items()
                if k in TrainingConfig.__annotations__
            }
        ),
    )
