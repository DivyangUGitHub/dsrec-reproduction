"""Phase 10 — DSRec evaluation / ranking metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.data.collate import collate_batch
from src.data.dataset import DSRecDataset, Example
from src.data.sequences import build_sequences
from src.data.split import leave_one_out_split
from src.data.time_features import load_time_bucketizer
from src.evaluation import evaluate_ranking
from src.models.dsrec import DSRec


DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")


def build_validation_examples(
    df: pd.DataFrame,
) -> list[Example]:
    """Build exactly one validation example per eligible user."""

    sequences = build_sequences(df)
    val_examples: list[Example] = []

    for sequence in sequences.values():
        split = leave_one_out_split(sequence)
        if split is None:
            continue

        val_examples.append(
            Example(
                context_items=split.val_context_items,
                context_gaps=split.val_context_gaps,
                target=split.val_target,
            )
        )

    return val_examples


def load_model(
    checkpoint_path: Path,
    n_items: int,
    device: torch.device,
    config: Config,
) -> DSRec:
    """Create the configured DSRec variant and load a checkpoint."""

    model = DSRec(
        n_items=n_items,
        d_model=int(config.model.d_model),
        n_time_buckets=int(config.data.n_time_buckets),
        n_blocks=int(config.model.n_blocks),
        d_state=int(config.model.d_state),
        conv_width=int(config.model.conv_width),
        expansion=int(config.model.expansion),
        dropout=float(config.model.dropout),
        cross_fusion=bool(config.model.cross_fusion),
        dual_interest=bool(config.model.dual_interest),
        short_ssm=bool(config.model.short_ssm),
        long_branch=str(config.model.long_branch),
        short_branch=str(config.model.short_branch),
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSRec ranking evaluation"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate. Defaults to the baseline best checkpoint or the configured ablation checkpoint.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config used to construct the matching model variant.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum validation batches. None = full validation set.",
    )

    args = parser.parse_args()

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be >= 1")

    config = load_config(args.config) if args.config else Config()
    processed_dir = Path(config.data.processed_dir)

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    elif config.ablation:
        checkpoint_path = (
            Path("experiments/checkpoints")
            / str(config.ablation)
            / "best.pt"
        )
    else:
        checkpoint_path = DEFAULT_CHECKPOINT_DIR / "best.pt"

    batch_size = int(config.training.batch_size)
    max_len = int(config.data.max_sequence_length)

    print("=== PHASE 10: EVALUATION ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print("Config:", args.config or "built-in defaults")
    print(f"Checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    interactions_path = processed_dir / "interactions.pkl"
    print(f"Loading interactions from: {interactions_path}")

    df = pd.read_pickle(interactions_path)
    print(f"Interactions: {len(df):,}")

    print("Building validation examples...")
    val_examples = build_validation_examples(df)
    print(f"Validation examples: {len(val_examples):,}")

    bucketizer = load_time_bucketizer(
        processed_dir / "time_bucketizer.pkl"
    )

    dataset = DSRecDataset(
        examples=val_examples,
        bucketizer=bucketizer,
        max_len=max_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(f"Validation batches: {len(loader):,}")

    n_items = int(df["item_id"].max())
    model = load_model(
        checkpoint_path=checkpoint_path,
        n_items=n_items,
        device=device,
        config=config,
    )

    print(f"Items: {n_items}")
    print("Model configuration:")
    print(f"  cross_fusion: {config.model.cross_fusion}")
    print(f"  dual_interest: {config.model.dual_interest}")
    print(f"  short_ssm: {config.model.short_ssm}")
    print(f"  long_branch: {config.model.long_branch}")
    print(f"  short_branch: {config.model.short_branch}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint train loss: {checkpoint['train_loss']:.4f}")
    print(f"Checkpoint val loss: {checkpoint['val_loss']:.4f}")

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    print("\nRunning evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if args.max_batches is not None and batch_idx > args.max_batches:
                break

            item_ids = batch["item_ids"].to(device)
            time_bucket_ids = batch["time_bucket_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            logits = model(item_ids, time_bucket_ids, mask)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

            if batch_idx == 1 or batch_idx % 50 == 0:
                print(
                    f"  evaluated batch {batch_idx:,}/{len(loader):,}"
                )

    if not all_logits:
        raise RuntimeError("No validation batches were evaluated.")

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = evaluate_ranking(
        logits=logits,
        targets=targets,
        ks=(1, 5, 10, 20),
    )

    print("\n=== RANKING RESULTS ===")
    for name, value in metrics.items():
        print(f"{name.upper():8s}: {value:.6f}")

    print(f"\nEvaluated examples: {len(targets):,}")
    print("=== PHASE 10 COMPLETE ===")


if __name__ == "__main__":
    main()
