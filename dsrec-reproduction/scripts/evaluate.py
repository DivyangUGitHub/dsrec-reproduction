"""
Phase 10 — DSRec evaluation / ranking metrics.

Loads a trained checkpoint and evaluates it on the validation examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.collate import collate_batch
from src.data.dataset import DSRecDataset, Example
from src.data.sequences import build_sequences
from src.data.split import leave_one_out_split
from src.data.time_features import load_time_bucketizer
from src.evaluation import evaluate_ranking
from src.models.dsrec import DSRec


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("data/checkpoints")

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

BATCH_SIZE = 32
MAX_LEN = 50

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

D_MODEL = 64
N_TIME_BUCKETS = 10
N_BLOCKS = 2
D_STATE = 32
CONV_WIDTH = 4
EXPANSION = 2
DROPOUT = 0.2


def build_validation_examples(
    df: pd.DataFrame,
) -> list[Example]:
    """
    Build exactly one validation example per eligible user.

    The split logic is identical to Phase 9.
    """

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
) -> DSRec:
    """Create DSRec and load a saved checkpoint."""

    model = DSRec(
        n_items=n_items,
        d_model=D_MODEL,
        n_time_buckets=N_TIME_BUCKETS,
        n_blocks=N_BLOCKS,
        d_state=D_STATE,
        conv_width=CONV_WIDTH,
        expansion=EXPANSION,
        dropout=DROPOUT,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def main() -> None:

    parser = argparse.ArgumentParser(
        description="DSRec ranking evaluation"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "best.pt",
        help="Checkpoint to evaluate.",
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum validation batches. None = full validation set.",
    )

    args = parser.parse_args()

    print("=== PHASE 10: EVALUATION ===")

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # -----------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    # -----------------------------------------------------------------
    # Load interactions
    # -----------------------------------------------------------------

    interactions_path = (
        PROCESSED_DIR / "interactions.pkl"
    )

    print(
        f"Loading interactions from: "
        f"{interactions_path}"
    )

    df = pd.read_pickle(
        interactions_path
    )

    print(
        f"Interactions: {len(df):,}"
    )

    # -----------------------------------------------------------------
    # Validation examples
    # -----------------------------------------------------------------

    print(
        "Building validation examples..."
    )

    val_examples = build_validation_examples(
        df
    )

    print(
        f"Validation examples: "
        f"{len(val_examples):,}"
    )

    # -----------------------------------------------------------------
    # Bucketizer
    # -----------------------------------------------------------------

    bucketizer = load_time_bucketizer(
        PROCESSED_DIR / "time_bucketizer.pkl"
    )

    # -----------------------------------------------------------------
    # Dataset / loader
    # -----------------------------------------------------------------

    dataset = DSRecDataset(
        examples=val_examples,
        bucketizer=bucketizer,
        max_len=MAX_LEN,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(
        f"Validation batches: "
        f"{len(loader):,}"
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    n_items = int(
        df["item_id"].max()
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        n_items=n_items,
        device=device,
    )

    print(
        f"Items: {n_items}"
    )

    # -----------------------------------------------------------------
    # Checkpoint metadata
    # -----------------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint train loss: "
        f"{checkpoint['train_loss']:.4f}"
    )

    print(
        f"Checkpoint val loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    print("\nRunning evaluation...")

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            loader,
            start=1,
        ):

            if (
                args.max_batches is not None
                and batch_idx > args.max_batches
            ):
                break

            item_ids = batch["item_ids"].to(
                device
            )

            time_bucket_ids = batch[
                "time_bucket_ids"
            ].to(device)

            mask = batch["mask"].to(
                device
            )

            targets = batch["target"].to(
                device
            )

            logits = model(
                item_ids,
                time_bucket_ids,
                mask,
            )

            all_logits.append(
                logits.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

            if (
                batch_idx == 1
                or batch_idx % 50 == 0
            ):
                print(
                    f"  evaluated batch "
                    f"{batch_idx:,}/"
                    f"{len(loader):,}"
                )

    if not all_logits:
        raise RuntimeError(
            "No validation batches were evaluated."
        )

    logits = torch.cat(
        all_logits,
        dim=0,
    )

    targets = torch.cat(
        all_targets,
        dim=0,
    )

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    metrics = evaluate_ranking(
        logits=logits,
        targets=targets,
        ks=(1, 5, 10, 20),
    )

    print("\n=== RANKING RESULTS ===")

    for name, value in metrics.items():
        print(
            f"{name.upper():8s}: {value:.6f}"
        )

    print(
        f"\nEvaluated examples: "
        f"{len(targets):,}"
    )

    print(
        "=== PHASE 10 COMPLETE ==="
    )


if __name__ == "__main__":
    main()