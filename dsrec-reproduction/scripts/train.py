"""
Phase 9 — DSRec training loop with validation and checkpointing.
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
from src.data.split import generate_training_examples, leave_one_out_split
from src.data.time_features import load_time_bucketizer
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


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

LEARNING_RATE = 1e-3
EPOCHS = 1


def build_examples(
    df: pd.DataFrame,
) -> tuple[list[Example], list[Example]]:
    """
    Build training and validation examples.

    Training:
        Sliding-window examples from the training portion.

    Validation:
        One example per user:
        validation context -> validation target.
    """

    sequences = build_sequences(df)

    train_examples: list[Example] = []
    val_examples: list[Example] = []

    for sequence in sequences.values():

        split = leave_one_out_split(sequence)

        if split is None:
            continue

        # -------------------------------------------------------------
        # Training examples
        # -------------------------------------------------------------

        for example in generate_training_examples(
            split,
            mode="sliding_window",
        ):
            train_examples.append(
                Example(
                    context_items=example.context_items,
                    context_gaps=example.context_gaps,
                    target=example.target,
                )
            )

        # -------------------------------------------------------------
        # Validation example
        # -------------------------------------------------------------

        val_examples.append(
            Example(
                context_items=split.val_context_items,
                context_gaps=split.val_context_gaps,
                target=split.val_target,
            )
        )

    return train_examples, val_examples


def evaluate(
    model: DSRec,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """
    Evaluate average validation loss.

    If max_batches is provided, only that many validation batches
    are evaluated. This is useful for controlled smoke tests.
    """

    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            loader,
            start=1,
        ):

            if (
                max_batches is not None
                and batch_idx > max_batches
            ):
                break

            item_ids = batch["item_ids"].to(device)
            time_bucket_ids = batch["time_bucket_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            logits = model(
                item_ids,
                time_bucket_ids,
                mask,
            )

            loss = criterion(
                logits,
                targets,
            )

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

    model.train()

    if total_examples == 0:
        raise RuntimeError(
            "Validation produced zero examples. "
            "Increase --max-val-batches or check the validation dataset."
        )

    return total_loss / total_examples


def save_checkpoint(
    model: DSRec,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    path: Path,
) -> None:
    """Save model + optimizer state."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:

    # -----------------------------------------------------------------
    # CLI arguments
    # -----------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="DSRec training + validation"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs. Default: {EPOCHS}",
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "Maximum training batches per epoch. "
            "Default: all training batches."
        ),
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help=(
            "Maximum validation batches. "
            "Default: all validation batches."
        ),
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if (
        args.max_train_batches is not None
        and args.max_train_batches < 1
    ):
        parser.error("--max-train-batches must be >= 1")

    if (
        args.max_val_batches is not None
        and args.max_val_batches < 1
    ):
        parser.error("--max-val-batches must be >= 1")

    print("=== PHASE 9: TRAINING + VALIDATION ===")

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    print(
        "Epochs:",
        args.epochs,
    )

    print(
        "Max train batches:",
        (
            args.max_train_batches
            if args.max_train_batches is not None
            else "ALL"
        ),
    )

    print(
        "Max validation batches:",
        (
            args.max_val_batches
            if args.max_val_batches is not None
            else "ALL"
        ),
    )

    # -----------------------------------------------------------------
    # Load interactions
    # -----------------------------------------------------------------

    interactions_path = PROCESSED_DIR / "interactions.pkl"

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
    # Build examples
    # -----------------------------------------------------------------

    print(
        "Building train/validation examples..."
    )

    train_examples, val_examples = build_examples(df)

    print(
        f"Training examples: "
        f"{len(train_examples):,}"
    )

    print(
        f"Validation examples: "
        f"{len(val_examples):,}"
    )

    # -----------------------------------------------------------------
    # Time bucketizer
    # -----------------------------------------------------------------

    bucketizer = load_time_bucketizer(
        PROCESSED_DIR / "time_bucketizer.pkl"
    )

    # -----------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------

    train_dataset = DSRecDataset(
        examples=train_examples,
        bucketizer=bucketizer,
        max_len=MAX_LEN,
    )

    val_dataset = DSRecDataset(
        examples=val_examples,
        bucketizer=bucketizer,
        max_len=MAX_LEN,
    )

    # -----------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(
        f"Training batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader):,}"
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    n_items = int(
        df["item_id"].max()
    )

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

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Items: {n_items}"
    )

    print(
        f"Model parameters: "
        f"{parameter_count:,}"
    )

    # -----------------------------------------------------------------
    # Optimizer / loss
    # -----------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # -----------------------------------------------------------------
    # Checkpoint tracking
    # -----------------------------------------------------------------

    best_val_loss = float("inf")

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0
        total_examples = 0

        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        for batch_idx, batch in enumerate(
            train_loader,
            start=1,
        ):

            if (
                args.max_train_batches is not None
                and batch_idx > args.max_train_batches
            ):
                break

            item_ids = batch["item_ids"].to(device)
            time_bucket_ids = batch[
                "time_bucket_ids"
            ].to(device)

            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            logits = model(
                item_ids,
                time_bucket_ids,
                mask,
            )

            loss = criterion(
                logits,
                targets,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at "
                    f"epoch={epoch}, batch={batch_idx}"
                )

            loss.backward()

            optimizer.step()

            batch_size = targets.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_examples += batch_size

            if (
                batch_idx == 1
                or batch_idx % 100 == 0
                or (
                    args.max_train_batches is not None
                    and batch_idx == args.max_train_batches
                )
            ):

                running_loss = (
                    total_loss
                    / total_examples
                )

                print(
                    f"  batch "
                    f"{batch_idx:,}/"
                    f"{len(train_loader):,} "
                    f"loss={loss.item():.4f} "
                    f"avg={running_loss:.4f}"
                )

        if total_examples == 0:
            raise RuntimeError(
                "Training processed zero examples. "
                "Increase --max-train-batches."
            )

        train_loss = (
            total_loss
            / total_examples
        )

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        val_loss = evaluate(
    model=model,
    loader=val_loader,
    criterion=criterion,
    device=device,
    max_batches=args.max_val_batches,
) 

        if not torch.isfinite(
            torch.tensor(val_loss)
        ):
            raise RuntimeError(
                "Validation loss is non-finite."
            )

        print(
            f"\nEpoch {epoch} complete"
        )

        print(
            f"  train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  val loss:   "
            f"{val_loss:.4f}"
        )

        # -------------------------------------------------------------
        # Last checkpoint
        # -------------------------------------------------------------

        last_path = (
            CHECKPOINT_DIR
            / "last.pt"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            path=last_path,
        )

        print(
            f"Saved checkpoint: "
            f"{last_path}"
        )

        # -------------------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = (
                CHECKPOINT_DIR
                / "best.pt"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                path=best_path,
            )

            print(
                f"New best validation loss: "
                f"{best_val_loss:.4f}"
            )

            print(
                f"Saved best checkpoint: "
                f"{best_path}"
            )

    print(
        "\n=== PHASE 9 COMPLETE ==="
    )


if __name__ == "__main__":
    main()