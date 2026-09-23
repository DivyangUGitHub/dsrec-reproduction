"""
Phase 9 — DSRec training loop with validation and checkpointing.
"""

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
from src.data.split import generate_training_examples, leave_one_out_split
from src.data.time_features import load_time_bucketizer
from src.models.dsrec import DSRec


# ---------------------------------------------------------------------
# Paths / built-in defaults
# ---------------------------------------------------------------------

CHECKPOINT_DIR = Path("data/checkpoints")


# ---------------------------------------------------------------------
# Data / model / training defaults
# ---------------------------------------------------------------------

DEFAULT_CONFIG = Config()

BATCH_SIZE = DEFAULT_CONFIG.training.batch_size
MAX_LEN = DEFAULT_CONFIG.data.max_sequence_length
N_TIME_BUCKETS = DEFAULT_CONFIG.data.n_time_buckets

D_MODEL = DEFAULT_CONFIG.model.d_model
N_BLOCKS = DEFAULT_CONFIG.model.n_blocks
D_STATE = DEFAULT_CONFIG.model.d_state
CONV_WIDTH = DEFAULT_CONFIG.model.conv_width
EXPANSION = DEFAULT_CONFIG.model.expansion
DROPOUT = DEFAULT_CONFIG.model.dropout

LEARNING_RATE = DEFAULT_CONFIG.training.learning_rate
EPOCHS = DEFAULT_CONFIG.training.epochs


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
    """Evaluate average validation loss."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            item_ids = batch["item_ids"].to(device)
            time_bucket_ids = batch["time_bucket_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            logits = model(item_ids, time_bucket_ids, mask)
            loss = criterion(logits, targets)

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

    path.parent.mkdir(parents=True, exist_ok=True)

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
    parser = argparse.ArgumentParser(
        description="DSRec training + validation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML configuration file. "
            "If omitted, built-in defaults are used."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Number of epochs. "
            "Overrides config value when provided."
        ),
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

    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max-train-batches must be >= 1")

    if args.max_val_batches is not None and args.max_val_batches < 1:
        parser.error("--max-val-batches must be >= 1")

    config = load_config(args.config) if args.config else Config()

    data_config = config.data
    model_config = config.model
    training_config = config.training

    epochs = (
        args.epochs
        if args.epochs is not None
        else int(training_config.epochs)
    )

    batch_size = int(training_config.batch_size)
    max_len = int(data_config.max_sequence_length)
    n_time_buckets = int(data_config.n_time_buckets)

    d_model = int(model_config.d_model)
    n_blocks = int(model_config.n_blocks)
    d_state = int(model_config.d_state)
    conv_width = int(model_config.conv_width)
    expansion = int(model_config.expansion)
    dropout = float(model_config.dropout)

    learning_rate = float(training_config.learning_rate)
    processed_dir = Path(data_config.processed_dir)

    print("=== PHASE 9: TRAINING + VALIDATION ===")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print("Config:", args.config or "built-in defaults")
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Max sequence length:", max_len)
    print("Model dimension:", d_model)
    print("Blocks:", n_blocks)
    print("State dimension:", d_state)
    print("Learning rate:", learning_rate)
    print(
        "Max train batches:",
        args.max_train_batches if args.max_train_batches is not None else "ALL",
    )
    print(
        "Max validation batches:",
        args.max_val_batches if args.max_val_batches is not None else "ALL",
    )

    interactions_path = processed_dir / "interactions.pkl"

    print(f"Loading interactions from: {interactions_path}")

    df = pd.read_pickle(interactions_path)

    print(f"Interactions: {len(df):,}")

    print("Building train/validation examples...")

    train_examples, val_examples = build_examples(df)

    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")

    bucketizer = load_time_bucketizer(
        processed_dir / "time_bucketizer.pkl"
    )

    train_dataset = DSRecDataset(
        examples=train_examples,
        bucketizer=bucketizer,
        max_len=max_len,
    )

    val_dataset = DSRecDataset(
        examples=val_examples,
        bucketizer=bucketizer,
        max_len=max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    print(f"Training batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")

    n_items = int(df["item_id"].max())

    model = DSRec(
        n_items=n_items,
        d_model=d_model,
        n_time_buckets=n_time_buckets,
        n_blocks=n_blocks,
        d_state=d_state,
        conv_width=conv_width,
        expansion=expansion,
        dropout=dropout,
    ).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Items: {n_items}")
    print(f"Model parameters: {parameter_count:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_examples = 0

        print(f"\nEpoch {epoch}/{epochs}")

        for batch_idx, batch in enumerate(train_loader, start=1):
            if (
                args.max_train_batches is not None
                and batch_idx > args.max_train_batches
            ):
                break

            item_ids = batch["item_ids"].to(device)
            time_bucket_ids = batch["time_bucket_ids"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            logits = model(item_ids, time_bucket_ids, mask)
            loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch={epoch}, batch={batch_idx}"
                )

            loss.backward()
            optimizer.step()

            current_batch_size = targets.size(0)
            total_loss += loss.item() * current_batch_size
            total_examples += current_batch_size

            if (
                batch_idx == 1
                or batch_idx % 100 == 0
                or (
                    args.max_train_batches is not None
                    and batch_idx == args.max_train_batches
                )
            ):
                running_loss = total_loss / total_examples
                print(
                    f"  batch {batch_idx:,}/{len(train_loader):,} "
                    f"loss={loss.item():.4f} avg={running_loss:.4f}"
                )

        if total_examples == 0:
            raise RuntimeError(
                "Training processed zero examples. Increase --max-train-batches."
            )

        train_loss = total_loss / total_examples

        val_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError("Validation loss is non-finite.")

        print(f"\nEpoch {epoch} complete")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  val loss:   {val_loss:.4f}")

        last_path = CHECKPOINT_DIR / "last.pt"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            path=last_path,
        )

        print(f"Saved checkpoint: {last_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = CHECKPOINT_DIR / "best.pt"

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                path=best_path,
            )

            print(f"New best validation loss: {best_val_loss:.4f}")
            print(f"Saved best checkpoint: {best_path}")

    print("\n=== PHASE 9 COMPLETE ===")


if __name__ == "__main__":
    main()
