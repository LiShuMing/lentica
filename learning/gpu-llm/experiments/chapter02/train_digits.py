"""Reusable PyTorch training template for the sklearn digits dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "chapter02"


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout: float = 0.2
    num_workers: int = 0
    device: str = "auto"
    precision: str = "fp32"
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


class DigitsDataset(Dataset[tuple[Tensor, Tensor]]):
    """Small custom Dataset that stores normalized images and integer labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.labels[index]


class DigitsMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 10),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


@dataclass
class DataLoaders:
    train: DataLoader[tuple[Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor]]
    test: DataLoader[tuple[Tensor, Tensor]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def make_dataloaders(config: TrainConfig, device: torch.device) -> DataLoaders:
    digits = load_digits()
    features = (digits.data / 16.0).astype(np.float32)
    labels = digits.target.astype(np.int64)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=0.15,
        random_state=config.seed,
        stratify=labels,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.15 / 0.85,
        random_state=config.seed,
        stratify=y_train_val,
    )

    generator = torch.Generator().manual_seed(config.seed)
    common: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": config.num_workers > 0,
    }
    return DataLoaders(
        train=DataLoader(
            DigitsDataset(x_train, y_train),
            shuffle=True,
            generator=generator,
            **common,
        ),
        validation=DataLoader(DigitsDataset(x_validation, y_validation), shuffle=False, **common),
        test=DataLoader(DigitsDataset(x_test, y_test), shuffle=False, **common),
    )


def autocast_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype(device),
            enabled=use_amp,
        ):
            logits = model(features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype(device),
            enabled=use_amp,
        ):
            logits = model(features)
            loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def checkpoint_state(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    history: list[dict[str, float | int]],
    best_validation_accuracy: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": asdict(config),
        "history": history,
        "best_validation_accuracy": best_validation_accuracy,
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_train(args: argparse.Namespace) -> None:
    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_workers=args.num_workers,
        device=args.device,
        precision=args.precision,
        output_dir=str(args.output_dir),
    )
    device = select_device(config.device)
    resume_state: dict[str, Any] | None = None

    if args.resume is not None:
        resume_state = load_checkpoint(args.resume, device)
        saved = resume_state["config"]
        config.seed = int(saved["seed"])
        config.hidden_dim = int(saved["hidden_dim"])
        config.dropout = float(saved["dropout"])

    set_seed(config.seed)
    loaders = make_dataloaders(config, device)
    model = DigitsMLP(config.hidden_dim, config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.precision == "amp"
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp and device.type == "cuda",
    )

    start_epoch = 1
    best_validation_accuracy = 0.0
    history: list[dict[str, float | int]] = []
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_validation_accuracy = float(resume_state["best_validation_accuracy"])
        history = list(resume_state["history"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"

    print(
        f"device={device} precision={config.precision} "
        f"train/val/test={len(loaders.train.dataset)}/"
        f"{len(loaders.validation.dataset)}/{len(loaders.test.dataset)}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, config.epochs + 1):
        synchronize(device)
        start = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model, loaders.train, criterion, optimizer, scaler, device, use_amp
        )
        validation_loss, validation_accuracy = evaluate(model, loaders.validation, criterion, device, use_amp)
        synchronize(device)
        elapsed = time.perf_counter() - start

        metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "seconds": elapsed,
        }
        history.append(metrics)
        improved = validation_accuracy > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_accuracy

        state = checkpoint_state(
            epoch,
            model,
            optimizer,
            scaler,
            config,
            history,
            best_validation_accuracy,
        )
        torch.save(state, latest_path)
        if improved:
            torch.save(state, best_path)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.3f} val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.3f} time={elapsed:.2f}s"
        )

    if not best_path.is_file():
        raise RuntimeError("no best checkpoint exists; set --epochs above the resumed checkpoint epoch")
    best_state = load_checkpoint(best_path, device)
    model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, loaders.test, criterion, device, use_amp)

    summary: dict[str, Any] = {
        "config": asdict(config),
        "best_epoch": int(best_state["epoch"]),
        "best_validation_accuracy": float(best_state["best_validation_accuracy"]),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "history": history,
    }
    if device.type == "cuda":
        summary["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"final best_epoch={summary['best_epoch']} test_loss={test_loss:.4f} test_acc={test_accuracy:.3f}")
    print(f"checkpoints={output_dir}")


def run_eval(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    state = load_checkpoint(args.checkpoint, device)
    saved = state["config"]
    config = TrainConfig(**saved)
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.device = args.device

    set_seed(config.seed)
    loaders = make_dataloaders(config, device)
    model = DigitsMLP(config.hidden_dim, config.dropout).to(device)
    model.load_state_dict(state["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    use_amp = args.precision == "amp"
    test_loss, test_accuracy = evaluate(model, loaders.test, criterion, device, use_amp)
    print(
        f"checkpoint={args.checkpoint} epoch={state['epoch']} device={device} "
        f"test_loss={test_loss:.4f} test_acc={test_accuracy:.3f}"
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and save checkpoints")
    add_runtime_arguments(train_parser)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--dropout", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--resume", type=Path)
    train_parser.set_defaults(handler=run_train)

    eval_parser = commands.add_parser("eval", help="evaluate a saved checkpoint")
    add_runtime_arguments(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.set_defaults(handler=run_eval)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
