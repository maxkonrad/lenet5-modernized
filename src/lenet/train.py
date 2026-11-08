"""Training entry point: ``python -m lenet.train`` / ``lenet-train``."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lenet.augment import AugmentConfig, augment_batch
from lenet.data import default_data_dir, download_mnist, load_mnist
from lenet.model import LeNet5
from lenet.optim import SGD, Adam

log = logging.getLogger("lenet")


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "adam"
    seed: int = 0
    val_size: int = 5000
    train_limit: int | None = None
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    log_every: int = 100
    augment: bool = False
    weight_decay: float = 0.0
    dropout: float = 0.0


def build_optimizer(name: str, lr: float, weight_decay: float = 0.0) -> Adam | SGD:
    if name == "adam":
        return Adam(lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r} (choose adam|sgd)")


def train(config: TrainConfig) -> LeNet5:
    rng = np.random.default_rng(config.seed)
    root = Path(config.data_dir)
    if not (root / "train-images.idx3-ubyte").exists():
        log.info("MNIST not found in %s, downloading…", root)
        download_mnist(root)

    x_train, y_train, x_test, y_test = load_mnist(root)
    if config.train_limit is not None:
        x_train, y_train = x_train[: config.train_limit], y_train[: config.train_limit]

    perm = rng.permutation(x_train.shape[0])
    n_val = min(config.val_size, x_train.shape[0] // 5)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_tr, y_tr = x_train[train_idx], y_train[train_idx]

    net = LeNet5(seed=config.seed, fc_dropout=config.dropout)
    opt = build_optimizer(config.optimizer, config.lr, config.weight_decay)
    aug_cfg = AugmentConfig()
    ckpt = Path(config.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    n = x_tr.shape[0]
    for epoch in range(config.epochs):
        order = rng.permutation(n)
        x_tr, y_tr = x_tr[order], y_tr[order]
        # Cosine decay to zero; keeps late epochs stable without extra flags.
        decay = 0.5 * (1 + np.cos(np.pi * epoch / max(config.epochs, 1)))
        opt.lr = config.lr * decay  # type: ignore[attr-defined]

        epoch_loss, correct = 0.0, 0
        batches = tqdm(range(0, n, config.batch_size), desc=f"epoch {epoch:02d}", leave=False)
        for i in batches:
            xb, yb = x_tr[i : i + config.batch_size], y_tr[i : i + config.batch_size]
            if config.augment:
                xb = augment_batch(xb, rng, aug_cfg)
            probs, cache = net.forward(xb)
            grads, loss = net.backward(probs, cache, yb)
            opt.step(net, grads)
            epoch_loss += loss * xb.shape[0]
            correct += int(np.sum(np.argmax(probs, axis=1) == yb))
            if (i // config.batch_size) % config.log_every == 0:
                batches.set_postfix(loss=f"{loss:.4f}")

        val_loss, val_acc = net.evaluate(x_val, y_val)
        train_loss, train_acc = epoch_loss / n, 100.0 * correct / n
        log.info(
            "epoch %02d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%% | lr %.2e",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            opt.lr,  # type: ignore[attr-defined]
        )
        net.save(ckpt / "last.npz")
        if val_acc > best_val:
            best_val = val_acc
            net.save(ckpt / "best.npz")

    test_loss, test_acc = net.evaluate(x_test, y_test)
    log.info("test loss %.4f acc %.2f%% (best val %.2f%%)", test_loss, test_acc, best_val)
    return net


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train LeNet-5 from scratch (NumPy)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--augment", action="store_true", help="random shifts + elastic distortions"
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args(argv)
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        seed=args.seed,
        val_size=args.val_size,
        train_limit=args.train_limit,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        augment=args.augment,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(parse_args(argv))


if __name__ == "__main__":
    main()
