"""A small MNIST CNN used as an *independent* oracle for sample evaluation.

Why a separate model: evaluating a generator with its own encoder is circular
(the decoder was trained to satisfy that encoder). An independently trained
classifier gives an honest feature space (its penultimate layer) for FID /
precision-recall / novelty, and class probabilities for the classifier-oracle
metrics.

This is the *MNIST-specific* piece; the metrics in :mod:`ssllab.eval.generative`
are modality-agnostic. For a bio modality you would swap this oracle for a domain
model (e.g. a splice predictor, ESM) exposing the same ``features``/``proba`` API.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssllab.data.mnist import get_mnist_dataloaders

FEATURE_DIM = 128


class MnistCNN(nn.Module):
    """Conv-Conv-FC classifier; ``features()`` returns the 128-d penultimate layer."""

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14->7
        )
        self.fc1 = nn.Linear(64 * 7 * 7, FEATURE_DIM)
        self.head = nn.Linear(FEATURE_DIM, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)
        return F.relu(self.fc1(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    @torch.no_grad()
    def proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


def train_oracle(
    epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    data_dir: str = "data",
    device: torch.device | str = "cpu",
) -> tuple[MnistCNN, float]:
    """Train the oracle from scratch. Returns ``(model, test_accuracy)``."""
    train_loader, test_loader = get_mnist_dataloaders(batch_size, data_dir)
    model = MnistCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # test accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
    return model, correct / max(total, 1)


def load_or_train_oracle(
    path: str | Path,
    device: torch.device | str = "cpu",
    epochs: int = 2,
    data_dir: str = "data",
) -> MnistCNN:
    """Load a cached oracle, or train + cache one at ``path``."""
    path = Path(path)
    model = MnistCNN().to(device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    model, acc = train_oracle(epochs=epochs, data_dir=data_dir, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[oracle] trained MNIST CNN (test_acc={acc:.4f}) -> {path}")
    return model
