"""Train an MLP on MNIST and export weights to binary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP


def get_dataloader(train: bool, batch_size: int = 128) -> DataLoader:
    """Return a MNIST DataLoader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def train_one_epoch(model: MLP, loader: DataLoader, optimizer, device) -> tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = nn.functional.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (output.argmax(1) == targets).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model: MLP, loader: DataLoader, device) -> float:
    """Return accuracy on the given loader."""
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        correct += (output.argmax(1) == targets).sum().item()
        total += images.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="checkpoints/weights", help="Output path (without extension)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = get_dataloader(train=True, batch_size=args.batch_size)
    test_loader = get_dataloader(train=False, batch_size=args.batch_size)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:3d} | loss {loss:.4f} | train_acc {train_acc:.4f} | test_acc {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            model.export_weights(f"{args.output}.bin")
            model.export_pt(f"{args.output}.pt")
            print(f"  → saved best weights ({best_acc:.4f})")


if __name__ == "__main__":
    main()
