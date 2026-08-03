"""Minimal MLP for MNIST (784 → 64 → 10)."""

from __future__ import annotations

import struct
from pathlib import Path

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer perceptron with ReLU activation.

    Architecture::

        Linear(784, 64) → ReLU → Linear(64, 10)

    All computations are in fp32.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten to (batch, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # ------------------------------------------------------------------
    # PyTorch export
    # ------------------------------------------------------------------

    def export_pt(self, path: str | Path) -> None:
        """Save the model and metadata in PyTorch format.

        Saved items:
            model_state:     state_dict of the MLP
            in_size:         input size (784)
            hidden_size:     hidden layer size (64)
            out_size:        output size (10)
            normalize_mean:  MNIST normalization mean
            normalize_std:   MNIST normalization std
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.state_dict(),
                "in_size": 784,
                "hidden_size": 64,
                "out_size": 10,
                "normalize_mean": 0.1307,
                "normalize_std": 0.3081,
            },
            path,
        )

    # ------------------------------------------------------------------
    # Binary export — C89-friendly format
    # ------------------------------------------------------------------

    def export_weights(self, path: str | Path) -> None:
        """Write model weights to a little-endian binary file.

        Header (all int32):
            magic:          0x4E4D5354 ("NMST")
            num_layers:     2
            in_dim[0]:      784
            in_dim[1]:      64
            out_dim[0]:     64
            out_dim[1]:     10

        Then, for each layer (in definition order):
            weights: float32 array of shape (out, in) — row-major
            bias:    float32 array of shape (out,)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Layer metadata
        layers = [
            (self.fc1.weight.data, self.fc1.bias.data),
            (self.fc2.weight.data, self.fc2.bias.data),
        ]
        in_dims = [784, 64]
        out_dims = [64, 10]

        with open(path, "wb") as f:
            # Header
            f.write(struct.pack("<I", 0x4E4D5354))  # magic
            f.write(struct.pack("<i", len(layers)))  # num_layers
            for d in in_dims:
                f.write(struct.pack("<i", d))
            for d in out_dims:
                f.write(struct.pack("<i", d))

            # Weights and biases, layer by layer
            for w, b in layers:
                # Row-major: (out, in) — move to CPU first if on GPU
                f.write(w.detach().cpu().numpy().astype("float32").tobytes())
                f.write(b.detach().cpu().numpy().astype("float32").tobytes())
