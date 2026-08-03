"""Unit tests for the MLP model."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from model import MLP


@pytest.fixture
def model() -> MLP:
    return MLP()


def test_output_shape(model: MLP) -> None:
    """A batch of images should produce 10 logits per sample."""
    x = torch.randn(32, 1, 28, 28)
    y = model(x)
    assert y.shape == (32, 10)


def test_hidden_layer_size(model: MLP) -> None:
    """The hidden layer must have exactly 64 units."""
    assert model.fc1.out_features == 64
    assert model.fc1.in_features == 784
    assert model.fc2.in_features == 64
    assert model.fc2.out_features == 10


def test_export_weights(tmp_path: Path, model: MLP) -> None:
    """Weights file should have the correct header and total size."""
    out = tmp_path / "weights.bin"
    model.export_weights(out)
    assert out.exists()

    with open(out, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x4E4D5354

        num_layers = struct.unpack("<i", f.read(4))[0]
        assert num_layers == 2

        in_dims = [struct.unpack("<i", f.read(4))[0] for _ in range(2)]
        out_dims = [struct.unpack("<i", f.read(4))[0] for _ in range(2)]

    assert in_dims == [784, 64]
    assert out_dims == [64, 10]

    # Total weight bytes:
    # Layer 0: 64*784*4 + 64*4 = 200,832 + 256 = 201,088
    # Layer 1: 10*64*4  + 10*4  = 2,560 + 40    = 2,600
    expected_bytes = (64 * 784 + 64 + 10 * 64 + 10) * 4
    assert out.stat().st_size == 24 + expected_bytes  # 24 bytes header + weights
