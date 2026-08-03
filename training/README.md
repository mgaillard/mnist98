# Minimal MNIST MLP

A small PyTorch project that trains a two-layer perceptron on MNIST.

## Architecture

```
Input (784) → Linear(64) → ReLU → Linear(10) → Softmax (via CrossEntropyLoss)
```

All computations are in fp32.

## Quick Start

Install [uv](https://docs.astral.sh/uv/) (or use an existing Python 3.10+ interpreter).

```bash
# Install dependencies (creates a venv automatically)
uv sync

# Train the model
uv run python train.py --epochs 5
```

To install into an existing virtual environment:

```bash
uv pip install -e .
```

## Checkpoints

Training saves the best model (by test accuracy) to two formats:

| File | Format | Use |
|------|--------|-----|
| `checkpoints/weights.bin` | C89 little-endian binary | Embedded inference (C) |
| `checkpoints/weights.pt`  | PyTorch checkpoint        | Python inference / fine-tuning |

### PyTorch checkpoint (`.pt`)

Contains the full `state_dict` plus metadata:

```python
{
    "model_state":      state_dict of the MLP,
    "in_size":          784,
    "hidden_size":      64,
    "out_size":         10,
    "normalize_mean":   0.1307,
    "normalize_std":    0.3081,
}
```

### Binary weights (`.bin`)

C89-friendly little-endian binary format:

| Offset | Field | Type |
|--------|-------|------|
| 0 | magic (`0x4E4D5354`) | int32 |
| 4 | num_layers | int32 |
| 8 | in_dim[0..1] | int32[2] |
| 16 | out_dim[0..1] | int32[2] |
| 24 | weights + biases | float32[] |

Layer 0: weights `[64][784]`, bias `[64]`
Layer 1: weights `[10][64]`, bias `[10]`

Total weight data: 203,560 bytes (50,890 float32 values).

## Inference

Run inference on a single BMP image using a trained `.pt` checkpoint:

```bash
uv run python infer.py data/test_real3.bmp --weights checkpoints/weights.pt
```

Prints the predicted digit and a bar chart of all 10 class probabilities.

## Extract Digits as BMP

Save MNIST examples as 28x28 grayscale BMP files:

```bash
# All training digits
uv run python extract_bmp.py ../build

# Five examples of the digit 3
uv run python extract_bmp.py ../build --label 3 --count 5

# Three examples of the digit 7, starting at index 10
uv run python extract_bmp.py ../build --label 7 --start 10 --count 3
```

Files are named `{label}_{index:04d}.bmp` (e.g. `3_0012.bmp`).
