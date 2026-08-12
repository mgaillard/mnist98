# MLP Digit Recognition on Windows 98

An experiment in running a multilayer perceptron for handwritten digit recognition (MNIST) on **Windows 98** — no GPU, no modern libraries, just C89 and floating point.

The project has two halves: a modern Python training pipeline, and a lightweight C89 inference program cross-compiled for Win32.

## How it works

```
Python (training)          C89 (inference)
┌─────────────────┐         ┌──────────────────┐
│  PyTorch MLP    │         │  BMP loader      │
│  784 → 64 → 10  │    ->   │  Weight loader   │
│  Export .bin    │   .bin  │  Forward pass    │
└─────────────────┘         │  Predict digit   │
                            │  Benchmark mode  │
                            └──────────────────┘
```

### Architecture

```
Input (784) → Linear(64) → ReLU → Linear(10) → ArgMax
```

All computations are in `float32`. The model uses MNIST normalization (`mean=0.1307`, `std=0.3081`).

## Project layout

```
├── training/               # Python training code
│   ├── train.py            # Train the MLP on MNIST
│   ├── model.py            # Model definition + weight export
│   ├── infer.py            # Python inference (verify predictions)
│   └── extract_bmp.py      # Save MNIST samples as BMP images
│
├── src/                    # C89 inference code
│   ├── inference.c         # Main program (CLI, prediction, benchmark)
│   ├── weights.c           # Load .bin weight files
│   └── bmp.c               # BMP image loader (24-bit + 8-bit palette)
│
├── include/                # C headers
├── checkpoints/            # Trained weights (.bin and .pt)
├── data/                   # Sample BMP digits (0-9)
└── build/                  # Compiled binaries
```

## Training

See [training/README.md](training/README.md) for full details.

```bash
cd training
uv sync
uv run python train.py --epochs 10
```

Training saves the best model to two formats:
- **`checkpoints/weights.bin`** — C89 little-endian binary (for the C program)
- **`checkpoints/weights.pt`** — PyTorch checkpoint (for Python inference)

## Building

Requires `gcc` and `i686-w64-mingw32-gcc` (MinGW cross-compiler). On Ubuntu the Windows 32-bit compiler can be installed with:
```bash
sudo apt install gcc-mingw-w64-i686-win32
```

Build the executables with:

```bash
make
```

This produces four binaries:

| Target | Platform | Notes |
|--------|----------|-------|
| `build/inference` | Linux | Release build (`-O2`) |
| `build/inference-debug` | Linux | Debug build (`-g`) |
| `build/inference-win32` | Windows 98 | Baseline x86 |
| `build/inference-win32-sse` | Windows 98 | With SSE/MMX (`-march=pentium3`) |

All builds use `-std=c89 -pedantic -Wall -Wextra -Werror`.

## Usage

Copy `checkpoints/weights.bin`, a BMP image, and `build/inference-win32.exe` (or the Linux binary) to the same directory.

### Predict a single image

```bash
./build/inference data/3.bmp --weights training/checkpoints/weights.bin
# 3
```

### Benchmark mode

Run the forward pass `N` times and report throughput:

```bash
./build/inference data/3.bmp --weights training/checkpoints/weights.bin --benchmark 1000
# 3
# Benchmark (1000 runs): avg 0.022 ms/img, 45972.8 img/s
```

Use this to compare baseline vs. SSE builds, or to see how fast a Pentium III can classify digits.

## Extract test images

Save MNIST samples as 28×28 grayscale BMPs:

```bash
cd training
uv run python extract_bmp.py ../build --label 0 --count 2
uv run python extract_bmp.py ../build --label 1 --count 2
uv run python extract_bmp.py ../build --label 2 --count 2
uv run python extract_bmp.py ../build --label 3 --count 2
uv run python extract_bmp.py ../build --label 4 --count 2
uv run python extract_bmp.py ../build --label 5 --count 2
uv run python extract_bmp.py ../build --label 6 --count 2
uv run python extract_bmp.py ../build --label 7 --count 2
uv run python extract_bmp.py ../build --label 8 --count 2
uv run python extract_bmp.py ../build --label 9 --count 2
```

## Bin weight format

The `.bin` file is a flat little-endian structure:

| Offset | Field | Type |
|--------|-------|------|
| 0 | magic (`0x4E4D5354`) | int32 |
| 4 | num_layers | int32 |
| 8 | in_dim[0..1] | int32[2] |
| 16 | out_dim[0..1] | int32[2] |
| 24 | weights + biases | float32[] |

Layer 0: weights `[64][784]`, bias `[64]` · Layer 1: weights `[10][64]`, bias `[10]`

Total: 203,560 bytes (50,890 float32 values).
