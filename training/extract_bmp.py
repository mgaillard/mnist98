#!/usr/bin/env python3
"""Extract MNIST examples as 28x28 grayscale BMP files.

Usage:
    python extract_bmp.py <output_dir> [--label N] [--start N] [--count N]

Examples:
    python extract_bmp.py ../build          # all training digits
    python extract_bmp.py ../build --label 3 --count 5  # five 3s
    python extract_bmp.py ../build --label 7 --start 10 --count 3  # 3 examples of 7
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from torchvision import datasets, transforms
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save MNIST examples as 28x28 grayscale BMP files."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save BMP files into.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="Only extract this digit (0-9). Default: all digits.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the first example to extract (default: 0).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of examples to extract. Default: all matching.",
    )
    args = parser.parse_args()

    if args.label is not None and not 0 <= args.label <= 9:
        print(f"Error: --label must be 0-9, got {args.label}", file=sys.stderr)
        sys.exit(1)

    # Load MNIST training set.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    # Filter by label if requested.
    if args.label is not None:
        indices = [i for i, (_, label) in enumerate(dataset) if label == args.label]
    else:
        indices = list(range(len(dataset)))

    # Slice by start/count.
    start = args.start
    count = args.count if args.count is not None else len(indices)
    selected = indices[start:start + count]

    if not selected:
        print(f"Error: no examples found (label={args.label}, start={start}, count={count})",
              file=sys.stderr)
        sys.exit(1)

    # Create output directory.
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx in selected:
        img_tensor, label = dataset[idx]
        # img_tensor is already 0-1 (raw pixel / 255).
        # Convert to 0-255 for the BMP file.
        arr = (img_tensor[0].numpy() * 255).clip(0, 255).astype("uint8")
        img_pil = Image.fromarray(arr, mode="L")
        filename = f"{label}_{saved:04d}.bmp"
        img_pil.save(out / filename)
        saved += 1

    print(f"Saved {saved} example(s) to {out}/")


if __name__ == "__main__":
    main()
