#!/usr/bin/env python3
"""Inference script: load a trained model and an image, output prediction + scores.

Usage:
    python infer.py <image.bmp> --weights checkpoints/weights.pt

The script loads the model architecture and weights from the .pt file,
preprocesses the image the same way as training, runs a forward pass,
and prints the predicted digit along with all 10 class probabilities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import MLP


def load_model(weights_path: str | Path) -> MLP:
    """Load a trained MLP from a .pt checkpoint."""
    checkpoint = torch.load(weights_path, weights_only=True)
    model = MLP()
    model.load_state_dict(checkpoint["model_state"])
    return model


def preprocess_image(image_path: str | Path) -> torch.Tensor:
    """Load a BMP image and preprocess it to match training conditions.

    Steps:
        1. Load as grayscale (0-255)
        2. Convert to float tensor and normalize to 0-1
        3. Apply MNIST normalization: (x - mean) / std
    """
    img = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform(img).unsqueeze(0)  # add batch dimension


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MNIST inference on an image using a trained model."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to the input BMP image.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file.",
    )
    args = parser.parse_args()

    # Load model
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    model = load_model(weights_path)
    model.eval()

    # Load and preprocess image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    x = preprocess_image(image_path)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        prediction = probs.argmax().item()
        scores = probs.tolist()

    # Print results
    print(f"Image: {image_path.name}")
    print(f"Prediction: {prediction}")
    print("Scores:")
    for digit, score in enumerate(scores):
        bar = "#" * int(score * 50)
        print(f"  {digit}: {score:.4f}  {bar}")


if __name__ == "__main__":
    main()
