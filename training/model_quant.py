"""Quantized MLP for MNIST (784 → 64 → 10) with integer arithmetic.

All weights and biases are stored in widened integer buffers with per-layer
scale factors so they can be dequantized back to float32 if needed.

Storage rules
-------------
  int8 quantized value  →  stored in int16 buffer
  int16 quantized value →  stored in int32 buffer

Forward pass (fully integer)
-----------------------------
  Input (float32)
    → symmetric quantise to int8, stored as int16
    → Layer 1: int8-w (int16 buf) @ int8-x (int16 buf) + int8-b (int16 buf)
              → int16 output
    → ReLU: max(0, x) on int16 → int16
    → Layer 2: int16-w (int32 buf) @ int16-x (int16 buf) + int16-b (int32 buf)
              → int32 output
    → argmax on int32 (no dequantisation needed)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Symmetric quantisation helper
# ---------------------------------------------------------------------------

def _symmetric_quantize(
    fp32: torch.Tensor,
    max_abs: torch.Tensor,
    qmax: int,
) -> tuple[torch.Tensor, float]:
    """Symmetric quantisation centred at zero.

    Maps ``fp32`` values to the range ``[-(qmax+1), qmax]``.
    For int8  : [-128, 127]
    For int16 : [-32768, 32767]

    Returns
    -------
    quantized : torch.Tensor
        Quantized tensor (dtype int16, same shape as *fp32*).
    scale : float
        Scale factor so that ``fp32 ≈ quantized * scale``.
    """
    assert max_abs != 0.0, "Cannot quantise a tensor with all zeros"

    scale = max_abs / qmax
    qmin = -(qmax + 1)
    quantized = (fp32 / scale).round().clamp(qmin, qmax)
    return quantized, scale


# ---------------------------------------------------------------------------
# Quantized linear layer
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """Linear layer whose weights and bias are stored as quantized integers.

    Parameters
    ----------
    in_features :
        Input dimension.
    out_features :
        Output dimension.
    in_qmax :
        Maximum positive quantisation level for weights and bias
        (127 for int8, 32767 for int16).
    out_qmax :
        Maximum positive quantisation level for bias
        (127 for int8, 32767 for int16).
    storage_dtype :
        Dtype of the weight storage buffer
        (int16 for int8 weights, int32 for int16 weights).
    output_dtype :
        Dtype used for the matmul output (int16 or int32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_qmax: int,
        out_qmax: int,
        storage_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_qmax = in_qmax
        self.out_qmax = out_qmax
        self.output_dtype = output_dtype

        # Non-quantized weights (float32)
        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.float32),
        )

        # Quantized weights — stored in widened buffer
        self.register_buffer(
            "weight_q",
            torch.zeros(out_features, in_features, dtype=storage_dtype),
        )

        # Non-quantized weights (float32)
        self.register_buffer(
            "bias",
            torch.zeros(out_features, dtype=torch.float32),
        )

        # Quantized bias — stored in widened buffer
        self.register_buffer(
            "bias_q",
            torch.zeros(out_features, dtype=storage_dtype),
        )

        # Per-layer scale factors (float32) for optional dequantisation
        self.register_buffer("scale_x", torch.tensor(1.0))
        self.register_buffer("scale_w", torch.tensor(1.0))
        self.register_buffer("scale_b", torch.tensor(1.0))

    # ------------------------------------------------------------------
    # Loading from float32
    # ------------------------------------------------------------------

    def load_from_float(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        """Symmetrically quantise float32 weights and bias in-place."""
        # Save non-quantized weights and bias for optional dequantisation later
        self.weight.copy_(weight.to(self.weight.dtype).to(self.weight.device))
        self.bias.copy_(bias.to(self.bias.dtype).to(self.bias.device))

        # Find the scale for the input
        print(f"Hardcoded x_max_abs to 2.821487 for testing purposes")
        x_max_abs = 4 * 2.821487 # TODO: divide by 4 to leave 2 bits of headroom for the matmul output
        s_x = x_max_abs / self.in_qmax

        # Find the scale for the weights
        weight_max_abs = 4 * weight.abs().max().item() # TODO: divide by 4 to leave 2 bits of headroom for the matmul output
        print(f"Hardcoded weight_max_abs to {weight_max_abs:.6f} for testing purposes")

        # Find the scale for the bias is the product of the input and weight scales
        # We make sure the final scale is equal to the product of the input and weight scales.
        # But qmax stays the same because it is the maximum quantization level for the bias.
        bias_max_abs = x_max_abs * weight_max_abs * self.in_qmax
        assert bias_max_abs > bias.abs().max().item(), "Bias is too large for the input and weight scales"

        w_q, s_w = _symmetric_quantize(weight, weight_max_abs, self.in_qmax)
        b_q, s_b = _symmetric_quantize(bias, bias_max_abs, self.out_qmax)

        print(f"Maximum w: {weight.abs().max().item():.6f}, weight_max_abs: {weight_max_abs:.6f}")
        print(f"Maximum w_q: {w_q.abs().max().item():.6f}, scale_w: {s_w:.6f}")

        print(f"Maximum b: {bias.abs().max().item():.6f}, bias_max_abs: {bias_max_abs:.6f}")
        print(f"Maximum b_q: {b_q.abs().max().item():.6f}, scale_b: {s_b:.6f}")

        self.weight_q.copy_(w_q.to(self.weight_q.dtype).to(self.weight_q.device))
        self.bias_q.copy_(b_q.to(self.bias_q.dtype).to(self.bias_q.device))

        self.scale_x.fill_(s_x)
        self.scale_w.fill_(s_w)
        self.scale_b.fill_(s_b)

    # ------------------------------------------------------------------
    # Quantize the input
    # ------------------------------------------------------------------

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Quantise float32 input to values compatible with the layer's weights and bias.

        Uses symmetric quantisation centred at zero.
        The quantisation is the same as for the weights so that the matmul is scale-consistent.
        """
        scale = self.scale_x.item()
        qmax = self.in_qmax
        qmin = -(qmax + 1)

        return (x / scale).round().clamp(qmin, qmax).to(self.weight_q.dtype)

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integer forward pass: ``x @ W^T + B`` computed in *output_dtype*.

        Both *x* and the weight buffer are cast to *output_dtype* so that
        the matmul operands match.  PyTorch promotes ``int16 @ int16 → int32``
        automatically; for an ``int16`` output_dtype the result is explicitly
        cast back.
        """
        w = self.weight_q.to(self.output_dtype)
        b = self.bias_q.to(self.output_dtype)

        result = x.to(self.output_dtype) @ w.t() + b
        
        return result

    def forward_float_quantized(self, x: torch.Tensor) -> torch.Tensor:
        """Float32 forward pass using quantized weights and bias.

        This is useful for testing the quantisation error.
        """
        xq = self.quantize_input(x)

        yq = self.forward(xq)

        y = yq.to(torch.float32) * self.scale_b.item()

        return y

    def forward_float(self, x: torch.Tensor) -> torch.Tensor:
        """Float32 forward pass: ``x @ W^T + B`` computed in float32."""
        return x @ self.weight.t() + self.bias

    # ------------------------------------------------------------------
    # Dequantisation helpers
    # ------------------------------------------------------------------

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct float32 weights and bias from quantized storage."""
        w = self.weight_q.to(torch.float32) * self.scale_w.item()
        b = self.bias_q.to(torch.float32) * self.scale_b.item()

        return w, b


# ---------------------------------------------------------------------------
# Quantized MLP
# ---------------------------------------------------------------------------

class QuantizedMLP(nn.Module):
    """Two-layer perceptron with fully integer arithmetic forward pass.

    Architecture::

        Input(float32)
          → quantise to int8 (stored as int16)
          → Linear(784, 64):
              weights  : int8 in int16 buffer
              bias     : int8 in int16 buffer
              output   : int16
          → ReLU(max(0, x)) on int16 → int16
          → Linear(64, 10):
              weights  : int16 in int32 buffer
              bias     : int16 in int32 buffer
              output   : int32
          → argmax on int32 (scale-invariant, no dequantisation needed)
    """

    def __init__(self) -> None:
        super().__init__()

        # TODO: get the maximum value from the input data as parameter and give it to the layers.

        # Layer 1 — int8 weights/bias, int16 output
        self.fc1 = QuantizedLinear(
            in_features=784,
            out_features=64,
            in_qmax=32767,             # int16
            out_qmax=32767,            # int16
            storage_dtype=torch.int32, # int16 → int32
            output_dtype=torch.int32,
        )

        # Layer 2 — int16 weights/bias, int32 output
        self.fc2 = QuantizedLinear(
            in_features=64,
            out_features=10,
            in_qmax=32767,             # int16
            out_qmax=32767,            # int16
            storage_dtype=torch.int32, # int16 → int32
            output_dtype=torch.int32,
        )

    # ------------------------------------------------------------------
    # Integer forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integer forward pass returning int32 logits.

        Parameters
        ----------
        x :
            Float32 input tensor of shape ``(batch, 1, 28, 28)``
            or ``(batch, 784)``.

        Returns
        -------
        Int32 tensor of shape ``(batch, 10)`` ready for argmax.
        """
        x = x.view(x.size(0), -1)  # flatten to (batch, 784)

        test1 = self.fc1.forward_float_quantized(x)
        test2 = self.fc1.forward_float(x)

        error = (test1 - test2).abs().max().item()
        print(f"Max error in fc1 forward: {error:.6f}")

        # 1. Quantise input: float32 → int16 (stored as int32)
        x = self.fc1.quantize_input(x)

        # 2. Layer 1: int16 @ int16.T → int16 + int16 → int16
        x = self.fc1(x)

        # 3. ReLU: max(0, x) on int16 → stays int16
        x = torch.clamp(x, min=0)

        # TODO: rescale to int16 range
        x = torch.bitwise_right_shift(x, 16)

        # 4. Layer 2: int16 @ int32.T → int32 + int32 → int32
        x = self.fc2(x)

        return x  # int32 logits of shape (batch, 10)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices via argmax of int32 output.

        No dequantisation is needed — argmax is scale-invariant.
        """
        return self.forward(x).argmax(dim=1)

    # ------------------------------------------------------------------
    # Loading from float32 model
    # ------------------------------------------------------------------

    def load_from_float_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Load weights from a standard float32 MLP state_dict and quantise.

        Parameters
        ----------
        state_dict :
            The ``state_dict()`` of an fp32 ``MLP`` (keys like
            ``fc1.weight``, ``fc1.bias``, ``fc2.weight``, ``fc2.bias``).
        """
        self.fc1.load_from_float(
            state_dict["fc1.weight"],
            state_dict["fc1.bias"],
        )
        self.fc2.load_from_float(
            state_dict["fc2.weight"],
            state_dict["fc2.bias"],
        )

    def load_from_model(self, model: nn.Module) -> None:
        """Convenience wrapper: quantise from an fp32 MLP instance."""
        self.load_from_float_state_dict(model.state_dict())

    # ------------------------------------------------------------------
    # Dequantisation helpers
    # ------------------------------------------------------------------

    def dequantize_output(self, int32_output: torch.Tensor) -> torch.Tensor:
        """Approximately dequantise int32 output back to float32.

        The effective output scale is the product of the input scale and
        both layers' weight scales.
        """
        output_scale = (
            self.fc1.scale_w.item()
            * self.fc2.scale_w.item()
        )
        return int32_output.to(torch.float32) * output_scale

    def dequantize_all(self) -> dict[str, torch.Tensor]:
        """Return all weights and biases dequantized to float32."""
        w1, b1 = self.fc1.dequantize()
        w2, b2 = self.fc2.dequantize()
        return {
            "fc1.weight": w1,
            "fc1.bias": b1,
            "fc2.weight": w2,
            "fc2.bias": b2,
        }
