"""Stochastic Quantization (QSGD style) for communication-efficient federated learning.

Implements Quantized SGD (Alistarh et al., NeurIPS 2017):
1. Quantizes floating-point updates to a low bit-width representation (e.g. 8-bit, 4-bit, 2-bit).
2. Uses randomized rounding so that the quantized vector is an UNBIASED estimator of the original:
     E[dequantize(quantize(v))] = v
3. Server dequantizes using the transmitted norm scale factor before FedAvg averaging.
"""

from __future__ import annotations

from typing import Tuple, Optional
import torch

from research.src.fl_core.compression.comms_accounting import compute_quantized_bytes


class QSGDQuantizer:
    """Stochastic unbiased quantizer (QSGD)."""

    def __init__(
        self,
        bits: int = 8,
    ) -> None:
        """
        Args:
            bits: Bit-width per coordinate (e.g., 8, 4, 2).
        """
        if bits not in (8, 4, 2, 1):
            raise ValueError(f"Supported bit-widths are (8, 4, 2, 1), got {bits}")
        self.bits = int(bits)

        # Number of non-zero quantization levels per direction (s)
        # For 8-bit: s = 127
        # For 4-bit: s = 7
        # For 2-bit: s = 1
        # For 1-bit: s = 1 (sign-only)
        if self.bits > 1:
            self.levels = (1 << (self.bits - 1)) - 1
        else:
            self.levels = 1

    def compress(
        self,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, int]:
        """Quantize a 1D tensor using stochastic unbiased rounding.

        Args:
            delta: 1D parameter update tensor.

        Returns:
            Tuple of:
              - quantized: 1D int8/int16 tensor containing signed discrete levels.
              - norm: Float scalar representing the L2 norm of the original vector.
              - wire_bytes: Exact wire bytes for packed payload and scale metadata.
        """
        total_elements = delta.numel()
        norm = float(torch.norm(delta, p=2).item())

        if norm == 0.0 or self.levels == 0:
            quantized = torch.zeros(total_elements, device=delta.device, dtype=torch.int8)
            wire_bytes = compute_quantized_bytes(total_elements, bits=self.bits)
            return quantized, 0.0, wire_bytes

        # Normalized absolute magnitude: xi in [0, 1]
        abs_delta = torch.abs(delta)
        xi = abs_delta / norm

        if self.bits == 1:
            # 1-bit signSGD style
            # Store sign: +1 or -1 scaled by mean/norm
            sign = torch.sign(delta)
            # If delta is zero, assign +1 or -1 randomly or 0
            sign[sign == 0] = 1.0
            quantized = sign.to(dtype=torch.int8)
        else:
            # Multi-bit stochastic rounding (QSGD)
            scaled = xi * self.levels
            l = torch.floor(scaled)
            # Probability of rounding up: p = scaled - l
            prob = scaled - l
            # Sample Bernoulli random variables with prob p
            rand = torch.rand_like(prob)
            round_up = (rand < prob).float()

            quant_levels = l + round_up
            quant_levels = torch.clamp(quant_levels, 0, self.levels)

            sign = torch.sign(delta)
            sign[sign == 0] = 1.0
            quantized = (sign * quant_levels).to(dtype=torch.int8)

        wire_bytes = compute_quantized_bytes(
            num_params=total_elements,
            bits=self.bits,
            scale_bytes=4,
            header_bytes=8,
        )

        return quantized, norm, wire_bytes

    def decompress(
        self,
        quantized: torch.Tensor,
        norm: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Dequantize discrete levels back to an unbiased floating-point estimate.

        Args:
            quantized: 1D tensor of signed integer levels.
            norm: Transmitted L2 norm scale factor.
            device: Target device for output tensor.
            dtype: Target dtype (e.g. torch.float32).

        Returns:
            Dense 1D floating-point reconstructed tensor.
        """
        target_device = device if device is not None else quantized.device
        target_dtype = dtype if dtype is not None else torch.float32

        if norm == 0.0:
            return torch.zeros(quantized.shape, device=target_device, dtype=target_dtype)

        q_float = quantized.to(device=target_device, dtype=target_dtype)

        if self.bits == 1:
            # For 1-bit signSGD, scale factor is norm / sqrt(d)
            scale = norm / math.sqrt(float(quantized.numel()))
            return q_float * scale
        else:
            # Unbiased estimator: v_approx = norm * (q / s)
            return (norm / float(self.levels)) * q_float
