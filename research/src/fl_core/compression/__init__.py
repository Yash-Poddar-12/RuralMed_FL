"""Compression package for communication-efficient federated learning.

Provides:
- TopKSparsifier: Top-k sparsification with client-side error feedback (DGC).
- QSGDQuantizer: Stochastic unbiased quantization (QSGD).
- Tensor flattening/unflattening helpers for PyTorch state dicts.
- ModelCompressor: Unified client-server interface.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import torch

from research.src.fl_core.compression.comms_accounting import (
    compute_uncompressed_bytes,
    compute_sparse_bytes,
    compute_quantized_bytes,
    calculate_compression_stats,
)
from research.src.fl_core.compression.sparsification import TopKSparsifier
from research.src.fl_core.compression.quantization import QSGDQuantizer


def flatten_tensor_list(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[torch.Size, int]]]:
    """Flatten a list of multi-dimensional tensors into a single 1D contiguous tensor.

    Args:
        tensors: List of PyTorch tensors.

    Returns:
        flat: 1D contiguous tensor containing all elements.
        metadata: List of (shape, numel) tuples to reconstruct original tensors.
    """
    metadata = [(t.shape, t.numel()) for t in tensors]
    if not tensors:
        return torch.empty(0), []
    flat = torch.cat([t.reshape(-1) for t in tensors], dim=0)
    return flat, metadata


def unflatten_tensor_list(
    flat: torch.Tensor,
    metadata: List[Tuple[torch.Size, int]],
) -> List[torch.Tensor]:
    """Reconstruct a list of multi-dimensional tensors from a 1D flat tensor.

    Args:
        flat: 1D tensor.
        metadata: List of (shape, numel) tuples.

    Returns:
        List of tensors reshaped to their original shapes.
    """
    tensors = []
    offset = 0
    for shape, numel in metadata:
        chunk = flat[offset : offset + numel].reshape(shape)
        tensors.append(chunk)
        offset += numel
    return tensors


class ModelCompressor:
    """Unified client-server compression coordinator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.method = str(cfg.get("method", "none")).lower()
        self.topk_ratio = float(cfg.get("topk_ratio", 0.05))
        self.error_feedback = bool(cfg.get("error_feedback", True))
        self.quant_bits = int(cfg.get("quant_bits", 8))
        self.residual_dir = cfg.get("residual_dir", None)

        if self.method == "topk":
            self.sparsifier = TopKSparsifier(
                ratio=self.topk_ratio,
                error_feedback=self.error_feedback,
                residual_dir=self.residual_dir,
            )
            self.quantizer = None
        elif self.method in ("qsgd", "quantization"):
            self.sparsifier = None
            self.quantizer = QSGDQuantizer(bits=self.quant_bits)
        else:
            self.method = "none"
            self.sparsifier = None
            self.quantizer = None

    def compress_update(
        self,
        flat_delta: torch.Tensor,
        client_id: str = "default",
    ) -> Tuple[Dict[str, Any], int]:
        """Compress a 1D update vector.

        Returns:
            payload: Dict containing compressed components.
            wire_bytes: Exact wire byte count.
        """
        total_elements = flat_delta.numel()

        if self.method == "topk" and self.sparsifier is not None:
            indices, values, wire_bytes = self.sparsifier.compress(flat_delta, client_id=client_id)
            payload = {
                "method": "topk",
                "indices": indices.cpu(),
                "values": values.cpu(),
                "total_elements": total_elements,
            }
            return payload, wire_bytes

        elif self.method in ("qsgd", "quantization") and self.quantizer is not None:
            quantized, norm, wire_bytes = self.quantizer.compress(flat_delta)
            payload = {
                "method": "qsgd",
                "quantized": quantized.cpu(),
                "norm": norm,
                "total_elements": total_elements,
            }
            return payload, wire_bytes

        else:
            # Uncompressed baseline
            wire_bytes = compute_uncompressed_bytes(total_elements, dtype_bytes=4)
            payload = {
                "method": "none",
                "values": flat_delta.cpu(),
                "total_elements": total_elements,
            }
            return payload, wire_bytes

    def decompress_update(
        self,
        payload: Dict[str, Any],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Decompress a received payload back into a dense 1D vector."""
        method = payload.get("method", "none")
        total_elements = int(payload["total_elements"])

        if method == "topk" and self.sparsifier is not None:
            return self.sparsifier.decompress(
                indices=payload["indices"],
                values=payload["values"],
                total_elements=total_elements,
                device=device,
                dtype=dtype,
            )

        elif method == "qsgd" and self.quantizer is not None:
            return self.quantizer.decompress(
                quantized=payload["quantized"],
                norm=float(payload["norm"]),
                device=device,
                dtype=dtype,
            )

        else:
            raw_vals = payload["values"]
            if device is not None:
                raw_vals = raw_vals.to(device=device)
            if dtype is not None:
                raw_vals = raw_vals.to(dtype=dtype)
            return raw_vals


__all__ = [
    "compute_uncompressed_bytes",
    "compute_sparse_bytes",
    "compute_quantized_bytes",
    "calculate_compression_stats",
    "TopKSparsifier",
    "QSGDQuantizer",
    "flatten_tensor_list",
    "unflatten_tensor_list",
    "ModelCompressor",
]
