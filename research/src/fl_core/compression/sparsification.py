"""Top-k gradient/weight sparsification with error feedback (residual accumulation).

Implements Deep Gradient Compression (Lin et al., ICLR 2018) style sparsification:
1. Each client accumulates unsent updates locally in a residual buffer (error feedback).
2. Only the top-k entries by absolute magnitude are transmitted each round.
3. Coordinates not transmitted remain in the residual buffer to accumulate across rounds,
   preventing small update directions from stalling.
4. Server reconstructs the sparse vector before standard weighted FedAvg aggregation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch

from research.src.fl_core.compression.comms_accounting import compute_sparse_bytes


class TopKSparsifier:
    """Top-k parameter sparsifier with client-side error feedback."""

    def __init__(
        self,
        ratio: float = 0.05,
        error_feedback: bool = True,
        value_dtype_bytes: int = 4,
        residual_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            ratio: Fraction of elements to keep (0.0 < ratio <= 1.0).
                   e.g., 0.05 means keep top 5% (20x parameter reduction).
            error_feedback: If True, accumulate unsent coordinates in a local residual.
            value_dtype_bytes: 4 for float32, 2 for float16.
            residual_dir: Optional filesystem path to persist per-client residuals.
                          Essential in Flower Ray simulation where client actors are pooled.
        """
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"Sparsification ratio must be in (0, 1], got {ratio}")

        self.ratio = float(ratio)
        self.error_feedback = bool(error_feedback)
        self.value_dtype_bytes = int(value_dtype_bytes)
        self.residual_dir = Path(residual_dir) if residual_dir is not None else None
        if self.residual_dir is not None:
            self.residual_dir.mkdir(parents=True, exist_ok=True)

        # In-memory map client_id -> residual tensor (used if residual_dir is None)
        self.residuals: Dict[str, torch.Tensor] = {}

    def _get_residual(self, client_id: str, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.error_feedback:
            return torch.zeros(shape, device=device, dtype=dtype)

        if self.residual_dir is not None:
            p = self.residual_dir / f"{client_id}.pt"
            if p.exists():
                try:
                    return torch.load(p, map_location=device, weights_only=True).to(dtype=dtype)
                except Exception:
                    pass
            return torch.zeros(shape, device=device, dtype=dtype)
        else:
            if client_id in self.residuals:
                res = self.residuals[client_id]
                return res.to(device=device, dtype=dtype)
            return torch.zeros(shape, device=device, dtype=dtype)

    def _save_residual(self, client_id: str, residual: torch.Tensor) -> None:
        if not self.error_feedback:
            return

        if self.residual_dir is not None:
            p = self.residual_dir / f"{client_id}.pt"
            torch.save(residual.cpu(), p)
        else:
            self.residuals[client_id] = residual.detach()

    def compress(
        self,
        delta: torch.Tensor,
        client_id: str = "default",
        ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compress a 1D update vector using top-k selection and error feedback.

        Args:
            delta: 1D parameter update tensor (w_local - w_global).
            client_id: Unique client identifier to track client-specific residual.
            ratio: Optional override for sparsification ratio (e.g. during warmup).

        Returns:
            Tuple of:
              - indices: 1D int64/int32 tensor of chosen coordinates.
              - values: 1D float tensor of values at those coordinates.
              - wire_bytes: Exact wire byte count for this compressed update.
        """
        total_elements = delta.numel()
        eff_ratio = float(ratio) if ratio is not None else self.ratio
        k = max(1, min(total_elements, int(math.ceil(total_elements * eff_ratio))))

        # 1. Incorporate local residual (error feedback)
        res = self._get_residual(client_id, delta.shape, delta.device, delta.dtype)
        accumulated = delta + res

        # 2. Select top-k entries by absolute magnitude
        if k >= total_elements:
            # Full density (no dropping)
            indices = torch.arange(total_elements, device=delta.device, dtype=torch.int64)
            values = accumulated
            self._save_residual(client_id, torch.zeros_like(accumulated))
        else:
            abs_vals = torch.abs(accumulated)
            _, topk_indices = torch.topk(abs_vals, k=k, sorted=False)
            indices = topk_indices
            values = accumulated[indices]

            # 3. Update client residual buffer: unsent entries remain
            new_res = accumulated.clone()
            new_res[indices] = 0.0
            self._save_residual(client_id, new_res)

        wire_bytes = compute_sparse_bytes(
            num_kept=k,
            index_bytes=4,
            value_bytes=self.value_dtype_bytes,
        )

        return indices, values, wire_bytes

    @staticmethod
    def decompress(
        indices: torch.Tensor,
        values: torch.Tensor,
        total_elements: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Reconstruct a dense 1D vector from sparse indices and values on the server.

        Args:
            indices: 1D tensor of non-zero positions.
            values: 1D tensor of values.
            total_elements: Length of the full parameter vector.
            device: Target device for reconstructed tensor.
            dtype: Target dtype for reconstructed tensor.

        Returns:
            Dense 1D tensor of length total_elements with values placed at indices.
        """
        target_device = device if device is not None else values.device
        target_dtype = dtype if dtype is not None else values.dtype

        dense = torch.zeros(total_elements, device=target_device, dtype=target_dtype)
        idx = indices.to(device=target_device, dtype=torch.int64)
        val = values.to(device=target_device, dtype=target_dtype)
        dense.scatter_(dim=0, index=idx, src=val)
        return dense

    def reset_residuals(self, client_id: Optional[str] = None) -> None:
        """Clear residual buffers (for a specific client or all)."""
        if client_id is not None:
            self.residuals.pop(client_id, None)
        else:
            self.residuals.clear()
