"""Communication byte accounting for federated learning compression.

Computes exact wire-level bytes for:
1. Uncompressed baseline (float32 / float16)
2. Top-k sparsification (indices + values)
3. Stochastic quantization (packed bits + scale factors)
"""

from __future__ import annotations

import math
from typing import Dict, Any


def compute_uncompressed_bytes(num_params: int, dtype_bytes: int = 4) -> int:
    """Bytes required to transmit full uncompressed parameter vector.

    Args:
        num_params: Total number of scalar parameters.
        dtype_bytes: Bytes per scalar (4 for float32, 2 for float16).
    """
    return int(num_params * dtype_bytes)


def compute_sparse_bytes(
    num_kept: int,
    index_bytes: int = 4,
    value_bytes: int = 4,
    header_bytes: int = 8,
) -> int:
    """Exact wire bytes required to transmit a sparse update (indices + values).

    Args:
        num_kept: Number of non-zero entries transmitted (k).
        index_bytes: Bytes per index (4 for uint32/int32, covers up to 4.29B params).
        value_bytes: Bytes per value (4 for float32, 2 for float16).
        header_bytes: Metadata overhead (e.g. k count and total_size int32).
    """
    return int(num_kept * (index_bytes + value_bytes) + header_bytes)


def compute_quantized_bytes(
    num_params: int,
    bits: int = 8,
    scale_bytes: int = 4,
    header_bytes: int = 8,
) -> int:
    """Exact wire bytes required to transmit a bit-packed quantized update.

    Args:
        num_params: Total number of scalar parameters.
        bits: Number of bits per scalar (8, 4, 2, or 1).
        scale_bytes: Bytes for norm / scale factor (4 for float32).
        header_bytes: Metadata overhead (bit-width and length fields).
    """
    packed_payload_bytes = math.ceil(num_params * bits / 8.0)
    return int(packed_payload_bytes + scale_bytes + header_bytes)


def calculate_compression_stats(
    compressed_bytes: int,
    baseline_bytes: int,
) -> Dict[str, Any]:
    """Calculate ratio and bandwidth savings percentage against baseline."""
    if compressed_bytes <= 0:
        ratio = 1.0
        saving_pct = 0.0
    else:
        ratio = float(baseline_bytes) / float(compressed_bytes)
        saving_pct = (1.0 - float(compressed_bytes) / float(baseline_bytes)) * 100.0

    return {
        "compressed_bytes": compressed_bytes,
        "baseline_bytes": baseline_bytes,
        "compression_ratio": round(ratio, 2),
        "bandwidth_saved_pct": round(saving_pct, 2),
    }
