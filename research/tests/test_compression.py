"""Unit tests for Phase 3 compression modules.

Tests:
1. Comms byte accounting calculations.
2. Flattening and unflattening multi-dimensional tensors.
3. Top-k sparsification with error-feedback residual accumulation.
4. QSGD stochastic quantization unbiasedness and reconstruction.
5. ModelCompressor unified wrapper.
"""

import pytest
import torch
import math

from research.src.fl_core.compression.comms_accounting import (
    compute_uncompressed_bytes,
    compute_sparse_bytes,
    compute_quantized_bytes,
    calculate_compression_stats,
)
from research.src.fl_core.compression.sparsification import TopKSparsifier
from research.src.fl_core.compression.quantization import QSGDQuantizer
from research.src.fl_core.compression import (
    flatten_tensor_list,
    unflatten_tensor_list,
    ModelCompressor,
)


def test_comms_accounting_math():
    """Verify exact byte calculations against analytical formulas."""
    num_params = 1_000_000

    # Uncompressed fp32: exactly 4 MB
    uncompressed_bytes = compute_uncompressed_bytes(num_params, dtype_bytes=4)
    assert uncompressed_bytes == 4_000_000

    # 5% Top-k sparse: 50,000 kept elements.
    # (50,000 * 8 bytes) + 8 header = 400,008 bytes (~10x reduction)
    sparse_bytes = compute_sparse_bytes(50_000, index_bytes=4, value_bytes=4, header_bytes=8)
    assert sparse_bytes == 400_008

    stats = calculate_compression_stats(sparse_bytes, uncompressed_bytes)
    assert stats["compression_ratio"] == pytest.approx(10.0, rel=1e-2)
    assert stats["bandwidth_saved_pct"] == pytest.approx(90.0, rel=1e-2)

    # 8-bit QSGD: 1,000,000 * 1 byte + 12 overhead = 1,000,012 bytes (~4x reduction)
    q8_bytes = compute_quantized_bytes(num_params, bits=8, scale_bytes=4, header_bytes=8)
    assert q8_bytes == 1_000_012
    q8_stats = calculate_compression_stats(q8_bytes, uncompressed_bytes)
    assert q8_stats["compression_ratio"] == pytest.approx(4.0, rel=1e-2)

    # 4-bit QSGD: 500,000 bytes + 12 overhead (~8x reduction)
    q4_bytes = compute_quantized_bytes(num_params, bits=4, scale_bytes=4, header_bytes=8)
    assert q4_bytes == 500_012


def test_flatten_unflatten_invertibility():
    """Verify that flattening and unflattening perfectly restores tensor shapes and values."""
    t1 = torch.randn(32, 16, 3, 3)
    t2 = torch.randn(32)
    t3 = torch.randn(10, 5)

    flat, meta = flatten_tensor_list([t1, t2, t3])
    assert flat.numel() == t1.numel() + t2.numel() + t3.numel()
    assert flat.ndim == 1

    restored = unflatten_tensor_list(flat, meta)
    assert len(restored) == 3
    assert torch.equal(restored[0], t1)
    assert torch.equal(restored[1], t2)
    assert torch.equal(restored[2], t3)


def test_topk_sparsification_and_reconstruction():
    """Verify top-k picks the largest entries and places them at correct coordinates."""
    sparsifier = TopKSparsifier(ratio=0.2, error_feedback=False)

    # 10 elements, top 20% = top 2 elements
    delta = torch.tensor([0.1, 0.9, -0.8, 0.05, 0.2, -0.3, 0.15, -0.01, 0.04, 0.12])
    indices, values, wire_bytes = sparsifier.compress(delta, client_id="c1")

    assert indices.numel() == 2
    assert values.numel() == 2
    # The two largest in absolute magnitude are 0.9 (idx 1) and -0.8 (idx 2)
    chosen_indices = set(indices.tolist())
    assert chosen_indices == {1, 2}

    reconstructed = TopKSparsifier.decompress(indices, values, total_elements=10)
    assert reconstructed.shape == (10,)
    assert reconstructed[1] == pytest.approx(0.9)
    assert reconstructed[2] == pytest.approx(-0.8)
    assert reconstructed[0] == 0.0
    assert reconstructed[3] == 0.0


def test_topk_error_feedback_residual_accumulation():
    """Verify that unsent entries accumulate across rounds and are eventually transmitted."""
    sparsifier = TopKSparsifier(ratio=0.5, error_feedback=True)

    # Round 1: delta has 4 elements. Elements at idx 0 and 1 are large, idx 2 and 3 are small.
    r1_delta = torch.tensor([1.0, 1.0, 0.2, 0.1])
    idx1, val1, _ = sparsifier.compress(r1_delta, client_id="c1")

    # Indices 0 and 1 are transmitted
    assert set(idx1.tolist()) == {0, 1}

    # Verify residual has kept unsent values [0, 0, 0.2, 0.1]
    res1 = sparsifier.residuals["c1"]
    assert res1[0] == 0.0
    assert res1[1] == 0.0
    assert res1[2] == pytest.approx(0.2)
    assert res1[3] == pytest.approx(0.1)

    # Round 2: delta has a small update, but idx 2 gets enough added from residual to be transmitted
    r2_delta = torch.tensor([0.05, 0.05, 0.9, 0.05])
    # accumulated = [0.05, 0.05, 1.1, 0.15] -> Top 2 are idx 2 (1.1) and idx 3 (0.15)
    idx2, val2, _ = sparsifier.compress(r2_delta, client_id="c1")
    assert 2 in idx2.tolist()
    assert 3 in idx2.tolist()


def test_qsgd_unbiasedness():
    """Verify that QSGD expectation over multiple stochastic trials equals original vector."""
    quantizer = QSGDQuantizer(bits=4)

    # 100-dimensional vector
    torch.manual_seed(42)
    v = torch.randn(100) * 0.5

    # Average 500 stochastic quantizations
    samples = []
    for _ in range(500):
        q, norm, _ = quantizer.compress(v)
        v_hat = quantizer.decompress(q, norm)
        samples.append(v_hat)

    mean_estimate = torch.stack(samples, dim=0).mean(dim=0)
    # The mean estimate should closely match original v (unbiased estimator)
    max_abs_err = torch.max(torch.abs(mean_estimate - v)).item()
    assert max_abs_err < 0.05


def test_model_compressor_unified():
    """Verify ModelCompressor works across all methods (none, topk, qsgd)."""
    delta = torch.randn(1000)

    # 1. Uncompressed
    c_none = ModelCompressor({"method": "none"})
    p_none, b_none = c_none.compress_update(delta)
    rec_none = c_none.decompress_update(p_none)
    assert torch.equal(rec_none, delta.cpu())
    assert b_none == 4000

    # 2. Top-k
    c_topk = ModelCompressor({"method": "topk", "topk_ratio": 0.1})
    p_topk, b_topk = c_topk.compress_update(delta, client_id="client_01")
    rec_topk = c_topk.decompress_update(p_topk)
    assert rec_topk.shape == (1000,)
    assert b_topk < b_none

    # 3. QSGD
    c_qsgd = ModelCompressor({"method": "qsgd", "quant_bits": 8})
    p_qsgd, b_qsgd = c_qsgd.compress_update(delta)
    rec_qsgd = c_qsgd.decompress_update(p_qsgd)
    assert rec_qsgd.shape == (1000,)
    assert b_qsgd < b_none


def test_sparsifier_with_residual_dir(tmp_path):
    """Verify that residual_dir persists residuals across separate compressor instances."""
    res_dir = tmp_path / "residuals"
    delta1 = torch.tensor([1.0, 1.0, 0.2, 0.1])

    # Instance A (e.g. Actor 1)
    sp_a = TopKSparsifier(ratio=0.5, error_feedback=True, residual_dir=res_dir)
    idx1, val1, _ = sp_a.compress(delta1, client_id="c01")
    assert set(idx1.tolist()) == {0, 1}

    # Verify file was written
    assert (res_dir / "c01.pt").exists()

    # Instance B (e.g. Actor 2 spawned later)
    sp_b = TopKSparsifier(ratio=0.5, error_feedback=True, residual_dir=res_dir)
    delta2 = torch.tensor([0.05, 0.05, 0.9, 0.05])
    idx2, val2, _ = sp_b.compress(delta2, client_id="c01")

    # Due to preserved residual from Instance A, indices 2 and 3 should be selected!
    assert 2 in idx2.tolist()
    assert 3 in idx2.tolist()


def test_densenet_param_compression():
    """Verify that DenseNet parameter lists containing non-floating BatchNorm buffers compress cleanly."""
    import numpy as np
    from research.src.fl_core.models.densenet import build_densenet121, get_parameters, set_parameters

    model = build_densenet121(pretrained=False)
    local_params = get_parameters(model)
    initial_params = [np.array(p, copy=True) for p in local_params]

    float_indices = [
        i for i, lp in enumerate(local_params)
        if np.issubdtype(lp.dtype, np.floating)
    ]
    deltas = [
        torch.from_numpy(np.asarray(local_params[i] - initial_params[i])).float()
        for i in float_indices
    ]
    flat_delta, meta = flatten_tensor_list(deltas)

    compressor = ModelCompressor({"method": "topk", "topk_ratio": 0.05})
    payload, wire_bytes = compressor.compress_update(flat_delta, client_id="c_test")
    rec_flat = compressor.decompress_update(payload)
    rec_deltas = unflatten_tensor_list(rec_flat, meta)

    transmitted_params = list(local_params)
    for idx, rd in zip(float_indices, rec_deltas):
        ip = initial_params[idx]
        lp = local_params[idx]
        transmitted_params[idx] = (ip + rd.cpu().numpy()).astype(lp.dtype)

    set_parameters(model, transmitted_params)
    assert len(get_parameters(model)) == len(local_params)

