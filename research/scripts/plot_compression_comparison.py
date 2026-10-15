"""Generate publication-ready comparison graphs comparing compression schemes.

Compares:
1. Uncompressed FP32 (Phase 2 Baseline)
2. FP16 Half-Precision
3. QSGD 8-bit Stochastic Quantization
4. QSGD 4-bit Stochastic Quantization
5. Top-k 20% DGC + BatchNorm Preservation
6. Top-k 10% DGC + BatchNorm Preservation
7. Top-k 5% DGC + BatchNorm Preservation (Phase 3 Tested Model)
8. Top-k 1% DGC + BatchNorm Preservation (Extreme Sparsity)

Computes exact wire-level byte accounting for DenseNet-121 and renders high-res figures.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from research.src.data_prep.common import REPO_ROOT
from research.src.fl_core.models.densenet import build_densenet121
from research.src.fl_core.compression.comms_accounting import (
    compute_uncompressed_bytes,
    compute_sparse_bytes,
    compute_quantized_bytes,
)

OUT_DIR = REPO_ROOT / "research/artifacts/phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_model_parameter_breakdown():
    """Count non-BN vs BN parameters in DenseNet-121."""
    model = build_densenet121(num_classes=3, pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    
    bn_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch_bn_types())):
            bn_params += sum(p.numel() for p in module.parameters(recurse=False))
            
    non_bn_params = total_params - bn_params
    return total_params, non_bn_params, bn_params


def torch_bn_types():
    import torch.nn as nn
    return (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def main():
    print("Computing exact DenseNet-121 parameter breakdown ...")
    total_params, non_bn_params, bn_params = get_model_parameter_breakdown()
    print(f"  Total parameters : {total_params:,}")
    print(f"  Non-BN parameters: {non_bn_params:,} ({non_bn_params/total_params*100:.2f}%)")
    print(f"  BN parameters    : {bn_params:,} ({bn_params/total_params*100:.2f}%)")

    # Methods to evaluate:
    schemes = [
        {"name": "Uncompressed FP32\n(Phase 2 Baseline)", "type": "baseline_fp32", "marker": "o", "color": "#1f77b4"},
        {"name": "FP16 Baseline", "type": "fp16", "marker": "s", "color": "#aec7e8"},
        {"name": "QSGD 8-bit", "type": "qsgd", "bits": 8, "marker": "^", "color": "#ff7f0e"},
        {"name": "QSGD 4-bit", "type": "qsgd", "bits": 4, "marker": "v", "color": "#ffbb78"},
        {"name": "Top-k 20% DGC\n+ BN Bypass", "type": "topk", "ratio": 0.20, "marker": "D", "color": "#2ca02c"},
        {"name": "Top-k 10% DGC\n+ BN Bypass", "type": "topk", "ratio": 0.10, "marker": "P", "color": "#98df8a"},
        {"name": "Top-k 5% DGC\n(Phase 3 Tested)", "type": "topk", "ratio": 0.05, "marker": "*", "color": "#d62728"},
        {"name": "Top-k 1% DGC\n+ BN Bypass", "type": "topk", "ratio": 0.01, "marker": "X", "color": "#9467bd"},
    ]

    results = []
    baseline_bytes = compute_uncompressed_bytes(total_params, dtype_bytes=4)

    for s in schemes:
        stype = s["type"]
        if stype == "baseline_fp32":
            wire_bytes = baseline_bytes
            bn_wire = compute_uncompressed_bytes(bn_params, dtype_bytes=4)
            main_wire = wire_bytes - bn_wire
        elif stype == "fp16":
            wire_bytes = compute_uncompressed_bytes(total_params, dtype_bytes=2)
            bn_wire = compute_uncompressed_bytes(bn_params, dtype_bytes=2)
            main_wire = wire_bytes - bn_wire
        elif stype == "qsgd":
            bits = s["bits"]
            # Non-BN parameters quantized to `bits`, BN parameters transmitted uncompressed FP32
            main_wire = compute_quantized_bytes(non_bn_params, bits=bits)
            bn_wire = compute_uncompressed_bytes(bn_params, dtype_bytes=4)
            wire_bytes = main_wire + bn_wire
        elif stype == "topk":
            ratio = s["ratio"]
            num_kept = int(np.ceil(non_bn_params * ratio))
            # Non-BN parameters sparsified with Top-k (4-byte index + 4-byte value), BN intact FP32
            main_wire = compute_sparse_bytes(num_kept=num_kept)
            bn_wire = compute_uncompressed_bytes(bn_params, dtype_bytes=4)
            wire_bytes = main_wire + bn_wire
        else:
            raise ValueError(f"Unknown scheme type: {stype}")

        upload_mb = wire_bytes / (1024 * 1024)
        main_mb = main_wire / (1024 * 1024)
        bn_mb = bn_wire / (1024 * 1024)
        compression_ratio = baseline_bytes / wire_bytes
        pct_reduction = (1.0 - (wire_bytes / baseline_bytes)) * 100.0
        
        # 100-round total: 5 sampled clients per round x 100 rounds = 500 client uploads
        total_100_rounds_gb = (wire_bytes * 5 * 100) / (1024 ** 3)

        results.append({
            "name": s["name"],
            "label": s["name"].replace("\n", " "),
            "color": s["color"],
            "marker": s["marker"],
            "wire_bytes": wire_bytes,
            "upload_mb": upload_mb,
            "main_mb": main_mb,
            "bn_mb": bn_mb,
            "ratio": compression_ratio,
            "pct_reduction": pct_reduction,
            "total_100_gb": total_100_rounds_gb,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Plot: Publication-grade 4-Panel Comparison
    # ──────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    names = [r["name"] for r in results]
    x_pos = np.arange(len(names))
    colors = [r["color"] for r in results]

    # Panel A: Per-Client Upload Payload (MB) with Stacked BN contribution
    ax = axes[0, 0]
    main_bars = [r["main_mb"] for r in results]
    bn_bars = [r["bn_mb"] for r in results]
    ax.bar(x_pos, main_bars, label="Model Parameters (Compressed)", color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.bar(x_pos, bn_bars, bottom=main_bars, label="BatchNorm Parameters (Preserved FP32)", color="#7f7f7f", alpha=0.6, hatch="//", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Uplink Payload per Client Round (MB)", fontsize=11)
    ax.set_title("(A) Per-Round Client Uplink Payload Size", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")
    for i, r in enumerate(results):
        ax.text(i, r["upload_mb"] + 0.5, f"{r['upload_mb']:.2f} MB", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel B: Cumulative 100-Round Fleet Uplink Volume (GB)
    ax = axes[0, 1]
    fleet_gb = [r["total_100_gb"] for r in results]
    bars = ax.bar(x_pos, fleet_gb, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Cumulative Uplink Volume (GB)", fontsize=11)
    ax.set_title("(B) 100-Round Total Network Uplink (5 Clients / Round)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    for i, val in enumerate(fleet_gb):
        ax.text(i, val + 0.25, f"{val:.2f} GB", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel C: Uplink Compression Multiplier vs Baseline
    ax = axes[1, 0]
    ratios = [r["ratio"] for r in results]
    ax.plot(x_pos, ratios, color="#2b5c8f", linestyle="-", linewidth=2.2, marker="o", markersize=7)
    for i, (x, y) in enumerate(zip(x_pos, ratios)):
        ax.annotate(
            f"{y:.2f}×\n({results[i]['pct_reduction']:.1f}%)",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Compression Multiplier (× Baseline)", fontsize=11)
    ax.set_title("(C) Bandwidth Compression Ratio", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0, max(ratios) * 1.25)

    # Panel D: Tradeoff Comparison Table / Summary
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [["Scheme", "MB/Client", "100-Rnd GB", "Ratio", "Residual Memory"]]
    for r in results:
        mem = "26.22 MB" if "Top-k" in r["name"] else "None (0 MB)"
        table_data.append([
            r["label"],
            f"{r['upload_mb']:.2f} MB",
            f"{r['total_100_gb']:.2f} GB",
            f"{r['ratio']:.2f}×",
            mem,
        ])

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.32, 0.17, 0.17, 0.14, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    
    # Highlight header and Phase 3 row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif row == 7:  # Phase 3 row (Top-k 5%)
            cell.set_facecolor("#ffe0e0")
            cell.set_text_props(weight="bold")
    ax.set_title("(D) Quantitative Comparison & Memory Trade-off", fontsize=12, fontweight="bold", pad=12)

    fig.suptitle(
        "RuralMed-FL: Architectural Comparison of Communication Compression Schemes\nDenseNet-121 (6.96M Parameters) Across 10 Heterogeneous Clinical Partitions",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_path = OUT_DIR / "compression_methods_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure: {out_path}")


if __name__ == "__main__":
    main()
