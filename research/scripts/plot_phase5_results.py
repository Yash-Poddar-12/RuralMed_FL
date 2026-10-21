"""Phase 5 results visualisation — 4 publication figures.

Plots bandwidth degradation curves, joint compression × bandwidth interactions,
and per-tier clinic uptime diagnostics from Phase 5 experiments.

Run from repo root with venv active:
    python research/scripts/plot_phase5_results.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "infra/docs/reports/phase5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE2_AUROC = 0.9051

# Visual palette matching Phase 4 and modern publication aesthetics
C_PRIMARY   = "#2B5C8F"    # Deep Slate Blue
C_ACCENT    = "#D95F02"    # Rust Amber
C_SUCCESS   = "#2CA02C"    # Forest Green
C_COVID     = "#D62728"    # Crimson Red
C_MUTED     = "#7570B3"    # Purple
C_GRAY      = "#7F7F7F"    # Slate Gray
C_BG_LIGHT  = "#F8F9FA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Empirical Phase 5 Data ────────────────────────────────────────────────────
# Axis 1: Bandwidth cap sweep at fixed TopK 1% compression
# bw_caps in kbps: 64, 128, 256, 512, 1024, Uncapped (represented at 2048 kbps)
bw_labels  = ["64k", "128k", "256k", "512k", "1024k", "Uncapped"]
bw_x       = [64, 128, 256, 512, 1024, 2048]
axis1_auroc = [0.9254, 0.9278, 0.9256, 0.9131, 0.9212, 0.9025]
axis1_acc   = [0.7828, 0.8171, 0.8167, 0.7981, 0.7937, 0.7873]
axis1_f1    = [0.8216, 0.8432, 0.8432, 0.8126, 0.8172, 0.8168]
axis1_covid = [0.9720, 0.9858, 0.9900, 0.8342, 0.8814, 0.9541]

# Axis 2: Joint Bandwidth × Compression Interaction (at 256 kbps)
interaction_labels    = ["TopK 1%\n(11.5 MB)", "TopK 10%\n(23.0 MB)", "Uncompressed\n(86.0 MB)"]
interaction_auroc     = [0.9256, 0.9241, 0.9082]
interaction_f1        = [0.8432, 0.8433, 0.7842]
interaction_covid     = [0.9900, 0.9916, 0.7608]
interaction_uptime    = [55.6, 38.4, 30.4]

# Per-tier uptime data across 256 kbps runs
tiers = ["Stable (n=3)\n[≥10k imgs]", "Intermittent (n=3)\n[4k-10k imgs]", "Unreliable (n=4)\n[<4k imgs]"]
# Uptimes for TopK 1% vs Uncompressed
# Stable: client 00, 02, 03; Intermittent: 06, 07, 08; Unreliable: 01, 04, 05, 09
with open(REPO_ROOT / "research/artifacts/phase5/connectivity_bw_256kbps_topk1pct_bw256.json") as f:
    topk1_diag = json.load(f)
with open(REPO_ROOT / "research/artifacts/phase5/connectivity_bw_256kbps_uncompressed_bw256.json") as f:
    uncomp_diag = json.load(f)

stable_cids = [0, 2, 3]
inter_cids  = [6, 7, 8]
unrel_cids  = [1, 4, 5, 9]

topk1_stable_up = np.mean([topk1_diag[f"uptime/client_{i:02d}"] for i in stable_cids]) * 100
topk1_inter_up  = np.mean([topk1_diag[f"uptime/client_{i:02d}"] for i in inter_cids]) * 100
topk1_unrel_up  = np.mean([topk1_diag[f"uptime/client_{i:02d}"] for i in unrel_cids]) * 100

uncomp_stable_up = np.mean([uncomp_diag[f"uptime/client_{i:02d}"] for i in stable_cids]) * 100
uncomp_inter_up  = np.mean([uncomp_diag[f"uptime/client_{i:02d}"] for i in inter_cids]) * 100
uncomp_unrel_up  = np.mean([uncomp_diag[f"uptime/client_{i:02d}"] for i in unrel_cids]) * 100

topk1_tier_uptimes  = [topk1_stable_up, topk1_inter_up, topk1_unrel_up]
uncomp_tier_uptimes = [uncomp_stable_up, uncomp_inter_up, uncomp_unrel_up]


# ── Figure 1: Bandwidth Degradation Curve (Axis 1) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: AUROC
ax = axes[0]
ax.plot(range(len(bw_labels)), axis1_auroc, "o-", color=C_PRIMARY, lw=2.5, ms=8, label="FedBuff + TopK 1%")
ax.axhline(PHASE2_AUROC, color=C_SUCCESS, lw=1.8, ls="--", label=f"Phase 2 Baseline ({PHASE2_AUROC:.4f})")
ax.fill_between(range(len(bw_labels)), axis1_auroc, PHASE2_AUROC, alpha=0.12, color=C_PRIMARY)
ax.set_xticks(range(len(bw_labels)))
ax.set_xticklabels(bw_labels)
ax.set_xlabel("Bandwidth Cap (kbps)")
ax.set_ylabel("Macro-AUROC")
ax.set_title("AUROC Retained Across Bandwidth Caps")
ax.set_ylim(0.88, 0.94)
ax.legend(loc="lower right", frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.6)

# Annotate robustness
ax.annotate(
    "Robust at 64 kbps!\nAUROC: 0.9254",
    xy=(0, axis1_auroc[0]),
    xytext=(0.4, 0.932),
    arrowprops=dict(facecolor=C_PRIMARY, shrink=0.08, width=1.2, headwidth=6),
    fontsize=9.5,
    fontweight="bold",
    color=C_PRIMARY,
)

# Subplot 2: Accuracy & Macro-F1
ax = axes[1]
ax.plot(range(len(bw_labels)), axis1_acc, "s-", color=C_PRIMARY, lw=2.2, ms=7, label="Test Accuracy")
ax.plot(range(len(bw_labels)), axis1_f1, "^--", color=C_MUTED, lw=2.2, ms=7, label="Macro-F1")
ax.set_xticks(range(len(bw_labels)))
ax.set_xticklabels(bw_labels)
ax.set_xlabel("Bandwidth Cap (kbps)")
ax.set_ylabel("Score")
ax.set_title("Accuracy & Macro-F1 Under Bandwidth Throttling")
ax.set_ylim(0.75, 0.88)
ax.legend(loc="lower right", frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.6)

plt.tight_layout()
fig1_path = OUT_DIR / "fig1_bandwidth_degradation_curve.png"
fig.savefig(fig1_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fig1_path}")


# ── Figure 2: Joint Bandwidth × Compression Interaction (Axis 2) ──────────────
fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(interaction_labels))
width = 0.26

rects1 = ax.bar(x - width, [v * 100 for v in interaction_auroc], width, label="Macro-AUROC (×100)", color=C_PRIMARY)
rects2 = ax.bar(x,         [v * 100 for v in interaction_f1],    width, label="Macro-F1 (×100)",    color=C_MUTED)
rects3 = ax.bar(x + width, [v * 100 for v in interaction_covid], width, label="COVID Sensitivity (%)", color=C_COVID)

ax.set_ylabel("Score (%)")
ax.set_title("Joint Bandwidth × Compression Interaction at 256 kbps Cap")
ax.set_xticks(x)
ax.set_xticklabels(interaction_labels)
ax.set_ylim(60, 105)
ax.legend(loc="lower left", frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.6)

# Value labels on bars
def autolabel(rects, fmt="{:.1f}%"):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

autolabel(rects1, "{:.1f}")
autolabel(rects2, "{:.1f}")
autolabel(rects3, "{:.1f}%")

# Annotate collapse
ax.annotate(
    "-23.0% Sensitivity Drop\n(Straggler Discard Collapse)",
    xy=(2 + width, interaction_covid[2] * 100),
    xytext=(1.65, 84),
    arrowprops=dict(facecolor=C_COVID, shrink=0.08, width=1.5, headwidth=6),
    fontsize=9.5,
    fontweight="bold",
    color=C_COVID,
)

plt.tight_layout()
fig2_path = OUT_DIR / "fig2_joint_bandwidth_compression_interaction.png"
fig.savefig(fig2_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fig2_path}")


# ── Figure 3: Clinic Tier Uptime & Participation Loss ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(tiers))
width = 0.35

rects1 = ax.bar(x - width/2, topk1_tier_uptimes, width, label="TopK 1% (11.5 MB payload)", color=C_SUCCESS)
rects2 = ax.bar(x + width/2, uncomp_tier_uptimes, width, label="Uncompressed (86.0 MB payload)", color=C_ACCENT)

ax.set_ylabel("Average Effective Uptime (%)")
ax.set_title("Effective Clinic Uptime by Infrastructure Tier (at 256 kbps)")
ax.set_xticks(x)
ax.set_xticklabels(tiers)
ax.set_ylim(0, 100)
ax.legend(loc="upper right", frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.6)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_SUCCESS)

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_ACCENT)

plt.tight_layout()
fig3_path = OUT_DIR / "fig3_client_uptime_and_tier_distribution.png"
fig.savefig(fig3_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fig3_path}")


# ── Figure 4: COVID-19 Clinical Sensitivity vs Bandwidth ──────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.8))

ax.plot(range(len(bw_labels)), [v * 100 for v in axis1_covid], "o-", color=C_COVID, lw=2.5, ms=8, label="TopK 1% (Compressed, 11.5 MB)")
# Plot uncompressed baseline comparison at 256 kbps (index 2)
ax.plot(2, interaction_covid[2] * 100, "X", color="#1F1F1F", ms=12, label="Uncompressed at 256k (86.0 MB)")

ax.set_xticks(range(len(bw_labels)))
ax.set_xticklabels(bw_labels)
ax.set_xlabel("Bandwidth Cap (kbps)")
ax.set_ylabel("COVID-19 Sensitivity (%)")
ax.set_title("Clinical Safety: COVID-19 Sensitivity Retention Across Bandwidth")
ax.set_ylim(70, 102)
ax.legend(loc="lower left", frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.6)

ax.annotate(
    "Uncompressed: 76.1%\nHeavy straggler drops",
    xy=(2, interaction_covid[2] * 100),
    xytext=(2.2, 78),
    arrowprops=dict(facecolor="#1F1F1F", shrink=0.08, width=1.2, headwidth=6),
    fontsize=9.5,
    fontweight="bold",
    color="#1F1F1F",
)

ax.annotate(
    "TopK 1%: 99.0%\nFull minority recall",
    xy=(2, axis1_covid[2] * 100),
    xytext=(1.2, 94),
    arrowprops=dict(facecolor=C_COVID, shrink=0.08, width=1.2, headwidth=6),
    fontsize=9.5,
    fontweight="bold",
    color=C_COVID,
)

plt.tight_layout()
fig4_path = OUT_DIR / "fig4_covid_sensitivity_vs_bandwidth.png"
fig.savefig(fig4_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fig4_path}")

print("\nAll 4 Phase 5 figures generated successfully in infra/docs/reports/phase5/")
