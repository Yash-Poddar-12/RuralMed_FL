"""Phase 4 results visualisation — 4 publication figures.

Run from repo root with venv active:
    python research/scripts/plot_phase4_results.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "infra/docs/reports/phase4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE2_AUROC = 0.9051
C_FEDBUFF  = "#4C72B0"
C_SYNC     = "#DD8452"
C_P2       = "#55A868"
C_COVID    = "#C44E52"
C_NEUTRAL  = "#8172B2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

fedbuff_dropout   = [0.00, 0.10, 0.25, 0.40, 0.60]
fedbuff_auroc     = [0.9141, 0.9134, 0.9056, 0.9154, 0.9051]
fedbuff_acc       = [0.8218, 0.8068, 0.7757, 0.7981, 0.8149]
fedbuff_f1        = [0.8393, 0.8332, 0.8048, 0.8297, 0.8338]
fedbuff_covid_sen = [0.9862, 0.9749, 0.8973, 0.9854, 0.9633]
sync_dropout_vals = [0.25, 0.40]
sync_auroc        = [0.9208, 0.9143]
sync_f1           = [0.8462, 0.8423]
sync_covid_sen    = [0.9820, 0.9929]
buffer_ks         = [1, 3, 5]
buffer_auroc      = [0.8881, 0.9056, 0.9244]
buffer_f1         = [0.6622, 0.8048, 0.8455]

# ── Figure 1: Dropout retention curve ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(fedbuff_dropout, fedbuff_auroc, "o-", color=C_FEDBUFF, lw=2, ms=7, label="FedBuff (K=3)")
ax.axhline(PHASE2_AUROC, color=C_P2, lw=1.5, ls="--", label="Phase 2 Ceiling")
ax.fill_between(
    fedbuff_dropout, fedbuff_auroc, PHASE2_AUROC,
    where=[v < PHASE2_AUROC for v in fedbuff_auroc],
    alpha=0.12, color=C_FEDBUFF,
)
ax.set_xlabel("Dropout rate")
ax.set_ylabel("Macro-AUROC")
ax.set_title("FedBuff AUROC Under Increasing Dropout")
ax.set_xticks(fedbuff_dropout)
ax.set_xticklabels([str(int(d * 100)) + "%" for d in fedbuff_dropout])
ax.set_ylim(0.84, 0.95)
ax.legend(loc="lower left")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.plot(fedbuff_dropout, fedbuff_acc, "o-", color=C_FEDBUFF, lw=2, ms=7, label="Accuracy")
ax.plot(fedbuff_dropout, fedbuff_f1, "s--", color=C_NEUTRAL, lw=2, ms=7, label="Macro-F1")
ax.set_xlabel("Dropout rate")
ax.set_ylabel("Score")
ax.set_title("Accuracy and Macro-F1 Under Increasing Dropout")
ax.set_xticks(fedbuff_dropout)
ax.set_xticklabels([str(int(d * 100)) + "%" for d in fedbuff_dropout])
ax.set_ylim(0.60, 0.90)
ax.legend(loc="lower left")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Phase 4: FedBuff Dropout Sweep", y=1.01, fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_dropout_retention_curve.png", bbox_inches="tight")
plt.close(fig)
print("fig1 saved")

# ── Figure 2: FedBuff vs Sync comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.arange(2)
w = 0.32
fb_a = [fedbuff_auroc[fedbuff_dropout.index(d)] for d in sync_dropout_vals]
fb_f = [fedbuff_f1[fedbuff_dropout.index(d)] for d in sync_dropout_vals]

ax = axes[0]
b1 = ax.bar(x - w / 2, fb_a, w, color=C_FEDBUFF, label="FedBuff (K=3)")
b2 = ax.bar(x + w / 2, sync_auroc, w, color=C_SYNC, label="Sync+Dropout")
ax.axhline(PHASE2_AUROC, color=C_P2, lw=1.5, ls="--", label="No-Dropout Ceiling")
ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([str(int(d * 100)) + "% Dropout" for d in sync_dropout_vals])
ax.set_ylabel("Macro-AUROC")
ax.set_title("FedBuff vs Sync - AUROC")
ax.set_ylim(0.87, 0.95)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
b1 = ax.bar(x - w / 2, fb_f, w, color=C_FEDBUFF, label="FedBuff (K=3)")
b2 = ax.bar(x + w / 2, sync_f1, w, color=C_SYNC, label="Sync+Dropout")
ax.bar_label(b1, fmt="%.4f", padding=3, fontsize=8)
ax.bar_label(b2, fmt="%.4f", padding=3, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([str(int(d * 100)) + "% Dropout" for d in sync_dropout_vals])
ax.set_ylabel("Macro-F1")
ax.set_title("FedBuff vs Sync - Macro-F1")
ax.set_ylim(0.77, 0.88)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Phase 4: FedBuff vs Sync+Dropout", y=1.01, fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_fedbuff_vs_sync_comparison.png", bbox_inches="tight")
plt.close(fig)
print("fig2 saved")

# ── Figure 3: Buffer size ablation ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
xk = np.arange(3)
cols = [C_COVID, C_FEDBUFF, C_P2]

ax = axes[0]
b = ax.bar(xk, buffer_auroc, 0.5, color=cols)
ax.bar_label(b, fmt="%.4f", padding=3, fontsize=9)
ax.axhline(PHASE2_AUROC, color="gray", lw=1.5, ls="--", label="No-Dropout Ceiling")
ax.set_xticks(xk)
ax.set_xticklabels(["K=" + str(k) for k in buffer_ks])
ax.set_ylabel("Macro-AUROC")
ax.set_title("Buffer Size Ablation - AUROC")
ax.set_ylim(0.83, 0.95)
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
b = ax.bar(xk, buffer_f1, 0.5, color=cols)
ax.bar_label(b, fmt="%.4f", padding=3, fontsize=9)
ax.set_xticks(xk)
ax.set_xticklabels(["K=" + str(k) for k in buffer_ks])
ax.set_ylabel("Macro-F1")
ax.set_title("Buffer Size Ablation - Macro-F1")
ax.set_ylim(0.55, 0.90)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Phase 4: Buffer Size Ablation (25% Dropout)", y=1.01, fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_buffer_size_ablation.png", bbox_inches="tight")
plt.close(fig)
print("fig3 saved")

# ── Figure 4: COVID-19 sensitivity ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(fedbuff_dropout, fedbuff_covid_sen, "o-", color=C_COVID, lw=2, ms=8,
        label="FedBuff COVID-19 Sensitivity")
ax.scatter(sync_dropout_vals, sync_covid_sen, s=100, marker="D", color=C_SYNC,
           zorder=5, label="Sync+Dropout COVID-19 Sensitivity")
ax.axhline(0.90, color="gray", lw=1, ls=":", label="90% clinical threshold")
ax.fill_between(
    fedbuff_dropout, fedbuff_covid_sen, 0.90,
    where=[v < 0.90 for v in fedbuff_covid_sen],
    alpha=0.15, color=C_COVID,
)
ax.set_xticks(fedbuff_dropout)
ax.set_xticklabels([str(int(d * 100)) + "%" for d in fedbuff_dropout])
ax.set_xlabel("Dropout rate")
ax.set_ylabel("COVID-19 Sensitivity")
ax.set_title("COVID-19 Sensitivity Under Rural Client Dropout")
ax.set_ylim(0.30, 1.05)
ax.legend(loc="lower left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_covid_sensitivity_vs_dropout.png", bbox_inches="tight")
plt.close(fig)
print("fig4 saved")

print("All 4 Phase 4 figures saved to:", OUT_DIR)
