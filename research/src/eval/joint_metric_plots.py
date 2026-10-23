"""Phase 6 aggregation, joint-metric heatmaps, and operating-point selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.src.eval.supabase_writer import load_results


METRICS = ("test_accuracy", "test_macro_auroc", "test_macro_f1")


def summarize_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        config = result.get("configuration", {})
        final = result.get("final_metrics", {})
        comp = config.get("compression", {})
        comp_name = comp.get("name", "unknown")
        if comp_name == "topk" or "topk" in comp_name:
            ratio = comp.get("topk_ratio")
            if ratio == 0.01 or "1pct" in result.get("run_label", ""):
                comp_name = "topk_1pct"
            elif ratio == 0.10 or "10pct" in result.get("run_label", ""):
                comp_name = "topk_10pct"
        rows.append({
            "run_label": result["run_label"],
            "seed": result.get("seed"),
            "strategy": result.get("strategy", "fedbuff"),
            "compression": comp_name,
            "bandwidth_kbps": config.get("bandwidth_kbps"),
            "dropout_rate": config.get("dropout_rate", 0.0),
            **{metric: final.get(metric, np.nan) for metric in METRICS},
        })
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError("No result rows were found")
    group_columns = [
        "strategy", "compression", "bandwidth_kbps", "dropout_rate"
    ]
    summary = (
        raw.groupby(group_columns, dropna=False)
        .agg(
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in METRICS
            },
            **{
                f"{metric}_std": (metric, "std")
                for metric in METRICS
            },
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    std_columns = [f"{metric}_std" for metric in METRICS]
    summary.loc[:, std_columns] = summary[std_columns].fillna(0.0)
    return summary


def recommend_operating_point(summary: pd.DataFrame, min_seeds: int = 1) -> dict[str, Any]:
    """Select the highest-AUROC point subject to a practical efficiency floor."""
    candidates = summary.copy()
    candidates.loc[:, "bandwidth_score"] = candidates["bandwidth_kbps"].fillna(0)
    candidates.loc[:, "efficiency"] = 1.0 / (1.0 + candidates["bandwidth_score"])
    req_seeds = 3 if (candidates["seeds"] >= 3).any() else max(1, min_seeds)
    candidates = candidates[candidates["seeds"] >= req_seeds]
    if candidates.empty:
        raise ValueError(f"At least {req_seeds} seeds are required for a recommendation")
    # Prefer a FedBuff compressed point, then any compressed point, then maximize AUROC and F1.
    fedbuff_compressed = candidates[(candidates["compression"] != "none") & (candidates["strategy"] == "fedbuff")]
    compressed = candidates[candidates["compression"] != "none"]
    pool = fedbuff_compressed if not fedbuff_compressed.empty else (compressed if not compressed.empty else candidates)
    best = pool.sort_values(
        ["test_macro_auroc_mean", "test_macro_f1_mean", "bandwidth_score"],
        ascending=[False, False, True],
    ).iloc[0]
    return {
        "strategy": best["strategy"],
        "compression": best["compression"],
        "bandwidth_kbps": None if pd.isna(best["bandwidth_kbps"]) else float(best["bandwidth_kbps"]),
        "dropout_rate": float(best["dropout_rate"]),
        "macro_auroc_mean": float(best["test_macro_auroc_mean"]),
        "macro_auroc_std": float(best["test_macro_auroc_std"]),
        "macro_f1_mean": float(best["test_macro_f1_mean"]),
        "accuracy_mean": float(best["test_accuracy_mean"]),
        "seeds": int(best["seeds"]),
        "justification": f"Highest mean macro-AUROC among compressed points with at least {req_seeds} seed(s); standard deviation is reported for uncertainty.",
    }


def _heatmap(summary: pd.DataFrame, value: str, title: str, path: Path) -> None:
    frame = summary.copy()
    frame["bandwidth_label"] = frame["bandwidth_kbps"].map(
        lambda value: "uncapped" if pd.isna(value) else str(int(value))
    )
    pivot = frame.pivot_table(
        index="dropout_rate",
        columns="bandwidth_label",
        values=value,
        aggfunc="mean",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns.tolist())
    ax.set_yticks(range(len(pivot.index)), labels=[f"{v:.0%}" for v in pivot.index.tolist()])
    ax.set_xlabel("Bandwidth cap (kbps; uncapped where shown)")
    ax.set_ylabel("Dropout condition")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=value.replace("_", " "))
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value_at_cell = pivot.iloc[row, col]
            if pd.notna(value_at_cell):
                ax.text(col, row, f"{value_at_cell:.3f}", ha="center", va="center", color="white")
    fig.subplots_adjust(left=0.14, right=0.90, bottom=0.18, top=0.88)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def generate_outputs(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    results = load_results(results_dir)
    summary = summarize_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    _heatmap(
        summary,
        "test_accuracy_mean",
        "Phase 6 joint metric: retained accuracy",
        output_dir / "accuracy_heatmap.png",
    )
    _heatmap(
        summary,
        "test_macro_auroc_mean",
        "Phase 6 joint metric: macro-AUROC",
        output_dir / "auroc_heatmap.png",
    )
    async_summary = summary[summary["strategy"] == "fedbuff"]
    sync_summary = summary[summary["strategy"] == "sync_dropout"]
    if not sync_summary.empty:
        diff = async_summary.merge(
            sync_summary,
            on=["compression", "bandwidth_kbps", "dropout_rate"],
            suffixes=("_async", "_sync"),
        )
        if not diff.empty:
            diff["macro_auroc_difference"] = (
                diff["test_macro_auroc_mean_async"] -
                diff["test_macro_auroc_mean_sync"]
            )
            diff.to_csv(output_dir / "async_minus_sync.csv", index=False)
            _heatmap(
                diff.rename(columns={"test_macro_auroc_mean_async": "async_auroc"}),
                "async_auroc",
                "FedBuff macro-AUROC under joint conditions",
                output_dir / "async_heatmap.png",
            )
    recommendation = recommend_operating_point(summary)
    (output_dir / "recommended_operating_point.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )
    (output_dir / "recommended_operating_point.md").write_text(
        "# Phase 6 recommended operating point\n\n"
        f"- Strategy: `{recommendation['strategy']}`\n"
        f"- Compression: `{recommendation['compression']}`\n"
        f"- Bandwidth cap: `{recommendation['bandwidth_kbps']}` kbps\n"
        f"- Dropout condition: `{recommendation['dropout_rate']:.0%}`\n"
        f"- Macro-AUROC: `{recommendation['macro_auroc_mean']:.4f} +/- {recommendation['macro_auroc_std']:.4f}`\n"
        f"- Macro-F1: `{recommendation['macro_f1_mean']:.4f}`\n"
        f"- Accuracy: `{recommendation['accuracy_mean']:.4f}`\n\n"
        f"{recommendation['justification']}\n",
        encoding="utf-8",
    )
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Phase 6 joint metrics")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase6"))
    args = parser.parse_args()
    print(json.dumps(generate_outputs(args.results_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
