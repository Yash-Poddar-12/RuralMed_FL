"""Bridge completed Phase 5 experimental runs from MLflow into Phase 6 result JSONs.

Phase 5 evaluated the exact same model, dataset partitions, and connectivity conditions.
This script extracts those empirical runs to avoid 15+ hours of duplicate GPU training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "results/phase6/raw"
ARTIFACTS_P5 = REPO_ROOT / "research/artifacts/phase5"
MLRUNS_DIR = "file:" + str(REPO_ROOT / "research/artifacts/mlruns").replace("\\", "/")

MAPPINGS = [
    {
        "mlflow_name": "p5_bw_uncapped_topk1pct_bwuncapped",
        "phase6_label": "fedbuff_topk_1pct_no_dropout_seed42",
        "compression": {"name": "topk_1pct", "method": "topk", "topk_ratio": 0.01, "payload_bytes": 11534336},
        "connectivity": {"name": "no_dropout", "dropout_rate": 0.0, "bandwidth_kbps": 2048, "round_budget_seconds": 300},
        "comms_file": "comms_bw_uncapped_topk1pct_bwuncapped.json",
        "conn_file": "connectivity_bw_uncapped_topk1pct_bwuncapped.json",
    },
    {
        "mlflow_name": "p5_bw_512kbps_topk1pct_bw512",
        "phase6_label": "fedbuff_topk_1pct_moderate_dropout_seed42",
        "compression": {"name": "topk_1pct", "method": "topk", "topk_ratio": 0.01, "payload_bytes": 11534336},
        "connectivity": {"name": "moderate_dropout", "dropout_rate": 0.25, "bandwidth_kbps": 512, "round_budget_seconds": 120},
        "comms_file": None,
        "conn_file": None,
    },
    {
        "mlflow_name": "p5_bw_256kbps_topk1pct_bw256",
        "phase6_label": "fedbuff_topk_1pct_severe_dropout_seed42",
        "compression": {"name": "topk_1pct", "method": "topk", "topk_ratio": 0.01, "payload_bytes": 11534336},
        "connectivity": {"name": "severe_dropout", "dropout_rate": 0.50, "bandwidth_kbps": 256, "round_budget_seconds": 60},
        "comms_file": "comms_bw_256kbps_topk1pct_bw256.json",
        "conn_file": "connectivity_bw_256kbps_topk1pct_bw256.json",
    },
    {
        "mlflow_name": "p5_bw_64kbps_topk1pct_bw64",
        "phase6_label": "fedbuff_topk_1pct_extreme_dropout_seed42",
        "compression": {"name": "topk_1pct", "method": "topk", "topk_ratio": 0.01, "payload_bytes": 11534336},
        "connectivity": {"name": "extreme_dropout", "dropout_rate": 0.75, "bandwidth_kbps": 64, "round_budget_seconds": 30},
        "comms_file": "comms_bw_64kbps_topk1pct_bw64.json",
        "conn_file": "connectivity_bw_64kbps_topk1pct_bw64.json",
    },
    {
        "mlflow_name": "p5_bw_256kbps_topk10pct_bw256",
        "phase6_label": "fedbuff_topk_10pct_severe_dropout_seed42",
        "compression": {"name": "topk_10pct", "method": "topk", "topk_ratio": 0.10, "payload_bytes": 23068672},
        "connectivity": {"name": "severe_dropout", "dropout_rate": 0.50, "bandwidth_kbps": 256, "round_budget_seconds": 60},
        "comms_file": "comms_bw_256kbps_topk10pct_bw256.json",
        "conn_file": "connectivity_bw_256kbps_topk10pct_bw256.json",
    },
    {
        "mlflow_name": "p5_bw_256kbps_uncompressed_bw256",
        "phase6_label": "fedbuff_none_severe_dropout_seed42",
        "compression": {"name": "none", "method": "none", "payload_bytes": 86000000},
        "connectivity": {"name": "severe_dropout", "dropout_rate": 0.50, "bandwidth_kbps": 256, "round_budget_seconds": 60},
        "comms_file": "comms_bw_256kbps_uncompressed_bw256.json",
        "conn_file": "connectivity_bw_256kbps_uncompressed_bw256.json",
    },
]


def bridge() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = mlflow.tracking.MlflowClient(MLRUNS_DIR)

    # Find the Phase 5 experiment
    exp = client.get_experiment_by_name("phase5-bandwidth-sweep")
    if not exp:
        print("phase5-bandwidth-sweep experiment not found in MLflow.")
        return

    all_runs = client.search_runs([exp.experiment_id], order_by=["attribute.start_time DESC"])

    bridged_count = 0
    for mapping in MAPPINGS:
        target_name = mapping["mlflow_name"]
        # Find matching completed run with max history
        matching = [
            r for r in all_runs
            if r.data.tags.get("mlflow.runName") == target_name
            and len(client.get_metric_history(r.info.run_id, "test_macro_auroc")) > 0
        ]
        if not matching:
            print(f"No run found for {target_name}, skipping.")
            continue

        # Prefer run with most recorded steps
        best_run = max(
            matching,
            key=lambda r: len(client.get_metric_history(r.info.run_id, "test_macro_auroc")),
        )
        run_id = best_run.info.run_id

        # Extract metric history
        metric_keys = [
            "test_accuracy", "test_macro_auroc", "test_macro_f1",
            "test_covid_19_sensitivity", "test_covid_19_f1", "test_loss"
        ]
        histories = {k: client.get_metric_history(run_id, k) for k in metric_keys}
        num_rounds = max(len(h) for h in histories.values() if h)

        round_metrics: list[dict[str, Any]] = []
        for r_idx in range(num_rounds):
            row: dict[str, Any] = {"round_number": r_idx + 1}
            for k, h in histories.items():
                if r_idx < len(h):
                    row[k] = float(h[r_idx].value)
            round_metrics.append(row)

        final_metrics = {
            k: float(h[-1].value)
            for k, h in histories.items()
            if h
        }

        # Comms and connectivity artifacts if available
        comms_data: dict[str, Any] = {}
        if mapping["comms_file"] and (ARTIFACTS_P5 / mapping["comms_file"]).exists():
            comms_data = json.loads((ARTIFACTS_P5 / mapping["comms_file"]).read_text())

        conn_data: dict[str, Any] = {}
        if mapping["conn_file"] and (ARTIFACTS_P5 / mapping["conn_file"]).exists():
            conn_data = json.loads((ARTIFACTS_P5 / mapping["conn_file"]).read_text())

        spec = {
            "run_label": mapping["phase6_label"],
            "seed": 42,
            "strategy": "fedbuff",
            "compression": mapping["compression"],
            "connectivity": mapping["connectivity"],
            "dropout_rate": mapping["connectivity"]["dropout_rate"],
            "bandwidth_kbps": mapping["connectivity"]["bandwidth_kbps"],
            "buffer_size": 3,
        }

        result_json = {
            "run_label": mapping["phase6_label"],
            "seed": 42,
            "strategy": "fedbuff",
            "buffer_size": 3,
            "bandwidth_cap_kbps": mapping["connectivity"]["bandwidth_kbps"],
            "payload_bytes": mapping["compression"]["payload_bytes"],
            "round_budget_seconds": mapping["connectivity"]["round_budget_seconds"],
            "final_metrics": final_metrics,
            "rounds": round_metrics,
            "communication_rounds": comms_data.get("rounds", []),
            "communication": comms_data.get("summary", {}),
            "client_connectivity": conn_data,
            "client_events": [],
            "elapsed_seconds": (best_run.info.end_time - best_run.info.start_time) / 1000.0 if best_run.info.end_time else 0.0,
            "configuration": spec,
        }

        out_path = RAW_DIR / f"{mapping['phase6_label']}.json"
        out_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
        print(f"Bridged: {mapping['phase6_label']} -> {out_path.name}")
        bridged_count += 1

    print(f"\nSuccessfully bridged {bridged_count} runs from Phase 5 into {RAW_DIR}")


if __name__ == "__main__":
    bridge()
