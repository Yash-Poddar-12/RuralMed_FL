import json
from pathlib import Path

import pandas as pd

from research.src.eval.experiment_runner import expand_sweep, run_sweep
from research.src.eval.joint_metric_plots import recommend_operating_point, summarize_results
from research.src.eval.supabase_writer import build_rows


def _config():
    return {
        "experiment_name": "test",
        "seed": 42,
        "buffer_size": 3,
        "sweep": {
            "seeds": [42, 43, 44],
            "strategies": ["fedbuff"],
            "compression": [{"name": "topk_1pct", "method": "topk", "topk_ratio": 0.01}],
            "connectivity": [{
                "name": "moderate",
                "dropout_rate": 0.25,
                "bandwidth_kbps": 512,
                "round_budget_seconds": 120,
                "phase5_run_name": "bw_512kbps_topk1pct",
            }],
        },
    }


def test_expand_sweep_preserves_three_seed_grid():
    runs = expand_sweep(_config())
    assert len(runs) == 3
    assert {run["seed"] for run in runs} == {42, 43, 44}
    assert runs[0]["strategy"] == "fedbuff"


def test_dry_run_writes_manifest(tmp_path):
    config_path = tmp_path / "phase6.yml"
    config_path.write_text(
        "experiment_name: test\n"
        "seed: 42\n"
        "buffer_size: 3\n"
        "sweep:\n"
        "  seeds: [42]\n"
        "  strategies: [fedbuff]\n"
        "  compression: [{name: none, method: none, payload_bytes: 1}]\n"
        "  connectivity: [{name: moderate, dropout_rate: 0.25, bandwidth_kbps: 512, phase5_run_name: bw_512kbps_topk1pct}]\n",
        encoding="utf-8",
    )
    manifest = run_sweep(config_path, tmp_path / "raw", dry_run=True)
    assert manifest["planned_runs"] == 1
    assert json.loads((tmp_path / "raw/manifest.json").read_text())["runs"][0]["status"] == "planned"


def test_summary_and_recommendation_use_seed_variance():
    results = []
    for seed, auroc in zip([42, 43, 44], [0.90, 0.91, 0.92]):
        results.append({
            "run_label": f"run_{seed}",
            "seed": seed,
            "strategy": "fedbuff",
            "configuration": {
                "compression": {"name": "topk_1pct"},
                "bandwidth_kbps": 512,
                "dropout_rate": 0.25,
            },
            "final_metrics": {
                "test_accuracy": 0.8,
                "test_macro_auroc": auroc,
                "test_macro_f1": 0.78,
            },
        })
    summary = summarize_results(results)
    recommendation = recommend_operating_point(summary)
    assert recommendation["seeds"] == 3
    assert recommendation["macro_auroc_std"] > 0


def test_supabase_rows_keep_round_and_client_contracts():
    rows = build_rows("test", [{
        "run_label": "run_42",
        "seed": 42,
        "strategy": "fedbuff",
        "configuration": {
            "compression": {"name": "none"},
            "connectivity": {"name": "moderate"},
            "bandwidth_kbps": 512,
        },
        "final_metrics": {},
        "rounds": [{"round_number": 1, "test_accuracy": 0.8}],
        "client_connectivity": {"uptime/client_00": 0.5},
    }])
    assert rows["runs"][0]["external_id"] == "run_42"
    assert rows["rounds"][0]["run_external_id"] == "run_42"
    assert rows["clients"][0]["client_index"] == 0
    assert rows["client_round_events"][0]["uptime"] == 0.5
