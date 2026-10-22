"""Phase 6 experiment orchestration and result normalization.

The runner expands one YAML sweep into reproducible subprocesses.  Training
remains in the Phase 5 entry point; this module only supplies overrides and
collects each seed's raw result before writing an indexed manifest.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCRIPT = REPO_ROOT / "research/scripts/run_phase5.py"


def _payload_bytes(compression: dict[str, Any], fallback: int) -> int:
    if "payload_bytes" in compression:
        return int(compression["payload_bytes"])
    method = compression.get("method", "none")
    if method == "topk":
        return int(compression.get("payload_bytes_by_ratio", {}).get(
            str(compression.get("topk_ratio")), fallback
        ))
    if method == "qsgd":
        return int(compression.get("payload_bytes_by_bits", {}).get(
            str(compression.get("quant_bits")), fallback
        ))
    return fallback


def expand_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand compression, connectivity, strategy, and seed dimensions."""
    compressions = config["sweep"]["compression"]
    connectivity = config["sweep"]["connectivity"]
    strategies = config["sweep"].get("strategies", ["fedbuff"])
    seeds = config["sweep"].get("seeds", [config.get("seed", 42)])
    runs: list[dict[str, Any]] = []
    for compression, condition, strategy, seed in itertools.product(
        compressions, connectivity, strategies, seeds
    ):
        label = (
            f"{strategy}_{compression['name']}_{condition['name']}_seed{seed}"
        )
        runs.append({
            "run_label": label,
            "seed": int(seed),
            "strategy": strategy,
            "compression": compression,
            "connectivity": condition,
            "dropout_rate": float(condition.get("dropout_rate", 0.0)),
            "bandwidth_kbps": condition.get("bandwidth_kbps"),
            "buffer_size": int(config.get("buffer_size", 3)),
        })
    for extra in config["sweep"].get("additional_runs", []):
        for seed in extra.get("seeds", seeds):
            runs.append({
                "run_label": (
                    f"{extra['strategy']}_{extra['compression']['name']}_"
                    f"{extra['connectivity']['name']}_seed{seed}"
                ),
                "seed": int(seed),
                "strategy": extra["strategy"],
                "compression": extra["compression"],
                "connectivity": extra["connectivity"],
                "dropout_rate": float(extra["connectivity"].get("dropout_rate", 0.0)),
                "bandwidth_kbps": extra["connectivity"].get("bandwidth_kbps"),
                "buffer_size": int(extra.get("buffer_size", config.get("buffer_size", 3))),
            })
    return runs


def _command(spec: dict[str, Any], output: Path, config: dict[str, Any]) -> list[str]:
    compression = spec["compression"]
    connectivity = spec["connectivity"]
    payload = _payload_bytes(
        compression,
        int(config.get("default_payload_bytes", 86_000_000)),
    )
    command = [
        sys.executable,
        str(config.get("training_script", DEFAULT_SCRIPT)),
        "--config",
        str(config.get("phase5_config", REPO_ROOT / "research/configs/phase5_bandwidth_sweep.yml")),
        "--phase1-config",
        str(config.get("phase1_config", REPO_ROOT / "research/configs/phase1.yml")),
        "--run-name",
        connectivity.get("phase5_run_name", connectivity["name"]),
        "--seed",
        str(spec["seed"]),
        "--strategy",
        spec["strategy"],
        "--buffer-size",
        str(spec["buffer_size"]),
        "--rounds",
        str(config.get("num_rounds", 50)),
        "--compression-method",
        compression.get("method", "none"),
        "--payload-bytes",
        str(payload),
        "--result-json",
        str(output),
    ]
    if config.get("no_tracking", False):
        command.append("--no-tracking")
    if compression.get("method") == "topk":
        command.extend(["--topk-ratio", str(compression["topk_ratio"])])
    if compression.get("method") == "qsgd":
        command.extend(["--quant-bits", str(compression["quant_bits"])])
    if connectivity.get("bandwidth_kbps") is not None:
        command.extend(["--bw-cap", str(connectivity["bandwidth_kbps"])])
    else:
        command.extend(["--bw-cap", "none"])
    if connectivity.get("round_budget_seconds") is not None:
        command.extend(["--round-budget", str(connectivity["round_budget_seconds"])])
    return command


def run_sweep(
    config_path: Path,
    output_dir: Path,
    *,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    """Run the configured sweep and return its manifest."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs = expand_sweep(config)
    if max_runs is not None:
        specs = specs[:max_runs]
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for spec in specs:
        result_path = output_dir / f"{spec['run_label']}.json"
        command = _command(spec, result_path, config)
        record = {**spec, "result_path": str(result_path), "command": command}
        if dry_run:
            record["status"] = "planned"
        else:
            log_path = output_dir / f"{spec['run_label']}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record["status"] = "completed" if completed.returncode == 0 else "failed"
            record["returncode"] = completed.returncode
            record["log_path"] = str(log_path)
            if completed.returncode == 0 and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["configuration"] = spec
                result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            elif completed.returncode != 0:
                raise RuntimeError(
                    f"Phase 6 run failed ({spec['run_label']}); see {log_path}"
                )
        records.append(record)
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "config_path": str(config_path),
        "planned_runs": len(specs),
        "runs": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 6 experiment sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/phase6/raw")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    manifest = run_sweep(
        args.config,
        args.output_dir,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    print(json.dumps({
        "experiment_name": manifest["experiment_name"],
        "planned_runs": manifest["planned_runs"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
