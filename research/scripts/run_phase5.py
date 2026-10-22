"""Phase 5 — Intermittent Connectivity Engine simulation entry point.

Plugs the Markov-chain connectivity engine into Phase 4's FedBuff aggregation
strategy with zero changes to the aggregation logic itself.

Run from the repo root with the venv activated:

    # Connectivity profiles validation (3 bandwidth caps, 50 rounds each):
    python research/scripts/run_phase5.py --config research/configs/phase5_connectivity_profiles.yml --bw-cap none
    python research/scripts/run_phase5.py --config research/configs/phase5_connectivity_profiles.yml --bw-cap 500
    python research/scripts/run_phase5.py --config research/configs/phase5_connectivity_profiles.yml --bw-cap 200

    # Bandwidth sweep (6 cap points + 2 joint compression interaction):
    python research/scripts/run_phase5.py --config research/configs/phase5_bandwidth_sweep.yml --run-name bw_uncapped_topk1pct
    python research/scripts/run_phase5.py --config research/configs/phase5_bandwidth_sweep.yml --run-name bw_512kbps_topk1pct
    python research/scripts/run_phase5.py --config research/configs/phase5_bandwidth_sweep.yml --run-name bw_256kbps_topk1pct
    python research/scripts/run_phase5.py --config research/configs/phase5_bandwidth_sweep.yml --run-name bw_128kbps_topk1pct
    python research/scripts/run_phase5.py --config research/configs/phase5_bandwidth_sweep.yml --run-name bw_64kbps_topk1pct
    python research/scripts/run_phase5.py --config research/configs/phase5_bandwidth_sweep.yml --run-name bw_256kbps_topk10pct
    python research/scripts/run_phase5.py --config research/configs/phase5_bandwidth_sweep.yml --run-name bw_256kbps_uncompressed

CLI flags:
    --config          Path to Phase 5 YAML config
    --run-name        Select a specific run from a bandwidth_sweep config block
    --bw-cap          Override bandwidth cap in kbps (\"none\" for uncapped)
    --rounds          Override number of FL rounds
    --seed            Override RNG seed
    --no-tracking     Disable MLflow
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
import mlflow
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters

from research.src.data_prep.common import load_config as load_phase1_config
from research.src.fl_core.models.densenet import build_densenet121, get_parameters
from research.src.fl_core.comms_tracker import CommsTracker, model_bytes
from research.src.fl_core.client import XRayClient, DEVICE
from research.src.fl_core.server import build_evaluate_fn
from research.src.fl_core.aggregation import FedBuffStrategy
from research.src.fl_core.aggregation.sync_with_dropout import SyncWithDropout
from research.src.simulation import (
    MarkovConnectivityEngine,
    BandwidthDelayAdapter,
    build_client_profiles,
    profile_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_phase1(cfg1_path: Path) -> dict:
    return load_phase1_config(cfg1_path)


def _read_partition_sizes(partitions_root: Path, num_clients: int) -> list[int]:
    sizes = []
    for cid in range(num_clients):
        csv = partitions_root / f"client_{cid:02d}" / "train.csv"
        if csv.exists():
            with open(csv) as f:
                n = sum(1 for _ in f) - 1
            sizes.append(max(n, 0))
        else:
            sizes.append(0)
    return sizes


def _print_header(title: str) -> None:
    print("=" * 70, flush=True)
    print(title, flush=True)
    print(f"  PyTorch  : {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        print(f"  CUDA     : {torch.version.cuda}", flush=True)
        print(f"  Device   : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
    print(f"  Flower   : {fl.__version__}", flush=True)
    print("=" * 70, flush=True)


def _select_run_config(cfg: dict, run_name: str | None) -> tuple[float | None, int, str]:
    """Extract bw_cap_kbps, payload_bytes and a label for this run."""
    if "bandwidth_sweep" in cfg:
        sweep = cfg["bandwidth_sweep"]
        if run_name is None:
            # Default to first entry
            entry = sweep[0]
        else:
            matches = [e for e in sweep if e["run_name"] == run_name]
            if not matches:
                raise ValueError(f"run_name '{run_name}' not found in bandwidth_sweep")
            entry = matches[0]
        return entry.get("bw_cap_kbps"), entry["payload_compressed_bytes"], entry["run_name"]
    else:
        # Connectivity-profiles config
        label = run_name or cfg.get("experiment_name", "phase5_run")
        return None, cfg.get("payload_compressed_bytes", 11534336), label


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 connectivity-driven FL simulation")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "research/configs/phase5_bandwidth_sweep.yml"),
    )
    parser.add_argument("--phase1-config", default=str(REPO_ROOT / "research/configs/phase1.yml"),
                        help="Phase 1 data/config path; use phase1-dev.yml for the local sample")
    parser.add_argument("--run-name", default=None, dest="run_name",
                        help="Select a named entry from bandwidth_sweep block")
    parser.add_argument("--bw-cap", default=None, dest="bw_cap",
                        help="Override bandwidth cap in kbps, or 'none' for uncapped")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategy", choices=["fedbuff", "sync_dropout"], default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--compression-method", choices=["none", "topk", "qsgd"], default=None)
    parser.add_argument("--topk-ratio", type=float, default=None)
    parser.add_argument("--quant-bits", type=int, default=None)
    parser.add_argument("--round-budget", type=float, default=None)
    parser.add_argument("--payload-bytes", type=int, default=None)
    parser.add_argument("--result-json", default=None,
                        help="Write a Phase 6-compatible structured result JSON")
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────────
    cfg = _load_cfg(args.config)
    cfg1 = _resolve_phase1(Path(args.phase1_config))

    # Resolve run-specific bw_cap + payload
    bw_cap_kbps, payload_bytes, run_label = _select_run_config(cfg, args.run_name)
    if args.payload_bytes is not None:
        payload_bytes = args.payload_bytes

    # CLI overrides
    if args.bw_cap is not None:
        bw_cap_kbps = None if args.bw_cap.lower() == "none" else float(args.bw_cap)
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed: int = cfg.get("seed", 42)
    NUM_ROUNDS: int = args.rounds if args.rounds is not None else cfg.get("num_rounds", 50)
    NUM_CLIENTS: int = cfg.get("num_clients", 10)
    BUFFER_SIZE: int = cfg.get("buffer_size", 3)
    STALENESS_ALPHA: float = cfg.get("staleness_alpha", 0.5)
    ROUND_BUDGET_S: float = cfg.get("connectivity", {}).get("round_budget_seconds", 300.0)
    if args.buffer_size is not None:
        BUFFER_SIZE = args.buffer_size
    if args.round_budget is not None:
        ROUND_BUDGET_S = args.round_budget
    CONN_SEED: int = cfg.get("connectivity", {}).get("seed", seed)

    torch.manual_seed(seed)

    title = (
        f"Phase 5 — Intermittent Connectivity Engine  "
        f"[run={run_label}  bw_cap={bw_cap_kbps} kbps  budget={ROUND_BUDGET_S}s]"
    )
    _print_header(title)
    print(f"\nSeed: {seed}  |  Rounds: {NUM_ROUNDS}  |  Clients: {NUM_CLIENTS}", flush=True)
    print(f"Payload size     : {payload_bytes/1e6:.2f} MB", flush=True)
    print(f"BW cap           : {bw_cap_kbps} kbps  (None = use raw state ranges)", flush=True)
    print(f"Round budget     : {ROUND_BUDGET_S} s", flush=True)

    # ── Paths ─────────────────────────────────────────────────────────────────
    processed_root = cfg1["paths"]["processed"]
    partitions_root = Path(cfg1["paths"]["partitions"])
    if not partitions_root.exists():
        raise FileNotFoundError(f"Partitions directory not found: {partitions_root}")
    test_index = partitions_root / "splits" / "test.csv"
    if not test_index.exists():
        raise FileNotFoundError(f"Test index not found: {test_index}")

    # ── Partition sizes → profile assignment ──────────────────────────────────
    client_sizes = _read_partition_sizes(partitions_root, NUM_CLIENTS)
    print(f"\nClient sizes (train images): {client_sizes}", flush=True)
    profiles = build_client_profiles(client_sizes)
    print("\nConnectivity profile assignment:", flush=True)
    print(profile_summary(profiles), flush=True)

    # ── Markov engine + bandwidth adapter ─────────────────────────────────────
    engine = MarkovConnectivityEngine(profiles=profiles, seed=CONN_SEED, bw_cap_kbps=bw_cap_kbps)
    bw_adapter = BandwidthDelayAdapter(
        engine=engine,
        payload_bytes=payload_bytes,
        round_budget_s=ROUND_BUDGET_S,
        seed=seed + 10,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    # Load model config from a Phase 4 config (reuse DenseNet-121 settings)
    phase4_cfg_path = REPO_ROOT / "research/configs/phase4_async_bufferK_sweep.yml"
    cfg4 = _load_cfg(phase4_cfg_path)
    if args.compression_method is not None:
        compression = cfg4.setdefault("compression", {})
        compression["enabled"] = args.compression_method != "none"
        compression["method"] = args.compression_method
    if args.topk_ratio is not None:
        cfg4.setdefault("compression", {})["topk_ratio"] = args.topk_ratio
    if args.quant_bits is not None:
        cfg4.setdefault("compression", {})["quant_bits"] = args.quant_bits
    if args.strategy == "sync_dropout":
        cfg4.setdefault("async", {})["strategy"] = "sync_dropout"
    model_cfg = cfg4["model"]
    training_cfg = cfg4["training"]
    class_weights = cfg4["data"]["class_weights"]
    hw = cfg4["hardware"]

    print(f"\nBuilding DenseNet-121 (pretrained={model_cfg['pretrained']}) …", flush=True)
    global_model = build_densenet121(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
    )
    initial_params = get_parameters(global_model)
    initial_parameters = ndarrays_to_parameters(initial_params)
    param_bytes = model_bytes(global_model)
    print(
        f"  Parameters : {sum(p.numel() for p in global_model.parameters()):,}"
        f"  ({param_bytes/1e6:.1f} MB fp32)",
        flush=True,
    )
    del global_model

    # ── Communication tracker ─────────────────────────────────────────────────
    tracker = CommsTracker(bytes_per_model=param_bytes)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_run_id: str | None = None
    if not args.no_tracking:
        tracking_uri = (REPO_ROOT / "research/artifacts/mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg.get("experiment_name", "phase5-connectivity"))
        bw_str = str(int(bw_cap_kbps)) if bw_cap_kbps else "uncapped"
        run = mlflow.start_run(
            run_name=f"p5_{run_label}_bw{bw_str}",
            tags={
                "phase": "5",
                "algorithm": "FedBuff",
                "backbone": "DenseNet-121",
                "connectivity_engine": "markov",
            },
        )
        mlflow_run_id = run.info.run_id
        mlflow.log_params({
            "seed": seed, "rounds": NUM_ROUNDS, "clients": NUM_CLIENTS,
            "strategy": "fedbuff", "buffer_size": BUFFER_SIZE,
            "staleness_alpha": STALENESS_ALPHA,
            "bw_cap_kbps": bw_cap_kbps if bw_cap_kbps else -1,
            "payload_bytes": payload_bytes,
            "round_budget_s": ROUND_BUDGET_S,
            "run_name": run_label,
        })
        print(f"\nMLflow run ID: {mlflow_run_id}", flush=True)

    # ── Evaluate fn ───────────────────────────────────────────────────────────
    evaluate_fn = build_evaluate_fn(
        processed_root=processed_root,
        test_index=test_index,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        class_weights=class_weights,
        device=DEVICE,
        tracker=tracker,
        mlflow_run_id=mlflow_run_id,
    )

    # ── FedBuff strategy — Phase 4 aggregation, Phase 5 straggler interface ──
    # The BandwidthDelayAdapter replaces DropoutSampler from Phase 4.
    # FedBuffStrategy is not modified — only the sampler injected changes.
    def on_fit_config_fn(server_round: int) -> dict:
        return {"round": server_round}

    strategy_kwargs = dict(
        tracker=tracker,
        dropout_sampler=bw_adapter,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, NUM_CLIENTS),
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=initial_parameters,
    )
    if args.strategy == "sync_dropout":
        strategy = SyncWithDropout(**strategy_kwargs)
    else:
        strategy = FedBuffStrategy(
            buffer_size=BUFFER_SIZE,
            staleness_alpha=STALENESS_ALPHA,
            **strategy_kwargs,
        )

    # ── client_fn ─────────────────────────────────────────────────────────────
    def client_fn(context: fl.common.Context) -> fl.client.Client:
        partition_id = int(context.node_config["partition-id"])
        return XRayClient(
            partition_id=partition_id,
            processed_root=processed_root,
            partitions_root=str(partitions_root),
            cfg=cfg4,
            device=DEVICE,
        ).to_client()

    client_resources = {
        "num_cpus": hw["cpus_per_client"],
        "num_gpus": hw["gpus_per_client"],
    }
    print(f"\nClient resources: {client_resources}", flush=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\nStarting simulation — {NUM_ROUNDS} rounds × {NUM_CLIENTS} clients …\n", flush=True)
    t0 = time.time()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={
            "include_dashboard": False,
            "num_cpus": hw["cpus_per_client"] * NUM_CLIENTS,
            "object_store_memory": 256 * 1024 * 1024,
            "ignore_reinit_error": True,
        },
    )

    elapsed = time.time() - t0

    # ── Connectivity diagnostics ──────────────────────────────────────────────
    uptime_metrics = bw_adapter.uptime_summary()
    avg_bw_metrics = bw_adapter.avg_bandwidth_summary()
    engine_metrics = engine.summary()

    print("\n── Connectivity diagnostics ──", flush=True)
    for cid in range(NUM_CLIENTS):
        pname = profiles[cid].name
        uptime = uptime_metrics.get(f"uptime/client_{cid:02d}", -1)
        avg_bw = avg_bw_metrics.get(f"avg_bw_kbps/client_{cid:02d}", -1)
        print(
            f"  client_{cid:02d} [{pname:>12s}]  uptime={uptime:.1%}  avg_bw={avg_bw:.0f} kbps",
            flush=True,
        )

    # ── Save diagnostics ──────────────────────────────────────────────────────
    phase_dir = REPO_ROOT / "research/artifacts/phase5"
    phase_dir.mkdir(parents=True, exist_ok=True)
    bw_str_file = str(int(bw_cap_kbps)) if bw_cap_kbps else "uncapped"
    diag_path = phase_dir / f"connectivity_{run_label}_bw{bw_str_file}.json"
    full_diag = {
        "run_label": run_label, "bw_cap_kbps": bw_cap_kbps,
        "payload_bytes": payload_bytes, "round_budget_s": ROUND_BUDGET_S,
        **uptime_metrics, **avg_bw_metrics, **engine_metrics,
    }
    with open(diag_path, "w") as f:
        json.dump(full_diag, f, indent=2)
    print(f"\nConnectivity diagnostics saved: {diag_path}", flush=True)

    # ── Comms summary ─────────────────────────────────────────────────────────
    comms = tracker.summary()
    comms_path = phase_dir / f"comms_{run_label}_bw{bw_str_file}.json"
    tracker.save(comms_path)

    # ── Final metrics ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("Simulation complete", flush=True)
    print(f"  Wall time : {elapsed/3600:.2f} h  ({elapsed/60:.1f} min)", flush=True)
    print(f"  Comms     : {comms.get('comms/total_gb', 0):.2f} GB", flush=True)

    if history.metrics_centralized:
        final = {k: v[-1][1] for k, v in history.metrics_centralized.items() if v}
        for k in ["test_accuracy", "test_macro_auroc", "test_macro_f1",
                  "test_covid_19_sensitivity", "test_loss"]:
            if k in final:
                print(f"  {k:40s}: {final[k]:.4f}", flush=True)

    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        final = {
            k: float(v[-1][1])
            for k, v in history.metrics_centralized.items()
            if v
        }
        round_metrics = []
        for round_number in sorted({r for values in history.metrics_centralized.values() for r, _ in values}):
            row = {"round_number": int(round_number)}
            for key, values in history.metrics_centralized.items():
                row[key] = next((float(value) for r, value in values if r == round_number), None)
            round_metrics.append(row)
        result_path.write_text(json.dumps({
            "run_label": run_label,
            "seed": seed,
            "strategy": args.strategy or "fedbuff",
            "buffer_size": BUFFER_SIZE,
            "bandwidth_cap_kbps": bw_cap_kbps,
            "payload_bytes": payload_bytes,
            "round_budget_seconds": ROUND_BUDGET_S,
            "final_metrics": final,
            "rounds": round_metrics,
            "communication_rounds": tracker.rounds,
            "communication": comms,
            "client_connectivity": full_diag,
            "client_events": [
                {
                    "client_index": cid,
                    "round_number": round_number,
                    "status": "dropped" if dropped else "participated",
                    "effective_bandwidth": bw_adapter.bandwidth_log[cid][round_number],
                }
                for cid, drops in bw_adapter.dropout_log.items()
                for round_number, dropped in enumerate(drops, start=1)
            ],
            "elapsed_seconds": elapsed,
        }, indent=2), encoding="utf-8")
        print(f"Structured result saved: {result_path}", flush=True)

    if mlflow_run_id and not args.no_tracking:
        mlflow.log_metrics(comms, step=NUM_ROUNDS)
        mlflow.log_metrics(uptime_metrics, step=NUM_ROUNDS)
        mlflow.log_metrics(avg_bw_metrics, step=NUM_ROUNDS)
        mlflow.log_metric("wall_time_hours", elapsed / 3600, step=NUM_ROUNDS)
        mlflow.log_artifact(str(diag_path))
        mlflow.log_artifact(str(comms_path))
        mlflow.end_run()
        print(f"  MLflow run: {mlflow_run_id}", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
