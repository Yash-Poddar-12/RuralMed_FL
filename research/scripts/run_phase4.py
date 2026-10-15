"""Phase 4 — Straggler-Tolerant Asynchronous Aggregation simulation entry point.

Run from the repo root with the venv activated:

    python research/scripts/run_phase4.py --config research/configs/phase4_async_bufferK_sweep.yml

CLI flags:
    --config       Path to a Phase 4 YAML config (default: phase4_async_bufferK_sweep.yml)
    --seed         Override seed in the config
    --rounds       Override number of FL rounds
    --strategy     Override strategy: "fedbuff" | "sync_dropout"
    --buffer-size  Override buffer_size (K) for FedBuff
    --dropout      Override straggler.dropout_prob  (e.g. 0.0, 0.1, 0.25, 0.4, 0.6)
    --no-tracking  Disable MLflow

Example sweep (dropout-rate sweep with FedBuff K=3):
    python research/scripts/run_phase4.py --config research/configs/phase4_dropout_rate_sweep.yml --dropout 0.00
    python research/scripts/run_phase4.py --config research/configs/phase4_dropout_rate_sweep.yml --dropout 0.10
    python research/scripts/run_phase4.py --config research/configs/phase4_dropout_rate_sweep.yml --dropout 0.25
    python research/scripts/run_phase4.py --config research/configs/phase4_dropout_rate_sweep.yml --dropout 0.40
    python research/scripts/run_phase4.py --config research/configs/phase4_dropout_rate_sweep.yml --dropout 0.60

Buffer-size sweep (K in {1,3,5}, fixed 25% dropout):
    python research/scripts/run_phase4.py --config research/configs/phase4_async_bufferK_sweep.yml --buffer-size 1
    python research/scripts/run_phase4.py --config research/configs/phase4_async_bufferK_sweep.yml --buffer-size 3
    python research/scripts/run_phase4.py --config research/configs/phase4_async_bufferK_sweep.yml --buffer-size 5

Sync-dropout baseline (for the same dropout rates as FedBuff sweep):
    python research/scripts/run_phase4.py --config research/configs/phase4_dropout_rate_sweep.yml --strategy sync_dropout --dropout 0.25

Phase 3 + Phase 4 compatibility check (async + compression):
    python research/scripts/run_phase4.py --config research/configs/phase4_async_bufferK_sweep.yml --rounds 5
    (Edit the config to set compression.enabled: true first.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["GRPC_VERBOSITY"] = "ERROR"

# ── Repo root on sys.path ─────────────────────────────────────────────────────
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
from research.src.fl_core.aggregation import FedBuffStrategy, SyncWithDropout
from research.src.fl_core.straggler_model import DropoutSampler, DelaySampler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("schema_version") == 1, "Unsupported config schema version"
    return cfg


def _resolve_paths(cfg4: dict, cfg1: dict) -> tuple[str, Path]:
    processed_root = cfg1["paths"]["processed"]
    partitions_root = Path(cfg1["paths"]["partitions"])
    if not partitions_root.exists():
        raise FileNotFoundError(
            f"Partitions directory not found: {partitions_root}\n"
            "Make sure RURALMED_DATA_ROOT is set and Phase 1 data prep is done."
        )
    return processed_root, partitions_root


def _print_header(title: str) -> None:
    print("=" * 70, flush=True)
    print(title, flush=True)
    print(f"  PyTorch  : {torch.__version__}", flush=True)
    print(f"  CUDA     : {torch.version.cuda}", flush=True)
    print(f"  Device   : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
    print(f"  Flower   : {fl.__version__}", flush=True)
    print("=" * 70, flush=True)


def _read_partition_sizes(partitions_root: Path, num_clients: int) -> list[int]:
    """Return number of training images per client (used for size-correlated dropout)."""
    sizes = []
    for cid in range(num_clients):
        train_csv = partitions_root / f"client_{cid:02d}" / "train.csv"
        if train_csv.exists():
            with open(train_csv) as f:
                n = sum(1 for _ in f) - 1  # minus header
            sizes.append(max(n, 0))
        else:
            sizes.append(0)
    return sizes


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 async aggregation simulation")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "research/configs/phase4_async_bufferK_sweep.yml"),
        help="Path to phase4 YAML config",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--strategy", choices=["fedbuff", "sync_dropout"], default=None)
    parser.add_argument("--buffer-size", type=int, default=None, dest="buffer_size")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override straggler.dropout_prob")
    parser.add_argument("--no-tracking", action="store_true")
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────────
    cfg4 = _load_cfg(args.config)
    cfg1 = load_phase1_config(REPO_ROOT / "research/configs/phase1.yml")

    # CLI overrides
    if args.seed is not None:
        cfg4["seed"] = args.seed
    if args.rounds is not None:
        cfg4["federated"]["rounds"] = args.rounds
    if args.strategy is not None:
        cfg4["async"]["strategy"] = args.strategy
    if args.buffer_size is not None:
        cfg4["async"]["buffer_size"] = args.buffer_size
    if args.dropout is not None:
        cfg4["straggler"]["dropout_prob"] = args.dropout

    seed: int = cfg4["seed"]
    NUM_ROUNDS: int = cfg4["federated"]["rounds"]
    NUM_CLIENTS: int = cfg4["data"]["clients"]
    FRAC_FIT: float = cfg4["federated"]["fraction_fit"]

    async_cfg = cfg4.get("async", {})
    strategy_name: str = str(async_cfg.get("strategy", "fedbuff")).lower()
    buffer_size: int = int(async_cfg.get("buffer_size", 3))
    staleness_alpha: float = float(async_cfg.get("staleness_alpha", 0.5))

    straggler_cfg = cfg4.get("straggler", {})
    dropout_prob: float = float(straggler_cfg.get("dropout_prob", 0.0))
    size_correlated: bool = bool(straggler_cfg.get("size_correlated", True))
    size_scale: float = float(straggler_cfg.get("size_scale", 0.3))
    delay_mu: float = float(straggler_cfg.get("delay_mu", 0.0))
    delay_sigma: float = float(straggler_cfg.get("delay_sigma", 0.5))
    straggler_seed: int = int(straggler_cfg.get("seed", seed))

    comp_cfg = cfg4.get("compression", {})
    is_compressed: bool = comp_cfg.get("enabled", False)

    torch.manual_seed(seed)

    # ── Title ──────────────────────────────────────────────────────────────────
    strategy_label = "FedBuff" if strategy_name == "fedbuff" else "Sync+Dropout"
    title = (
        f"Phase 4 — {strategy_label}  "
        f"[K={buffer_size}  dropout={dropout_prob:.0%}  α={staleness_alpha}]"
    )
    if strategy_name == "fedbuff" and is_compressed:
        title += "  [+Top-k compression]"

    _print_header(title)
    print(f"\nSeed: {seed}  |  Rounds: {NUM_ROUNDS}  |  Clients: {NUM_CLIENTS}", flush=True)
    print(f"Strategy : {strategy_label}", flush=True)
    print(f"Buffer K : {buffer_size}  (FedBuff only)", flush=True)
    print(f"Dropout  : {dropout_prob:.0%}  (base probability per round)", flush=True)
    print(f"Staleness α : {staleness_alpha}", flush=True)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    processed_root, partitions_root = _resolve_paths(cfg4, cfg1)
    test_index = partitions_root / "splits" / "test.csv"
    if not test_index.exists():
        raise FileNotFoundError(f"Test index not found: {test_index}")

    print(f"\nProcessed images : {processed_root}", flush=True)
    print(f"Partitions root  : {partitions_root}", flush=True)
    print(f"Test index       : {test_index}", flush=True)

    # ── Build global model ────────────────────────────────────────────────────
    model_cfg = cfg4["model"]
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

    # ── Straggler models ──────────────────────────────────────────────────────
    print(f"\nBuilding straggler model …", flush=True)
    if size_correlated:
        client_sizes = _read_partition_sizes(partitions_root, NUM_CLIENTS)
        print(f"  Client sizes (train images): {client_sizes}", flush=True)
    else:
        client_sizes = None

    dropout_sampler = DropoutSampler(
        dropout_prob=dropout_prob,
        client_sizes=client_sizes,
        size_scale=size_scale,
        seed=straggler_seed,
    )
    delay_sampler = DelaySampler(
        base_delay_mu=delay_mu,
        base_delay_sigma=delay_sigma,
        client_sizes=client_sizes,
        size_scale=0.4,
        seed=straggler_seed + 1,
    )

    # ── Communication tracker ─────────────────────────────────────────────────
    tracker = CommsTracker(bytes_per_model=param_bytes)

    # ── MLflow ────────────────────────────────────────────────────────────────
    run_tag = (
        f"{strategy_name}_K{buffer_size}_drop{int(dropout_prob*100)}pct_seed{seed}"
    )
    mlflow_run_id: str | None = None
    if not args.no_tracking:
        tracking_cfg = cfg4["tracking"]
        tracking_uri = (
            REPO_ROOT / tracking_cfg["uri"].replace("file:", "")
        ).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(tracking_cfg["experiment"])
        run = mlflow.start_run(
            run_name=f"p4_{run_tag}",
            tags={
                "phase": "4",
                "algorithm": strategy_label,
                "backbone": "DenseNet-121",
                "buffer_size": str(buffer_size),
                "dropout_prob": str(dropout_prob),
            },
        )
        mlflow_run_id = run.info.run_id
        mlflow.log_params({
            "seed": seed, "rounds": NUM_ROUNDS, "clients": NUM_CLIENTS,
            "strategy": strategy_name, "buffer_size": buffer_size,
            "staleness_alpha": staleness_alpha, "dropout_prob": dropout_prob,
            "size_correlated": size_correlated, "size_scale": size_scale,
            "local_epochs": cfg4["training"]["local_epochs"],
            "batch_size": cfg4["training"]["batch_size"],
            "learning_rate": cfg4["training"]["learning_rate"],
            "compression_enabled": is_compressed,
        })
        print(f"\nMLflow run ID: {mlflow_run_id}", flush=True)

    # ── Evaluate function ─────────────────────────────────────────────────────
    evaluate_fn = build_evaluate_fn(
        processed_root=processed_root,
        test_index=test_index,
        model_cfg=model_cfg,
        training_cfg=cfg4["training"],
        class_weights=cfg4["data"]["class_weights"],
        device=DEVICE,
        tracker=tracker,
        mlflow_run_id=mlflow_run_id,
    )

    # ── Strategy ──────────────────────────────────────────────────────────────
    def on_fit_config_fn(server_round: int) -> dict:
        return {"round": server_round}

    strategy_kwargs = dict(
        tracker=tracker,
        dropout_sampler=dropout_sampler,
        fraction_fit=FRAC_FIT,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, int(NUM_CLIENTS * FRAC_FIT)),
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=initial_parameters,
    )

    if strategy_name == "fedbuff":
        strategy = FedBuffStrategy(
            buffer_size=buffer_size,
            staleness_alpha=staleness_alpha,
            **strategy_kwargs,
        )
    else:
        strategy = SyncWithDropout(**strategy_kwargs)

    # ── client_fn ─────────────────────────────────────────────────────────────
    def client_fn(context: fl.common.Context) -> fl.client.Client:
        partition_id = int(context.node_config["partition-id"])
        client = XRayClient(
            partition_id=partition_id,
            processed_root=processed_root,
            partitions_root=str(partitions_root),
            cfg=cfg4,
            device=DEVICE,
        )
        return client.to_client()

    # ── GPU resources ─────────────────────────────────────────────────────────
    hw = cfg4["hardware"]
    client_resources = {
        "num_cpus": hw["cpus_per_client"],
        "num_gpus": hw["gpus_per_client"],
    }
    print(f"\nClient resources: {client_resources}", flush=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(
        f"\nStarting simulation — {NUM_ROUNDS} rounds × {NUM_CLIENTS} clients …\n",
        flush=True,
    )
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
        },
    )

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    comms = tracker.summary()
    phase_dir = REPO_ROOT / "research/artifacts/phase4"
    phase_dir.mkdir(parents=True, exist_ok=True)
    comms_log_path = phase_dir / f"comms_{run_tag}.json"
    tracker.save(comms_log_path)

    # Staleness + participation diagnostics
    diag: dict = {}
    if strategy_name == "fedbuff" and hasattr(strategy, "staleness_summary"):
        staleness_diag = strategy.staleness_summary()
        participation_diag = strategy.participation_summary()
        diag.update(staleness_diag)
        diag.update(participation_diag)
        diag_path = phase_dir / f"diag_{run_tag}.json"
        with open(diag_path, "w") as f:
            json.dump({**staleness_diag, **participation_diag}, f, indent=2)
        print(f"\nStaleness diagnostics saved to: {diag_path}", flush=True)
        print("  " + "  ".join(f"{k}: {v:.2f}" for k, v in staleness_diag.items()), flush=True)

    print("\n" + "=" * 70, flush=True)
    print("Simulation complete", flush=True)
    print(f"  Wall time        : {elapsed/3600:.2f} h  ({elapsed/60:.1f} min)", flush=True)
    print(f"  Total comms      : {comms.get('comms/total_gb', 0):.2f} GB", flush=True)
    print(f"  Comms log        : {comms_log_path}", flush=True)

    if history.metrics_centralized:
        last_round_metrics = dict(history.metrics_centralized)
        final = {k: v[-1][1] for k, v in last_round_metrics.items() if v}
        print(f"\n  Final round metrics (round {NUM_ROUNDS}):", flush=True)
        for k in ["test_accuracy", "test_macro_auroc", "test_macro_f1",
                  "test_covid_19_sensitivity", "test_loss"]:
            if k in final:
                print(f"    {k:40s}: {final[k]:.4f}", flush=True)

    if mlflow_run_id and not args.no_tracking:
        mlflow.log_metrics(comms, step=NUM_ROUNDS)
        mlflow.log_metric("wall_time_hours", elapsed / 3600, step=NUM_ROUNDS)
        mlflow.log_artifact(str(comms_log_path))
        if diag:
            mlflow.log_metrics(diag, step=NUM_ROUNDS)
        mlflow.end_run()
        print(f"\n  MLflow run      : {mlflow_run_id}", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
