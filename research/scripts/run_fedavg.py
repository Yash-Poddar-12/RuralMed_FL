"""Phase 2 — FedAvg simulation entry point.

Run from the repo root with the venv activated and RURALMED_DATA_ROOT set:

    python research/scripts/run_fedavg.py

Optional flags:
    --config   path to YAML config (default: research/configs/phase2_fedavg_baseline.yml)
    --seed     override the seed in the config (for multi-seed sweeps)
    --rounds   override number of FL rounds
    --no-tracking  disable MLflow (useful for quick debugging)

Multi-seed example (run 3 seeds back-to-back):
    python research/scripts/run_fedavg.py --seed 42
    python research/scripts/run_fedavg.py --seed 123
    python research/scripts/run_fedavg.py --seed 999
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Suppress gRPC warnings from Ray/Flower (e.g. Error parsing 'content-type')
os.environ["GRPC_VERBOSITY"] = "ERROR"

# ── Ensure repo root is on sys.path regardless of CWD ────────────────────────
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
from research.src.fl_core.server import FedAvgWithTracking, build_evaluate_fn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_phase2_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("schema_version") == 1, "Unsupported phase2 config schema"
    return cfg


def _resolve_paths(cfg2: dict, cfg1: dict) -> tuple[str, Path]:
    """Return (processed_root, partitions_root) from the combined configs."""
    processed_root = cfg1["paths"]["processed"]
    partitions_root = Path(cfg1["paths"]["partitions"])
    if not partitions_root.exists():
        raise FileNotFoundError(
            f"Partitions directory not found: {partitions_root}\n"
            f"Make sure RURALMED_DATA_ROOT is set and the full data run is complete."
        )
    return processed_root, partitions_root


def _print_system_info(title: str = "Phase 2 — FedAvg Baseline") -> None:
    print("=" * 70, flush=True)
    print(title, flush=True)
    print(f"  PyTorch  : {torch.__version__}", flush=True)
    print(f"  CUDA     : {torch.version.cuda}", flush=True)
    print(f"  Device   : {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
    print(f"  Flower   : {fl.__version__}", flush=True)
    print("=" * 70, flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 FedAvg simulation")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "research/configs/phase2_fedavg_baseline.yml"),
        help="Path to phase2 YAML config",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--rounds", type=int, default=None, help="Number of FL rounds override")
    parser.add_argument("--no-tracking", action="store_true", help="Disable MLflow")
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────────
    cfg2 = _load_phase2_config(args.config)
    cfg1 = load_phase1_config(REPO_ROOT / "research/configs/phase1.yml")

    if args.seed is not None:
        cfg2["seed"] = args.seed
    if args.rounds is not None:
        cfg2["federated"]["rounds"] = args.rounds

    seed: int = cfg2["seed"]
    NUM_ROUNDS: int = cfg2["federated"]["rounds"]
    NUM_CLIENTS: int = cfg2["data"]["clients"]
    FRAC_FIT: float = cfg2["federated"]["fraction_fit"]

    torch.manual_seed(seed)

    # ── Check compression mode ────────────────────────────────────────────────
    comp_cfg = cfg2.get("compression", {})
    is_compressed = comp_cfg.get("enabled", False)
    if is_compressed:
        comp_method = str(comp_cfg.get("method", "topk")).lower()
        if comp_method == "topk":
            ratio = float(comp_cfg.get("topk_ratio", 0.05))
            comp_title = f"Phase 3 — Compressed FedAvg (Top-{ratio*100:.1f}%)"
            run_tag = f"topk_{int(ratio*100)}pct_seed{seed}"
        else:
            bits = int(comp_cfg.get("quant_bits", 8))
            comp_title = f"Phase 3 — Compressed FedAvg (QSGD-{bits}bit)"
            run_tag = f"qsgd_{bits}bit_seed{seed}"
        phase_dir = "phase3"
        tags = {"phase": "3", "algorithm": f"FedAvg+{comp_method}", "backbone": "DenseNet-121"}

        # Clear residual files for fresh run
        res_dir = REPO_ROOT / "research/artifacts/phase3/residuals"
        if res_dir.exists():
            for p in res_dir.glob("*.pt"):
                try:
                    p.unlink()
                except Exception:
                    pass
    else:
        comp_title = "Phase 2 — FedAvg Baseline"
        run_tag = f"seed{seed}"
        phase_dir = "phase2"
        tags = {"phase": "2", "algorithm": "FedAvg", "backbone": "DenseNet-121"}

    _print_system_info(title=comp_title)
    print(f"\nSeed: {seed}  |  Rounds: {NUM_ROUNDS}  |  Clients: {NUM_CLIENTS}", flush=True)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    processed_root, partitions_root = _resolve_paths(cfg2, cfg1)
    test_index = partitions_root / "splits" / "test.csv"
    if not test_index.exists():
        raise FileNotFoundError(f"Centralized test index not found: {test_index}")

    print(f"\nProcessed images : {processed_root}", flush=True)
    print(f"Partitions root  : {partitions_root}", flush=True)
    print(f"Test index       : {test_index}  ({sum(1 for _ in open(test_index))-1} images)", flush=True)

    # ── Build initial global model ────────────────────────────────────────────
    model_cfg = cfg2["model"]
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
        f"  ({param_bytes/1e6:.1f} MB fp32  |  {param_bytes/2/1e6:.1f} MB fp16)",
        flush=True,
    )
    del global_model  # free memory; clients build their own copies

    # ── Communication tracker ──────────────────────────────────────────────────
    tracker = CommsTracker(bytes_per_model=param_bytes)
    print(
        f"\nUncompressed comms per round: "
        f"{2 * NUM_CLIENTS * param_bytes / 1e9:.2f} GB  "
        f"(upload + download × {NUM_CLIENTS} clients)",
        flush=True,
    )
    print(
        f"Expected total over {NUM_ROUNDS} rounds: "
        f"{2 * NUM_CLIENTS * param_bytes * NUM_ROUNDS / 1e9:.1f} GB  "
        f"(uncompressed baseline)",
        flush=True,
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_run_id: str | None = None
    if not args.no_tracking:
        tracking_cfg = cfg2["tracking"]
        tracking_uri = (REPO_ROOT / tracking_cfg["uri"].replace("file:", "")).resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(tracking_cfg["experiment"])
        run = mlflow.start_run(
            run_name=f"fedavg_{run_tag}",
            tags=tags,
        )
        mlflow_run_id = run.info.run_id
        log_params_dict = {
            "seed": seed,
            "rounds": NUM_ROUNDS,
            "clients": NUM_CLIENTS,
            "local_epochs": cfg2["training"]["local_epochs"],
            "batch_size": cfg2["training"]["batch_size"],
            "learning_rate": cfg2["training"]["learning_rate"],
            "weight_decay": cfg2["training"]["weight_decay"],
            "fraction_fit": FRAC_FIT,
            "use_amp": cfg2["training"]["use_amp"],
            "architecture": model_cfg["architecture"],
            "pretrained": model_cfg["pretrained"],
            "dropout": model_cfg["dropout"],
            "class_weights": str(cfg2["data"]["class_weights"]),
            "param_bytes": param_bytes,
        }
        if is_compressed:
            log_params_dict.update({f"comp_{k}": v for k, v in comp_cfg.items()})
        mlflow.log_params(log_params_dict)
        print(f"\nMLflow run ID: {mlflow_run_id}", flush=True)

    # ── Evaluate function (centralized test set) ──────────────────────────────
    evaluate_fn = build_evaluate_fn(
        processed_root=processed_root,
        test_index=test_index,
        model_cfg=model_cfg,
        training_cfg=cfg2["training"],
        class_weights=cfg2["data"]["class_weights"],
        device=DEVICE,
        tracker=tracker,
        mlflow_run_id=mlflow_run_id,
    )

    # ── Strategy ──────────────────────────────────────────────────────────────
    strategy = FedAvgWithTracking(
        tracker=tracker,
        fraction_fit=FRAC_FIT,
        fraction_evaluate=0.0,          # no distributed evaluation rounds
        min_fit_clients=max(1, int(NUM_CLIENTS * FRAC_FIT)),
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        # FedAvg weighted aggregation: Flower uses the num_examples returned
        # by each client's fit() call — larger clinics get more weight.
        # This is correct for the sample-count imbalance from Phase 1.
    )

    # ── client_fn for Flower simulation ──────────────────────────────────────
    def client_fn(context: fl.common.Context) -> fl.client.Client:
        """Factory called by the Flower simulation backend per virtual client."""
        partition_id = int(context.node_config["partition-id"])
        client = XRayClient(
            partition_id=partition_id,
            processed_root=processed_root,
            partitions_root=str(partitions_root),
            cfg=cfg2,
            device=DEVICE,
        )
        return client.to_client()

    # ── GPU resource allocation ───────────────────────────────────────────────
    hw = cfg2["hardware"]
    client_resources = {
        "num_cpus": hw["cpus_per_client"],
        "num_gpus": hw["gpus_per_client"],
    }
    print(f"\nClient resources: {client_resources}", flush=True)
    print(
        f"Max simultaneous clients on GPU: "
        f"{int(1.0 / hw['gpus_per_client'])} "
        f"(total VRAM budget: "
        f"{int(1.0 / hw['gpus_per_client']) * param_bytes / 1e6:.0f} MB per forward pass, "
        f"~{int(1.0 / hw['gpus_per_client']) * 700:.0f} MB with activations+gradients)",
        flush=True,
    )

    # ── Run simulation ────────────────────────────────────────────────────────
    print(f"\nStarting simulation — {NUM_ROUNDS} rounds × {NUM_CLIENTS} clients …\n", flush=True)
    t0 = time.time()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"include_dashboard": False, "num_cpus": hw["cpus_per_client"] * NUM_CLIENTS},
    )

    elapsed = time.time() - t0

    # ── Final summary ─────────────────────────────────────────────────────────
    comms = tracker.summary()
    comms_log_path = REPO_ROOT / f"research/artifacts/{phase_dir}/comms_{run_tag}.json"
    tracker.save(comms_log_path)

    ratio_str = f"({comms.get('comms/compression_ratio', 1.0):.1f}x compression)" if is_compressed else "(uncompressed baseline)"

    print("\n" + "=" * 70, flush=True)
    print("Simulation complete", flush=True)
    print(f"  Wall time        : {elapsed/3600:.2f} h  ({elapsed/60:.1f} min)", flush=True)
    print(f"  Total comms      : {comms.get('comms/total_gb', 0):.2f} GB  {ratio_str}", flush=True)
    print(f"  Comms log        : {comms_log_path}", flush=True)

    if history.metrics_centralized:
        last_round_metrics = dict(history.metrics_centralized)
        # Print the last round's key metrics
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
        mlflow.end_run()
        print(f"\n  MLflow run      : {mlflow_run_id}", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
