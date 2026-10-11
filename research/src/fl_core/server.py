"""FedAvg server strategy with centralized evaluation and comms tracking.

Design:
    - Uses Flower's built-in FedAvg strategy with sample-count-weighted
      aggregation (critical given the 18x client volume imbalance from Phase 1).
    - ``evaluate_fn`` is called after every aggregation step on the held-out
      centralized test set — the ground truth for the paper's convergence curves.
    - Communication volume is logged every round via CommsTracker.
    - All round metrics are sent to MLflow so they can be compared across seeds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

import mlflow
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from research.src.data_prep.dataset import ImagingDataset
from research.src.fl_core.models.densenet import (
    build_densenet121,
    get_parameters,
    set_parameters,
)
from research.src.fl_core.comms_tracker import CommsTracker, model_bytes
from research.src.eval.metrics import compute_metrics, format_report


# ── Centralized test-set evaluation hook ─────────────────────────────────────

def build_evaluate_fn(
    processed_root: str,
    test_index: Path,
    model_cfg: dict,
    training_cfg: dict,
    class_weights: list,
    device: torch.device,
    tracker: CommsTracker,
    mlflow_run_id: Optional[str] = None,
):
    """Return a closure that evaluates the global model on the centralized test set.

    This function is passed to FedAvg as ``evaluate_fn``.  Flower calls it after
    every aggregation step with the freshly aggregated parameters.

    Returns:
        evaluate_fn(server_round, parameters, config) → (loss, metrics_dict)
    """
    # Build a test DataLoader once (reused every round — no re-construction cost)
    test_dataset = ImagingDataset(processed_root, test_index)
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Evaluation model lives on the server side (same process, same GPU)
    eval_model = build_densenet121(
        num_classes=model_cfg["num_classes"],
        pretrained=False,
        dropout=0.0,          # no dropout during inference
    ).to(device)
    eval_model.eval()

    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    use_amp: bool = training_cfg["use_amp"]

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # ── Load parameters ────────────────────────────────────────────────
        set_parameters(eval_model, parameters)

        # ── Evaluate ───────────────────────────────────────────────────────
        all_probs: list[np.ndarray] = []
        all_labels: list[int] = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels_dev = labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = eval_model(images)
                    loss = criterion(logits, labels_dev)

                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(labels.numpy().tolist())
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.array(all_labels)
        avg_loss = total_loss / max(total_samples, 1)

        metrics = compute_metrics(y_true, y_prob, prefix="test_")
        metrics["test_loss"] = float(avg_loss)
        metrics["round"] = float(server_round)

        # ── Communication tracking ──────────────────────────────────────────
        # num_clients tracked by the strategy; here we use the tracker summary
        comms_summary = tracker.summary()
        metrics.update(comms_summary)

        # ── Print ──────────────────────────────────────────────────────────
        print(
            f"\n[round {server_round:3d}] "
            f"loss={avg_loss:.4f}  "
            f"acc={metrics['test_accuracy']:.4f}  "
            f"auroc={metrics['test_macro_auroc']:.4f}  "
            f"f1={metrics['test_macro_f1']:.4f}  "
            f"covid_sens={metrics['test_covid_19_sensitivity']:.4f}  "
            f"comms_total={comms_summary.get('comms/total_gb', 0.0):.2f} GB",
            flush=True,
        )
        print(format_report(y_true, y_prob), flush=True)

        # ── Log to MLflow ───────────────────────────────────────────────────
        try:
            with mlflow.start_run(run_id=mlflow_run_id, nested=True):
                mlflow.log_metrics(metrics, step=server_round)
        except Exception:
            pass  # tracking failure must never crash training

        return float(avg_loss), metrics

    return evaluate_fn


# ── Custom FedAvg with comms tracking ────────────────────────────────────────

class FedAvgWithTracking(FedAvg):
    """FedAvg + per-round communication-volume logging.

    Inherits all aggregation logic from ``flwr.server.strategy.FedAvg``.
    The only addition is calling ``tracker.log_round`` inside
    ``aggregate_fit`` so we capture comms cost without duplicating the
    aggregation implementation.
    """

    def __init__(self, tracker: CommsTracker, **kwargs):
        super().__init__(**kwargs)
        self.tracker = tracker

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Record comms BEFORE aggregation (results contains participating clients)
        if results:
            self.tracker.log_round(server_round, num_clients=len(results))

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if failures:
            print(
                f"[round {server_round}] WARNING: {len(failures)} client(s) failed.",
                flush=True,
            )

        return aggregated_parameters, metrics
