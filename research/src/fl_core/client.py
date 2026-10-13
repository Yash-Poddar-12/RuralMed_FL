"""Flower NumPyClient for a single simulated rural clinic.

Each virtual client:
    1. Receives the current global model weights from the server.
    2. Trains for ``local_epochs`` on its own partition (train.csv).
    3. Returns the updated weights + sample count + local metrics.

GPU note:
    The device is resolved once at import time and printed explicitly.
    If torch.cuda.is_available() is False the script will abort with a
    clear error rather than silently training on CPU for hours.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from research.src.data_prep.common import REPO_ROOT, load_config
from research.src.data_prep.dataset import ImagingDataset
from research.src.fl_core.models.densenet import (
    build_densenet121,
    get_parameters,
    set_parameters,
)
from research.src.fl_core.comms_tracker import model_bytes
from research.src.fl_core.compression import (
    ModelCompressor,
    flatten_tensor_list,
    unflatten_tensor_list,
)
from research.src.eval.metrics import compute_metrics, format_report


# ── Device resolution (done once, printed on startup) ─────────────────────────
def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n\nCUDA is not available.\n"
            "Ensure:\n"
            "  1. The venv is activated (research\\.venv\\Scripts\\Activate.ps1)\n"
            "  2. NVIDIA drivers >= 525 are installed (nvidia-smi should show driver)\n"
            "  3. torch.cuda.is_available() returns True in a plain python -c check\n"
            "Training DenseNet-121 on CPU would take days — aborting."
        )
    device = torch.device("cuda:0")
    print(
        f"[device] Using GPU: {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)",
        flush=True,
    )
    return device


DEVICE: torch.device = _resolve_device()


# ── Client ────────────────────────────────────────────────────────────────────
class XRayClient(fl.client.NumPyClient):
    """One simulated rural clinic.

    Args:
        partition_id:   Integer client index (0–9 for the 10-clinic split).
        processed_root: Path to the processed image directory (contains
                        primary_manifest.csv and the image sub-tree).
        partitions_root: Path to research/data/partitions/full/public/.
        cfg:            Phase 2 config dict (from phase2_fedavg_baseline.yml).
        device:         Torch device; resolved once at module level.
    """

    def __init__(
        self,
        partition_id: int,
        processed_root: str,
        partitions_root: str,
        cfg: dict,
        device: torch.device = DEVICE,
    ) -> None:
        self.partition_id = partition_id
        self.device = device
        self.cfg = cfg
        self.client_dir = Path(partitions_root) / f"client_{partition_id:02d}"

        # ── Model ─────────────────────────────────────────────────────────────
        model_cfg = cfg["model"]
        self.model = build_densenet121(
            num_classes=model_cfg["num_classes"],
            pretrained=False,   # weights will be set by the server each round
            dropout=model_cfg["dropout"],
        ).to(device)

        # ── Class weights (global, not per-client) ───────────────────────────
        weights = torch.tensor(cfg["data"]["class_weights"], dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(device))

        # ── AMP scaler ────────────────────────────────────────────────────────
        self.use_amp: bool = cfg["training"]["use_amp"]
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # ── Data ──────────────────────────────────────────────────────────────
        train_cfg = cfg["training"]
        self.batch_size: int = train_cfg["batch_size"]
        self.local_epochs: int = train_cfg["local_epochs"]

        self.train_loader = self._make_loader(
            Path(processed_root),
            self.client_dir / "train.csv",
            shuffle=True,
        )
        self.val_loader = self._make_loader(
            Path(processed_root),
            self.client_dir / "val.csv",
            shuffle=False,
        )

        # ── Optimizer (re-created each round with fresh state per McMahan) ───
        self.lr: float = train_cfg["learning_rate"]
        self.wd: float = train_cfg["weight_decay"]

        # ── Compression (Phase 3) ───────────────────────────────────────────
        comp_cfg = cfg.get("compression", {})
        if comp_cfg.get("enabled", False):
            c_dict = dict(comp_cfg)
            if "residual_dir" not in c_dict:
                c_dict["residual_dir"] = REPO_ROOT / "research/artifacts/phase3/residuals"
            self.compressor = ModelCompressor(c_dict)
        else:
            self.compressor = ModelCompressor({"method": "none"})

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_loader(
        self, processed_root: Path, index_path: Path, *, shuffle: bool
    ) -> DataLoader:
        dataset = ImagingDataset(str(processed_root), index_path)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Fresh Adam per round — matches the stateless client assumption in
        McMahan et al. and avoids stale momentum accumulating across rounds."""
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

    # ── Flower NumPyClient interface ──────────────────────────────────────────

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:  # noqa: D102
        return get_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        set_parameters(self.model, parameters)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Local training for ``local_epochs`` and return updated weights."""
        initial_params = [np.array(p, copy=True) for p in parameters]
        self.set_parameters(parameters)
        optimizer = self._build_optimizer()
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(self.local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_loss += loss.item() * len(labels)
                    total_samples += len(labels)

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)

        # ── Apply Phase 3 compression if enabled ─────────────────────────────
        local_params = get_parameters(self.model)
        if self.compressor.method != "none":
            # Compressible parameters: Conv and Linear weights & biases.
            # BatchNorm parameters (containing 'norm' in key or non-floating buffers)
            # are transmitted uncompressed. Per Lin et al. (DGC, ICLR 2018), BatchNorm
            # accounts for < 2.5% of parameters and must be preserved intact to prevent
            # running-variance corruption and numerical divergence (NaNs).
            keys = list(self.model.state_dict().keys())
            compressible_indices = [
                i for i, (k, lp) in enumerate(zip(keys, local_params))
                if "norm" not in k and np.issubdtype(lp.dtype, np.floating)
            ]
            deltas = [
                torch.from_numpy(np.asarray(local_params[i] - initial_params[i])).float()
                for i in compressible_indices
            ]
            flat_delta, meta = flatten_tensor_list(deltas)
            client_id_str = f"client_{self.partition_id:02d}"

            # Dynamic warmup schedule: use higher density for early rounds if configured
            server_round = int(config.get("round", 1))
            warmup_rounds = int(self.compressor.cfg.get("warmup_rounds", 0))
            if server_round <= warmup_rounds:
                eff_ratio = float(self.compressor.cfg.get("warmup_ratio", 0.25))
            else:
                eff_ratio = self.compressor.topk_ratio

            payload, wire_bytes = self.compressor.compress_update(
                flat_delta.to(self.device),
                client_id=client_id_str,
                ratio=eff_ratio,
            )
            reconstructed_flat = self.compressor.decompress_update(payload, device=self.device)
            reconstructed_deltas = unflatten_tensor_list(reconstructed_flat, meta)

            transmitted_params = list(local_params)
            for idx, rd in zip(compressible_indices, reconstructed_deltas):
                ip = initial_params[idx]
                lp = local_params[idx]
                transmitted_params[idx] = (ip + rd.cpu().numpy()).astype(lp.dtype)

            # Account for exact wire bytes of uncompressed BatchNorm parameters and buffers
            uncompressed_bytes = sum(
                local_params[i].nbytes for i in range(len(local_params))
                if i not in compressible_indices
            )
            wire_bytes += uncompressed_bytes

            set_parameters(self.model, transmitted_params)
        else:
            wire_bytes = model_bytes(self.model)

        metrics: Dict[str, Scalar] = {
            "train_loss": float(avg_loss),
            "train_accuracy": float(accuracy),
            "client_id": self.partition_id,
            "wire_bytes": float(wire_bytes),
        }
        return get_parameters(self.model), total_samples, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate on the client's local validation split."""
        self.set_parameters(parameters)
        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_labels: list[int] = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels_dev = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels_dev)

                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.extend(labels.numpy().tolist())
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.array(all_labels)
        avg_loss = total_loss / max(total_samples, 1)

        metrics = compute_metrics(y_true, y_prob, prefix="val_")
        metrics["val_loss"] = float(avg_loss)
        metrics["client_id"] = float(self.partition_id)

        return float(avg_loss), total_samples, metrics
