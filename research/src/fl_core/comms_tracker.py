"""Communication-volume instrumentation for Phase 2 and beyond.

Every Phase 3+ compression result in the paper is expressed as a ratio
relative to the uncompressed FedAvg baseline measured here.  If this is not
instrumented correctly in Phase 2, all later efficiency claims become
unverifiable.

Per round, FedAvg transfers:
    server → client:  num_clients × model_bytes   (global model broadcast)
    client → server:  num_clients × model_bytes   (gradient update upload)
    total            = 2 × num_clients × model_bytes

All values are in bytes.  Gigabytes are derived on request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch.nn as nn


def model_bytes(model: nn.Module, dtype_bytes: int = 4) -> int:
    """Count the total bytes in all trainable parameters of *model*.

    Args:
        model:       The PyTorch model.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).

    Returns:
        Total parameter bytes (int).
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total * dtype_bytes


@dataclass
class CommsTracker:
    """Accumulate per-round communication volume for one simulation run.

    Usage::

        tracker = CommsTracker(bytes_per_model=model_bytes(model))
        # Inside the server's aggregate_fit callback:
        tracker.log_round(server_round, num_clients=len(results))
        # At the end of the run:
        summary = tracker.summary()
        mlflow.log_metrics(summary)
    """

    bytes_per_model: int
    rounds: list[dict] = field(default_factory=list)

    def log_round(self, server_round: int, num_clients: int) -> dict:
        """Record one FL round and return the round-level stats."""
        download = num_clients * self.bytes_per_model   # server→client
        upload = num_clients * self.bytes_per_model     # client→server
        total = download + upload
        entry = {
            "round": server_round,
            "num_clients": num_clients,
            "download_bytes": download,
            "upload_bytes": upload,
            "round_bytes": total,
        }
        self.rounds.append(entry)
        return entry

    def summary(self) -> dict:
        """Return cumulative stats suitable for ``mlflow.log_metrics``."""
        if not self.rounds:
            return {}
        total_bytes = sum(r["round_bytes"] for r in self.rounds)
        total_download = sum(r["download_bytes"] for r in self.rounds)
        total_upload = sum(r["upload_bytes"] for r in self.rounds)
        return {
            "comms/model_bytes": self.bytes_per_model,
            "comms/total_gb": total_bytes / 1e9,
            "comms/total_download_gb": total_download / 1e9,
            "comms/total_upload_gb": total_upload / 1e9,
            "comms/rounds_completed": len(self.rounds),
            "comms/avg_gb_per_round": total_bytes / len(self.rounds) / 1e9,
        }

    def save(self, path: Path) -> None:
        """Persist full per-round log as JSON (for offline analysis)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bytes_per_model": self.bytes_per_model,
            "summary": self.summary(),
            "rounds": self.rounds,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
