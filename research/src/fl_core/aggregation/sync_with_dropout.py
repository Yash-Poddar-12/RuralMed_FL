"""Naïve synchronous FedAvg + dropout baseline for Phase 4.

This is Baseline #2 from Phase 4 spec §5:
  "Synchronous FedAvg, but with the same dropout/straggler model applied."

Timeout / fallback policy (documented explicitly per spec §5):
  When a client drops out (``is_alive == 0``), its update is discarded.
  The aggregation proceeds with however many clients did return updates,
  weighted by sample count exactly as in standard FedAvg.  There is NO
  timeout wait and NO round-skip — the server simply aggregates the
  surviving clients' updates as-is.

  Rationale: This is the most common synchronous-FL-under-dropout policy
  in the literature (FedProx, SCAFFOLD papers use it as their degraded
  baseline).  It is also the *fairest* comparison for FedBuff because it
  gives the synchronous baseline the same minimum-viable-update guarantee
  (at least 1 client must survive) rather than making it artificially
  slow by requiring a full cohort.

Compression compatibility:
  Identical to FedBuffStrategy — compression lives in XRayClient.fit();
  this class is oblivious to it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from research.src.fl_core.comms_tracker import CommsTracker
from research.src.fl_core.straggler_model.dropout_sampler import DropoutSampler


class SyncWithDropout(FedAvg):
    """Standard FedAvg aggregation with a pluggable dropout model applied.

    Args:
        tracker:          CommsTracker instance (shared with the run script).
        dropout_sampler:  DropoutSampler that decides which clients drop out.
                          Pass None to run without any dropout (equivalent to
                          the Phase 2 baseline at that point).
        **kwargs:         Forwarded verbatim to FedAvg.
    """

    def __init__(
        self,
        tracker: CommsTracker,
        dropout_sampler: Optional[DropoutSampler] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tracker = tracker
        self.dropout_sampler = dropout_sampler

        # Effective participation tracking (mirrors FedBuffStrategy API)
        self.participation_counts: Dict[int, int] = {}

    # ── Flower hooks ──────────────────────────────────────────────────────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Mark which clients will drop out via the is_alive config field."""
        fit_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        selected_ids = [
            int(proxy.cid) if proxy.cid.isdigit() else idx
            for idx, (proxy, _) in enumerate(fit_instructions)
        ]

        if self.dropout_sampler is not None and self.dropout_sampler.base_prob > 0.0:
            surviving_ids = set(self.dropout_sampler.sample_dropouts(selected_ids))
        else:
            surviving_ids = set(selected_ids)

        for i, (proxy, fit_ins) in enumerate(fit_instructions):
            cid = selected_ids[i]
            new_config = dict(fit_ins.config)
            new_config["round"] = server_round
            new_config["is_alive"] = int(cid in surviving_ids)
            fit_instructions[i] = (proxy, FitIns(fit_ins.parameters, new_config))

        return fit_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Standard FedAvg aggregation after filtering dropped clients."""
        if not results:
            return None, {}

        # ── Filter dropped clients ────────────────────────────────────────
        active_results = [
            (proxy, fit_res)
            for proxy, fit_res in results
            if int(fit_res.metrics.get("is_alive", 1)) != 0
        ]

        n_dropped = len(results) - len(active_results)
        if n_dropped > 0:
            print(
                f"[SyncDropout round {server_round}] {n_dropped}/{len(results)} "
                f"clients dropped — aggregating {len(active_results)} survivors.",
                flush=True,
            )

        if not active_results:
            print(
                f"[SyncDropout round {server_round}] All clients dropped — "
                f"no update this round.",
                flush=True,
            )
            return self.initial_parameters, {}

        # ── Comms tracking ────────────────────────────────────────────────
        has_compressed = any(
            "wire_bytes" in fit_res.metrics for _, fit_res in active_results
        )
        if has_compressed:
            upload_bytes = sum(
                int(fit_res.metrics.get("wire_bytes", self.tracker.bytes_per_model))
                for _, fit_res in active_results
            )
        else:
            upload_bytes = None
        self.tracker.log_round(
            server_round,
            num_clients=len(active_results),
            custom_upload_bytes=upload_bytes,
        )

        # ── Track participation ───────────────────────────────────────────
        for _, fit_res in active_results:
            cid = int(fit_res.metrics.get("client_id", -1))
            self.participation_counts[cid] = (
                self.participation_counts.get(cid, 0) + 1
            )

        # ── Weighted FedAvg over survivors only ──────────────────────────
        # Re-use FedAvg's own aggregation by passing only active_results.
        aggregated_parameters, agg_metrics = super().aggregate_fit(
            server_round, active_results, failures
        )

        agg_metrics["sync_dropout/n_survived"] = float(len(active_results))
        agg_metrics["sync_dropout/n_dropped"] = float(n_dropped)
        agg_metrics["sync_dropout/dropout_rate"] = float(
            n_dropped / max(len(results), 1)
        )

        return aggregated_parameters, agg_metrics

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def participation_summary(self) -> Dict[str, float]:
        """Return per-client effective participation counts."""
        return {
            f"participation/client_{k}": float(v)
            for k, v in sorted(self.participation_counts.items())
        }
