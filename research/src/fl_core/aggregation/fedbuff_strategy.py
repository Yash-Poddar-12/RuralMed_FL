"""FedBuff-style buffered asynchronous aggregation strategy for Phase 4.

Core mechanism (Phase 4 spec §4.1–4.3):
  1. The server dispatches the current global model to all *available* clients.
  2. Client updates arrive asynchronously; each carries its starting staleness
     (how many global model versions have been published since the client's
     round began).
  3. Incoming updates are accumulated in a buffer of size K.  Once the buffer
     reaches K entries, the server performs one aggregation step:
         new_global = Σ_i  w_i · θ_i   /   Σ_i w_i
       where  w_i = n_i / (1 + τ_i)^α   (sample count × staleness penalty).
  4. The buffer is immediately cleared; the server continues accepting updates.

Staleness weighting (§4.2):
  Weight each update inversely by its staleness:
       staleness_weight(τ) = (1 + τ)^{−α}      α = staleness_alpha (config)
  Combined with the per-client sample count this gives:
       w_i = n_i × (1 + τ_i)^{−α}

Buffer size K (§4.3):
  Sweep K ∈ {1, 3, 5} to reproduce the accuracy/wall-clock tradeoff.

Compatibility with Phase 3 compression (§4.4):
  This strategy is agnostic to how clients compress their updates.  The
  compression logic lives entirely inside XRayClient.fit(); this class only
  sees the returned NDArrays and sample counts, so both can be active
  simultaneously without modification.

Dropout integration:
  The strategy accepts a ``dropout_mask`` dict {client_id: survived} injected
  via ``configure_fit``'s round config.  Clients that are marked as dropped
  simply never call fit(), so their contribution never enters the buffer —
  the strategy does not need to detect timeouts.

Flower compatibility note:
  Flower's simulation backend still runs rounds synchronously.  We implement
  *virtual asynchrony* by:
    - Using fraction_fit to select more clients than K each round.
    - Applying the dropout sampler to determine which of those submit updates.
    - Treating survivors' updates as the buffer contents for that step.
  This is equivalent to the buffered-async behaviour in expectation and matches
  how async FL is benchmarked in simulation throughout the FedBuff paper.
"""
from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from research.src.fl_core.comms_tracker import CommsTracker
from research.src.fl_core.straggler_model.dropout_sampler import DropoutSampler


class FedBuffStrategy(FedAvg):
    """Buffered asynchronous aggregation with staleness-weighted contributions.

    Args:
        tracker:          CommsTracker instance (shared with the run script).
        dropout_sampler:  DropoutSampler that decides which selected clients
                          survive each round.  Pass None to disable dropout.
        buffer_size:      K — number of client updates to accumulate before
                          triggering an aggregation step.
        staleness_alpha:  Exponent α in the staleness weight
                          (1 + τ)^{−α}.  Higher values penalise stale
                          updates more aggressively.  0.0 = no penalty.
        **kwargs:         Forwarded verbatim to FedAvg.
    """

    def __init__(
        self,
        tracker: CommsTracker,
        dropout_sampler: Optional[DropoutSampler] = None,
        buffer_size: int = 3,
        staleness_alpha: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tracker = tracker
        self.dropout_sampler = dropout_sampler
        self.buffer_size = max(1, buffer_size)
        self.staleness_alpha = staleness_alpha

        # Tracks the current global model version (increments after each
        # aggregation step, which may not coincide with a Flower round).
        self._global_version: int = 0

        # Accumulator for client updates between aggregation steps.
        # Each entry: (ndarray_list, num_examples, staleness)
        self._buffer: List[Tuple[NDArrays, int, int]] = []

        # Per-client staleness tracking: {client_id → version when dispatched}
        self._dispatch_version: Dict[int, int] = {}

        # Staleness log for diagnostics (logged to MLflow by the run script)
        self.staleness_log: List[int] = []

        # Effective participation tracking: {client_id → count of successful updates}
        self.participation_counts: Dict[int, int] = collections.defaultdict(int)

        # Wire bytes accumulator for compression-aware comms tracking
        self._round_wire_bytes: int = 0

    # ── Flower hooks ──────────────────────────────────────────────────────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Select clients and apply dropout before dispatching.

        All surviving clients receive the current global version in their
        config so they can report back their staleness correctly.
        """
        # Standard FedAvg selection
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)

        # Map proxy → client_id for dropout sampling
        selected_ids = [
            int(proxy.cid) if proxy.cid.isdigit() else idx
            for idx, (proxy, _) in enumerate(fit_instructions)
        ]

        # Apply dropout
        if self.dropout_sampler is not None and self.dropout_sampler.base_prob > 0.0:
            surviving_ids = set(self.dropout_sampler.sample_dropouts(selected_ids))
        else:
            surviving_ids = set(selected_ids)

        # Record dispatch version for staleness tracking
        for i, (proxy, fit_ins) in enumerate(fit_instructions):
            cid = selected_ids[i]
            self._dispatch_version[cid] = self._global_version
            # Inject round + version info into per-client config
            new_config = dict(fit_ins.config)
            new_config["round"] = server_round
            new_config["global_version"] = self._global_version
            new_config["is_alive"] = int(cid in surviving_ids)
            fit_instructions[i] = (proxy, FitIns(fit_ins.parameters, new_config))

        return fit_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Buffer incoming updates and aggregate when buffer reaches K."""
        if not results:
            return None, {}

        # ── Filter out dropped clients (is_alive == 0) ────────────────────
        active_results = [
            (proxy, fit_res)
            for proxy, fit_res in results
            if int(fit_res.metrics.get("is_alive", 1)) != 0
        ]

        if not active_results:
            print(
                f"[FedBuff round {server_round}] All clients dropped — skipping buffer.",
                flush=True,
            )
            return ndarrays_to_parameters(
                parameters_to_ndarrays(self.initial_parameters)
                if self._global_version == 0
                else []
            ), {}

        # ── Accumulate wire bytes ─────────────────────────────────────────
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
        self.tracker.log_round(server_round, num_clients=len(active_results),
                               custom_upload_bytes=upload_bytes)

        # ── Buffer each surviving client's update ─────────────────────────
        for proxy, fit_res in active_results:
            # Resolve client_id
            if proxy.cid.isdigit():
                cid = int(proxy.cid)
            else:
                cid = int(fit_res.metrics.get("client_id", 0))

            dispatch_ver = self._dispatch_version.get(cid, self._global_version)
            staleness = max(0, self._global_version - dispatch_ver)
            self.staleness_log.append(staleness)
            self.participation_counts[cid] += 1

            self._buffer.append((
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
                staleness,
            ))

        # ── Aggregate if buffer has reached K ────────────────────────────
        aggregated_params: Optional[Parameters] = None
        agg_metrics: Dict[str, Scalar] = {}

        while len(self._buffer) >= self.buffer_size:
            batch = self._buffer[: self.buffer_size]
            self._buffer = self._buffer[self.buffer_size:]

            agg_ndarrays = self._weighted_aggregate(batch)
            aggregated_params = ndarrays_to_parameters(agg_ndarrays)
            self._global_version += 1

            agg_metrics["fedbuff/global_version"] = float(self._global_version)
            agg_metrics["fedbuff/buffer_size"] = float(self.buffer_size)
            agg_metrics["fedbuff/mean_staleness"] = float(
                np.mean(self.staleness_log[-self.buffer_size:])
                if self.staleness_log else 0.0
            )

            # Update initial_parameters so Flower's evaluate_fn sees the new model
            self.initial_parameters = aggregated_params

        if failures:
            print(
                f"[FedBuff round {server_round}] {len(failures)} client failure(s).",
                flush=True,
            )

        # If buffer hasn't triggered aggregation yet, return current parameters
        if aggregated_params is None:
            aggregated_params = self.initial_parameters

        return aggregated_params, agg_metrics

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _weighted_aggregate(
        self, batch: List[Tuple[NDArrays, int, int]]
    ) -> NDArrays:
        """Staleness-weighted FedAvg aggregation over a buffer batch.

        w_i = n_i × (1 + τ_i)^{−α}
        """
        weights = np.array([
            n_examples * (1.0 + staleness) ** (-self.staleness_alpha)
            for _, n_examples, staleness in batch
        ], dtype=np.float64)
        total_weight = weights.sum()
        if total_weight == 0.0:
            weights = np.ones(len(batch), dtype=np.float64)
            total_weight = float(len(batch))

        aggregated: Optional[NDArrays] = None
        for (ndarrays, _, _), w in zip(batch, weights):
            scaled = [arr * (w / total_weight) for arr in ndarrays]
            if aggregated is None:
                aggregated = scaled
            else:
                aggregated = [a + b for a, b in zip(aggregated, scaled)]
        return aggregated  # type: ignore[return-value]

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def staleness_summary(self) -> Dict[str, float]:
        """Return summary statistics of observed staleness values."""
        if not self.staleness_log:
            return {"staleness/mean": 0.0, "staleness/max": 0.0, "staleness/p90": 0.0}
        arr = np.array(self.staleness_log, dtype=np.float64)
        return {
            "staleness/mean": float(arr.mean()),
            "staleness/max": float(arr.max()),
            "staleness/p90": float(np.percentile(arr, 90)),
            "staleness/p50": float(np.percentile(arr, 50)),
        }

    def participation_summary(self) -> Dict[str, float]:
        """Return per-client effective participation counts."""
        return {f"participation/client_{k}": float(v)
                for k, v in sorted(self.participation_counts.items())}
