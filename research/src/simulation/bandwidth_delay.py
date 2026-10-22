"""Bandwidth-to-transmission-time conversion and Phase 4 interface adapter.

This module implements Phase 5 spec §3.3 — converting a client's current
Markov-chain bandwidth state and the actual compressed payload size (bytes,
from Phase 3's accounting) into a concrete transmission delay and a binary
dropout decision.

It replaces Phase 4's DropoutSampler + DelaySampler with a connectivity-
trace-driven equivalent behind the **exact same interface**, so the FedBuff
strategy and SyncWithDropout baseline from Phase 4 require zero modifications.

Design decisions:
    • Bandwidth is sampled uniformly from the state's kbps range each round,
      introducing realistic jitter even within a single state.
    • A *round time budget* τ_budget (seconds) is set per experiment; clients
      whose transmission time exceeds τ_budget are treated as stragglers and
      reported as `is_alive=0` (dropped), consistent with Phase 4's discard
      policy.
    • Phase 3 compressed payload size is used directly when available;
      falls back to uncompressed model bytes otherwise — ensuring the
      bandwidth × compression interaction (Phase 5 spec §5) is measured
      correctly.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from research.src.simulation.connectivity_states import ConnState, STATE_BANDWIDTH_KBPS
from research.src.simulation.client_profiles import build_client_profiles
from research.src.simulation.markov_engine import MarkovConnectivityEngine


# ── Constants ────────────────────────────────────────────────────────────────

_BITS_PER_BYTE = 8
_KBPS_TO_BPS   = 1_000


class BandwidthDelayAdapter:
    """Implements Phase 4's straggler-model interface using trace-driven delays.

    This class:
      1. Steps each client's Markov chain one round forward.
      2. Samples an available bandwidth from the resulting state.
      3. Computes transmission time = payload_bytes × 8 / bandwidth_bps.
      4. Returns dropout (is_alive=0) if transmission time > round_budget_s.

    Args:
        engine:           MarkovConnectivityEngine managing per-client state.
        payload_bytes:    Size of the model update payload in bytes.
                          Should be the *compressed* size when Phase 3 is
                          active; uncompressed model bytes otherwise.
        round_budget_s:   Per-round time budget in seconds.  Clients whose
                          computed transmission time exceeds this are dropped.
        seed:             Optional RNG seed for bandwidth jitter.
    """

    def __init__(
        self,
        engine: "MarkovConnectivityEngine",
        payload_bytes: int,
        round_budget_s: float = 300.0,
        seed: Optional[int] = None,
    ) -> None:
        self.engine = engine
        self.payload_bytes = payload_bytes
        self.round_budget_s = round_budget_s
        self._rng = random.Random(seed)

        # Phase 4 interface compatibility: indicates dropout sampling is active
        self.base_prob: float = 1.0

        # Diagnostics
        self.bandwidth_log: Dict[int, List[float]] = {}     # {cid: [kbps per round]}
        self.delay_log:     Dict[int, List[float]] = {}     # {cid: [tx_time_s per round]}
        self.dropout_log:   Dict[int, List[bool]]  = {}     # {cid: [dropped per round]}

    # ── Phase 4-compatible interface ─────────────────────────────────────────

    def sample_dropouts_mask(self, client_ids: List[int]) -> Dict[int, bool]:
        """Return {client_id: survived} dict for this round."""
        survivors = set(self.sample_dropouts(client_ids))
        return {cid: cid in survivors for cid in client_ids}

    def sample_dropouts(self, client_ids: List[int]) -> List[int]:
        """Return the list of clients that survive this round.

        Drop clients whose transmission time exceeds round_budget_s.
        This is the same return type as Phase 4's DropoutSampler.sample_dropouts().
        """
        survivors: List[int] = []
        for cid in client_ids:
            state, bw_kbps, tx_time_s = self._step_client(cid)
            alive = (state != ConnState.DISCONNECTED) and (tx_time_s <= self.round_budget_s)
            if alive:
                survivors.append(cid)
        return survivors

    def sample_delays(self, client_ids: List[int]) -> Dict[int, float]:
        """Return {client_id: transmission_delay_seconds}.

        Compatible with Phase 4's DelaySampler.sample_delays() return type.
        Clients already evaluated this round reuse the cached delay.
        """
        delays: Dict[int, float] = {}
        for cid in client_ids:
            if cid in self.delay_log and self.delay_log[cid]:
                delays[cid] = self.delay_log[cid][-1]
            else:
                _, _, tx_time_s = self._step_client(cid)
                delays[cid] = tx_time_s
        return delays

    # ── Internal ──────────────────────────────────────────────────────────────

    def _step_client(self, cid: int) -> tuple[ConnState, float, float]:
        """Step client cid's Markov chain and compute bandwidth + delay.

        Returns:
            (new_state, bandwidth_kbps, transmission_time_seconds)
        """
        new_state = self.engine.step(cid)
        bw_lo, bw_hi = STATE_BANDWIDTH_KBPS[new_state]

        if new_state == ConnState.DISCONNECTED:
            bw_kbps = 0.0
            tx_time_s = float("inf")
        else:
            bw_kbps = self._rng.uniform(bw_lo, bw_hi)
            if self.engine.bw_cap_kbps is not None:
                bw_kbps = min(bw_kbps, self.engine.bw_cap_kbps)
            bw_bps  = bw_kbps * _KBPS_TO_BPS
            tx_bits = self.payload_bytes * _BITS_PER_BYTE
            tx_time_s = tx_bits / bw_bps

        # Store diagnostics
        self.bandwidth_log.setdefault(cid, []).append(bw_kbps)
        self.delay_log.setdefault(cid, []).append(tx_time_s if tx_time_s != float("inf") else -1.0)
        self.dropout_log.setdefault(cid, []).append(new_state == ConnState.DISCONNECTED or tx_time_s > self.round_budget_s)

        return new_state, bw_kbps, tx_time_s

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def uptime_summary(self) -> Dict[str, float]:
        """Return per-client uptime fraction (rounds alive / total rounds)."""
        summary: Dict[str, float] = {}
        for cid, drops in self.dropout_log.items():
            uptime = 1.0 - (sum(drops) / max(len(drops), 1))
            summary[f"uptime/client_{cid:02d}"] = uptime
        return summary

    def avg_bandwidth_summary(self) -> Dict[str, float]:
        """Return per-client average connected bandwidth in kbps."""
        summary: Dict[str, float] = {}
        for cid, bws in self.bandwidth_log.items():
            connected = [b for b in bws if b > 0.0]
            avg = sum(connected) / max(len(connected), 1)
            summary[f"avg_bw_kbps/client_{cid:02d}"] = avg
        return summary
