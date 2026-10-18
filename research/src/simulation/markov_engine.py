"""Markov-chain connectivity engine — per-client state machine.

Each client has its own independent Markov chain over
{GOOD, POOR, DISCONNECTED} with transition probabilities defined by
its ConnectivityProfile (from connectivity_states.py).

This module is the runtime engine: it holds state and advances it
one step per FL round per client.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from research.src.simulation.connectivity_states import ConnState, ConnectivityProfile
from research.src.simulation.client_profiles import build_client_profiles


class MarkovConnectivityEngine:
    """Per-client Markov-chain connectivity simulator.

    Args:
        profiles:   {client_id: ConnectivityProfile} as returned by
                    build_client_profiles().
        seed:       Optional RNG seed for reproducible state transitions.
        bw_cap_kbps: Optional global bandwidth cap applied on top of the
                    per-state bandwidth range (e.g. to simulate a national
                    bandwidth throttle for an entire sweep point).
                    None means no cap.
    """

    def __init__(
        self,
        profiles: Dict[int, ConnectivityProfile],
        seed: Optional[int] = None,
        bw_cap_kbps: Optional[float] = None,
    ) -> None:
        self.profiles = profiles
        self.bw_cap_kbps = bw_cap_kbps
        self._rng = random.Random(seed)

        # Initialise each client at its profile's starting state
        self._state: Dict[int, ConnState] = {
            cid: profile.initial_state
            for cid, profile in profiles.items()
        }
        # Full history (for trace visualisation and diagnostics)
        self._history: Dict[int, List[ConnState]] = {
            cid: [profile.initial_state]
            for cid, profile in profiles.items()
        }

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(self, client_id: int) -> ConnState:
        """Advance client_id's Markov chain one step and return the new state.

        Args:
            client_id: The integer ID of the client to step.

        Returns:
            The new ConnState after the transition.
        """
        profile = self.profiles[client_id]
        current = self._state[client_id]
        row = profile.transition_matrix[int(current)]
        r = self._rng.random()
        cumulative = 0.0
        new_state = ConnState.DISCONNECTED
        for next_idx, prob in enumerate(row):
            cumulative += prob
            if r < cumulative:
                new_state = ConnState(next_idx)
                break

        self._state[client_id] = new_state
        self._history[client_id].append(new_state)
        return new_state

    def step_all(self, client_ids: Optional[List[int]] = None) -> Dict[int, ConnState]:
        """Step all (or a subset of) clients and return their new states."""
        ids = client_ids if client_ids is not None else list(self.profiles.keys())
        return {cid: self.step(cid) for cid in ids}

    def current_state(self, client_id: int) -> ConnState:
        """Return the current state of client_id without advancing."""
        return self._state[client_id]

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def uptime_fraction(self, client_id: int) -> float:
        """Fraction of recorded steps where the client was NOT disconnected."""
        hist = self._history[client_id]
        if len(hist) <= 1:
            return 1.0
        connected = sum(1 for s in hist[1:] if s != ConnState.DISCONNECTED)
        return connected / (len(hist) - 1)

    def state_history(self, client_id: int) -> List[ConnState]:
        """Return the full list of states visited by client_id."""
        return list(self._history[client_id])

    def summary(self) -> Dict[str, float]:
        """Return per-client uptime fractions keyed for MLflow logging."""
        return {
            f"connectivity/uptime_client_{cid:02d}": self.uptime_fraction(cid)
            for cid in sorted(self.profiles.keys())
        }

    @classmethod
    def from_client_sizes(
        cls,
        client_sizes: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        bw_cap_kbps: Optional[float] = None,
    ) -> "MarkovConnectivityEngine":
        """Convenience constructor that auto-assigns profiles from data volumes."""
        profiles = build_client_profiles(client_sizes)
        return cls(profiles=profiles, seed=seed, bw_cap_kbps=bw_cap_kbps)
