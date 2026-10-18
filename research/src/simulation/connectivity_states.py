"""Markov-chain connectivity state definitions for Phase 5.

Each simulated clinic moves between discrete connectivity states according to
a transition probability matrix.  The state encodes both *whether* the client
is reachable and the *available bandwidth* while connected.

States (Approach A — recommended in Phase 5 spec §3.1):
    GOOD        — stable, decent rural broadband / 4G
    POOR        — throttled, congested, or weak signal
    DISCONNECTED — no connection (radio link lost / power cut)

Transition matrix rows index the current state; columns index the next state.
Entries are per-round probabilities (i.e. evaluated once per FL round).

Design decisions recorded (Phase 5 spec §3.2):
    • Three client profile tiers are defined below: STABLE, INTERMITTENT,
      UNRELIABLE.  Profile assignment (see client_profiles.py) correlates
      *negatively* with client data volume: larger clinics (Phase 1 data
      imbalance) are assumed to have better infrastructure.  This is the
      more realistic assumption for rural India / low-resource settings and
      is documented here explicitly so any accuracy-disparity observations
      in the final report can be interpreted correctly.
    • Dropout (DISCONNECTED state) feeds Phase 4's DropoutSampler interface
      unchanged — the aggregation strategy sees only a boolean "alive" flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List


# ── State enum ───────────────────────────────────────────────────────────────

class ConnState(IntEnum):
    GOOD         = 0
    POOR         = 1
    DISCONNECTED = 2


# ── Per-state bandwidth ranges (kbps) ────────────────────────────────────────
# Based on rural/intermittent cellular network characterisation from:
#   - TRAI India rural broadband report (2023): median 4G ~5–15 Mbps,
#     median 3G/edge ~64–256 kbps, frequent connectivity gaps.
#   - Used to calibrate Markov state bandwidths; Approach A per spec §3.1.
STATE_BANDWIDTH_KBPS: Dict[ConnState, tuple[float, float]] = {
    ConnState.GOOD:         (1024.0, 5120.0),   # 1–5 Mbps
    ConnState.POOR:         (64.0,   512.0),     # 64–512 kbps (edge/weak 4G)
    ConnState.DISCONNECTED: (0.0,    0.0),       # no transfer
}


# ── Transition matrices per connectivity profile ──────────────────────────────
# Rows: current state  (GOOD, POOR, DISCONNECTED)
# Cols: next state     (GOOD, POOR, DISCONNECTED)

@dataclass
class ConnectivityProfile:
    """Markov-chain connectivity profile for one client tier.

    Args:
        name:              Human-readable profile name.
        transition_matrix: 3×3 row-stochastic matrix.
        initial_state:     Starting state for a new simulation.
        description:       Short summary for reports/logs.
    """
    name: str
    transition_matrix: List[List[float]]
    initial_state: ConnState = ConnState.GOOD
    description: str = ""

    def __post_init__(self) -> None:
        for i, row in enumerate(self.transition_matrix):
            total = sum(row)
            assert abs(total - 1.0) < 1e-6, (
                f"Row {i} of profile '{self.name}' does not sum to 1.0 (got {total})"
            )


# ── Three canonical rural clinic profiles ────────────────────────────────────

PROFILES: Dict[str, ConnectivityProfile] = {

    # Tier 1 — Large, well-resourced clinic (district hospital level)
    # Mostly stable connectivity; occasional brief degradation; rare dropout.
    "stable": ConnectivityProfile(
        name="stable",
        description="Large clinic, district-level infrastructure, reliable link",
        initial_state=ConnState.GOOD,
        transition_matrix=[
            # → GOOD   POOR   DISC
            [0.85,   0.12,  0.03],   # from GOOD
            [0.50,   0.40,  0.10],   # from POOR
            [0.60,   0.30,  0.10],   # from DISCONNECTED
        ],
    ),

    # Tier 2 — Mid-size rural primary health centre
    # Alternates between GOOD and POOR; drops out with moderate frequency.
    "intermittent": ConnectivityProfile(
        name="intermittent",
        description="Rural PHC, shared cellular backhaul, frequent fluctuation",
        initial_state=ConnState.POOR,
        transition_matrix=[
            # → GOOD   POOR   DISC
            [0.55,   0.33,  0.12],   # from GOOD
            [0.30,   0.45,  0.25],   # from POOR
            [0.25,   0.45,  0.30],   # from DISCONNECTED
        ],
    ),

    # Tier 3 — Remote sub-centre, satellite or single-SIM link
    # Spends significant time fully disconnected; the worst-case clinic.
    "unreliable": ConnectivityProfile(
        name="unreliable",
        description="Remote sub-centre, single-SIM / satellite, frequent outages",
        initial_state=ConnState.DISCONNECTED,
        transition_matrix=[
            # → GOOD   POOR   DISC
            [0.35,   0.30,  0.35],   # from GOOD
            [0.15,   0.30,  0.55],   # from POOR
            [0.10,   0.20,  0.70],   # from DISCONNECTED
        ],
    ),
}


def get_profile(name: str) -> ConnectivityProfile:
    """Return a named connectivity profile (case-insensitive)."""
    key = name.lower()
    if key not in PROFILES:
        raise KeyError(
            f"Unknown connectivity profile '{name}'. "
            f"Available: {list(PROFILES.keys())}"
        )
    return PROFILES[key]
