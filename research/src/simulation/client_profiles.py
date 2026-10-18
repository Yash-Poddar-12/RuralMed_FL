"""Per-client heterogeneous connectivity profile assignment.

Maps each of the 10 simulated clinics to one of three connectivity tiers
defined in connectivity_states.py.

Design decision (Phase 5 spec §3.2):
    Profile tier correlates *negatively* with client data volume.
    Justification: in the Indian rural health system, larger clinics
    (district hospitals) typically have better infrastructure investment
    than smaller sub-centres.  This assumption is explicitly recorded here
    and cross-references Phase 1's data imbalance (Section 7.2).

    Client data volumes from Phase 1 (train images):
        client_00: 10972   ← STABLE       (large district clinic)
        client_01: 2860    ← UNRELIABLE   (small sub-centre)
        client_02: 17134   ← STABLE       (large clinic)
        client_03: 20722   ← STABLE       (largest clinic)
        client_04: 1150    ← UNRELIABLE   (smallest clinic)
        client_05: 2200    ← UNRELIABLE   (small clinic)
        client_06: 9189    ← INTERMITTENT (mid-size)
        client_07: 5897    ← INTERMITTENT (mid-size)
        client_08: 7956    ← INTERMITTENT (mid-size)
        client_09: 3448    ← UNRELIABLE   (small clinic)

    Threshold logic:
        ≥ 10 000 images → stable
        4 000–9 999     → intermittent
        < 4 000         → unreliable
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from research.src.simulation.connectivity_states import (
    ConnectivityProfile,
    ConnState,
    get_profile,
)


# ── Default assignment (10-client split, Phase 1 volumes) ────────────────────

_DEFAULT_CLIENT_SIZES = [10972, 2860, 17134, 20722, 1150, 2200, 9189, 5897, 7956, 3448]

_SIZE_TO_PROFILE = [
    (10_000, "stable"),
    (4_000,  "intermittent"),
    (0,      "unreliable"),
]


def assign_profile(client_size: int) -> str:
    """Return profile name for a given training-set size."""
    for threshold, name in _SIZE_TO_PROFILE:
        if client_size >= threshold:
            return name
    return "unreliable"


def build_client_profiles(
    client_sizes: Optional[Sequence[int]] = None,
    num_clients: int = 10,
) -> Dict[int, ConnectivityProfile]:
    """Return {client_id: ConnectivityProfile} for all clients.

    Args:
        client_sizes: Optional list of per-client training-set image counts.
                      Falls back to the Phase 1 default sizes if not provided.
        num_clients:  Number of simulated clients (ignored when client_sizes
                      is provided).

    Returns:
        Dict mapping integer client IDs (0-indexed) to their assigned profile.
    """
    sizes = list(client_sizes) if client_sizes is not None else _DEFAULT_CLIENT_SIZES
    profiles: Dict[int, ConnectivityProfile] = {}
    for cid, sz in enumerate(sizes):
        pname = assign_profile(sz)
        profiles[cid] = get_profile(pname)
    return profiles


def profile_summary(profiles: Dict[int, ConnectivityProfile]) -> str:
    """Return a formatted summary table of client connectivity assignments."""
    lines = [
        f"{'Client':>8}  {'Profile':>12}  {'Description'}",
        "-" * 70,
    ]
    tier_counts: Dict[str, int] = {}
    for cid in sorted(profiles):
        p = profiles[cid]
        lines.append(f"{'client_'+str(cid).zfill(2):>8}  {p.name:>12}  {p.description}")
        tier_counts[p.name] = tier_counts.get(p.name, 0) + 1
    lines.append("-" * 70)
    lines.append("Tier distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(tier_counts.items())))
    return "\n".join(lines)
