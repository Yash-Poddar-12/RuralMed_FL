"""Connectivity simulation components added from Phase 5 onward."""
from research.src.simulation.connectivity_states import (
    ConnState,
    ConnectivityProfile,
    STATE_BANDWIDTH_KBPS,
    PROFILES,
    get_profile,
)
from research.src.simulation.client_profiles import build_client_profiles, profile_summary
from research.src.simulation.markov_engine import MarkovConnectivityEngine
from research.src.simulation.bandwidth_delay import BandwidthDelayAdapter

__all__ = [
    "ConnState", "ConnectivityProfile", "STATE_BANDWIDTH_KBPS", "PROFILES", "get_profile",
    "build_client_profiles", "profile_summary",
    "MarkovConnectivityEngine",
    "BandwidthDelayAdapter",
]
