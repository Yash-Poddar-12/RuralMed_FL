"""Per-client straggler delay sampler.

Models how long each client takes to complete a local training + upload round,
producing a relative *delay score* (not a real-time value — the simulation is
virtual).  Downstream code uses this to decide which clients finish within the
current aggregation window and which are treated as stragglers.

Design decisions (Phase 4 spec §3):
  • Long-tailed (lognormal) distribution is used rather than uniform.  In
    real cross-device FL, a *few very slow* clients dominate tail latency; a
    uniform distribution would understate this effect.
  • Delay parameters are optionally linked to per-client partition size (larger
    data volume → longer local epoch time).  This is more defensible than
    purely random assignment for the final report.
  • Phase 5 will replace this sampler with a trace-driven connectivity engine
    via the same interface: ``sample_delays(client_ids) → Dict[int, float]``.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence


class DelaySampler:
    """Sample normalised delay scores for a set of clients.

    Delay scores ≥ 1.0 (stragglers have higher scores).  A *straggler
    threshold* τ can be set so that any client whose delay score exceeds
    τ is treated as having missed the current aggregation window.

    Args:
        base_delay_mu:    Mean of the underlying normal variable (log-space).
                          Tune this to move the centre of the delay
                          distribution.  Default 0.0 (median delay = 1.0).
        base_delay_sigma: Standard deviation in log-space.  Higher values
                          produce a heavier right tail (more extreme
                          stragglers).  Default 0.5 gives a moderately
                          long-tailed distribution.
        client_sizes:     Optional per-client dataset sizes.  When provided,
                          a client's μ is scaled upward by
                          ``size_scale * (size / max_size)``, so larger
                          clients have higher expected delay.
        size_scale:       Controls the maximum additive boost to μ from the
                          size correlation.  Default 0.4 (≈ 1.5× more
                          expected delay for the largest vs. smallest client).
        straggler_threshold: Delay-score threshold above which a client is
                          classified as a straggler.  None disables threshold
                          classification.
        seed:             Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        base_delay_mu: float = 0.0,
        base_delay_sigma: float = 0.5,
        client_sizes: Optional[Sequence[int]] = None,
        size_scale: float = 0.4,
        straggler_threshold: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.base_mu = base_delay_mu
        self.sigma = base_delay_sigma
        self.straggler_threshold = straggler_threshold
        self._rng = random.Random(seed)

        if client_sizes is not None and len(client_sizes) > 0:
            max_size = max(client_sizes) or 1
            self._mu_boost = [size_scale * (s / max_size) for s in client_sizes]
        else:
            self._mu_boost = None  # type: ignore[assignment]

    def sample_delays(self, client_ids: List[int]) -> Dict[int, float]:
        """Return {client_id: delay_score} for each client in *client_ids*.

        Delay scores are drawn from a lognormal distribution whose μ is
        optionally boosted by the client's data volume.

        Returns:
            Dict mapping each client_id to its sampled delay score (≥ 0).
        """
        delays: Dict[int, float] = {}
        for cid in client_ids:
            if self._mu_boost is not None and cid < len(self._mu_boost):
                mu = self.base_mu + self._mu_boost[cid]
            else:
                mu = self.base_mu
            # Lognormal: X = e^(N(mu, sigma))
            normal_sample = self._rng.gauss(mu, self.sigma)
            delays[cid] = math.exp(normal_sample)
        return delays

    def classify_stragglers(
        self, delays: Dict[int, float]
    ) -> Dict[int, bool]:
        """Return {client_id: is_straggler} given a delay dict.

        Requires ``straggler_threshold`` to be set on construction.

        Args:
            delays: Output of :meth:`sample_delays`.

        Returns:
            Dict where True means the client is a straggler.
        """
        if self.straggler_threshold is None:
            raise ValueError(
                "straggler_threshold must be set to use classify_stragglers()."
            )
        return {cid: d > self.straggler_threshold for cid, d in delays.items()}
