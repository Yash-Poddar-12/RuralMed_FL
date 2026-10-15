"""Per-round client dropout sampler.

A *dropout* event means the selected client fails to return an update for
the current round — modelling a clinic that loses connectivity entirely
mid-round.  Once a client drops in a round, its partial local progress is
fully discarded (not buffered for the next round).  This is the standard
assumption in FedBuff, FedAsync, and most async-FL literature and is the
safest, most literature-consistent default.

Design decision recorded (Phase 4 spec §3):
  Dropout → discard.  Preserving partial progress would require the client
  to store a local checkpoint and the server to track per-client pending
  state across rounds, which is out of scope until Phase 5's trace-driven
  engine needs it.

The dropout probability can optionally be correlated with per-client data
volume (clients with larger partitions may have higher connectivity demands
and therefore a higher probability of congestion/failure).
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence


class DropoutSampler:
    """Sample which clients drop out each round.

    Args:
        dropout_prob:     Base probability [0, 1] that a selected client
                          drops out during a round.
        client_sizes:     Optional list of per-client training-set sizes
                          (e.g. number of images).  When provided, the
                          dropout probability for client *i* is scaled
                          proportionally so that larger clients have a
                          modestly higher dropout risk — reflecting that
                          high-volume clinics are more likely to encounter
                          bandwidth saturation.
        size_scale:       Controls how much the size imbalance is allowed
                          to inflate dropout probability.  A value of 0.0
                          means all clients share the same base probability
                          (size-independent); 1.0 means the largest client
                          can have up to 2× the base probability.
        seed:             Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        dropout_prob: float,
        client_sizes: Optional[Sequence[int]] = None,
        size_scale: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1]; got {dropout_prob}")
        self.base_prob = dropout_prob
        self.size_scale = float(size_scale)
        self._rng = random.Random(seed)

        # Normalised size weights ∈ [1, 1 + size_scale]
        if client_sizes is not None and len(client_sizes) > 0:
            mn = min(client_sizes)
            mx = max(client_sizes)
            span = (mx - mn) or 1  # avoid /0 when all sizes equal
            self._size_weight = [
                1.0 + size_scale * (s - mn) / span for s in client_sizes
            ]
        else:
            self._size_weight = None  # type: ignore[assignment]

    def sample_dropouts(self, client_ids: List[int]) -> List[int]:
        """Return the subset of *client_ids* that survive (are NOT dropped).

        Args:
            client_ids: Clients selected to participate this round.

        Returns:
            List of client IDs that will actually complete and return updates.
        """
        survivors: List[int] = []
        for cid in client_ids:
            if self._size_weight is not None and cid < len(self._size_weight):
                p = min(1.0, self.base_prob * self._size_weight[cid])
            else:
                p = self.base_prob
            if self._rng.random() >= p:  # NOT dropping out
                survivors.append(cid)
        return survivors

    def sample_dropouts_mask(self, client_ids: List[int]) -> dict[int, bool]:
        """Return {client_id: survived} dict for this round."""
        survivors = set(self.sample_dropouts(client_ids))
        return {cid: cid in survivors for cid in client_ids}
