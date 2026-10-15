"""Phase 4 aggregation strategies.

Two strategies are provided:
  - FedBuffStrategy:      Buffered asynchronous aggregation with staleness weighting.
  - SyncWithDropout:      Naïve synchronous FedAvg that applies the same straggler/
                          dropout model as FedBuff (the Phase 5 Section-2 baseline).
"""
from .fedbuff_strategy import FedBuffStrategy
from .sync_with_dropout import SyncWithDropout

__all__ = ["FedBuffStrategy", "SyncWithDropout"]
