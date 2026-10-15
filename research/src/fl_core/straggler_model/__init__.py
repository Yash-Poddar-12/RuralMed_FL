"""Straggler and dropout models for Phase 4.

This package is intentionally decoupled from the aggregation strategy so that
Phase 5 can swap in a trace-driven connectivity model without touching the
FedBuff aggregation logic.
"""
from .dropout_sampler import DropoutSampler
from .delay_sampler import DelaySampler

__all__ = ["DropoutSampler", "DelaySampler"]
