"""Paired block bootstrap over consecutive 7-day blocks (notebook cell 76).

Blocks approximate temporal dependence. They don't remove dependence between candidates
across blocks, and they don't correct for hyperparameter selection.
"""
from __future__ import annotations

import numpy as np

BLOCK_DAYS = 7
N_RESAMPLES = 2000
SEED = 42


class BlockBootstrap:
    """Same resampled blocks for every comparison, so differences stay paired."""

    def __init__(self, days: np.ndarray, n: int = N_RESAMPLES, seed: int = SEED):
        self.blocks = (days - days.min()).astype("timedelta64[D]").astype(int) // BLOCK_DAYS
        self.unique = np.unique(self.blocks)
        self.counts = np.array([(self.blocks == b).sum() for b in self.unique])
        self.resamples = np.random.default_rng(seed).integers(0, len(self.unique), size=(n, len(self.unique)))

    def interval(self, values: np.ndarray, mask=None, level: float = 0.95) -> tuple[float, float, float]:
        """Mean and interval of `values`, optionally over a subset of queries."""
        mask = np.ones(len(values), dtype=bool) if mask is None else mask
        sums = np.array([values[(self.blocks == b) & mask].sum() for b in self.unique])
        counts = np.array([((self.blocks == b) & mask).sum() for b in self.unique])
        replicas = sums[self.resamples].sum(axis=1) / counts[self.resamples].sum(axis=1)
        tail = (1 - level) / 2
        low, high = np.quantile(replicas, [tail, 1 - tail])
        return values[mask].mean(), low, high

    def difference(self, a: np.ndarray, b: np.ndarray, mask=None) -> tuple[float, float, float]:
        return self.interval(a - b, mask)
