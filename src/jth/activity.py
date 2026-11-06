"""Daily job activity state and the filters built on it (notebook cells 64 and 68).

Activity here means candidate-job processes: it is a proxy for a job being open, and
its absence is not evidence of closure. An event of day t only affects days after t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from jth.context import Context

TAUS = (7, 30, 90)
HARD_WINDOWS = (7, 14, 30, 60, 90)
GRACE_DAYS = (0, 7)
SOFT_TAUS = (7, 30, 90)
SOFT_LAMBDAS = (0.25, 0.5, 1.0)


def _to_positions(ctx: Context, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    days = pd.to_datetime(frame["timestamp"]).to_numpy(dtype="datetime64[D]").astype(int)
    idx = ctx.job_index.get_indexer([ctx.evaluator._parse_id(x) for x in frame["job_id"]])
    assert (idx >= 0).all()
    return days, idx


def daily_states(ctx: Context, train: pd.DataFrame, target: pd.DataFrame, taus=TAUS) -> dict:
    """State of every job at the start of each query day.

    Rows of `train` dated on or after the first query day (the official split shares
    2024-03-23 between train and test) are revealed only after that day, like target rows.
    """
    query_days = np.unique(ctx.days)
    first = int(query_days.min().astype(int))
    d_train, i_train = _to_positions(ctx, train)
    d_target, i_target = _to_positions(ctx, target)
    early = d_train < first
    d_later = np.concatenate([d_train[~early], d_target])
    i_later = np.concatenate([i_train[~early], i_target])

    last = np.full(ctx.n_jobs, -np.inf)
    np.maximum.at(last, i_train[early], d_train[early])
    count = np.bincount(i_train[early], minlength=ctx.n_jobs).astype(float)
    decay = {t: np.bincount(i_train[early], weights=np.exp(-(first - d_train[early]) / t), minlength=ctx.n_jobs)
             for t in taus}
    states, previous = {}, first
    for day in query_days:
        day_int = int(day.astype(int))
        for t in taus:
            decay[t] *= np.exp(-(day_int - previous) / t)
        age = (day - ctx.job_dates).astype("timedelta64[D]").astype(float)
        age[np.isnat(ctx.job_dates)] = np.nan
        lag = day_int - last
        assert np.all(lag[np.isfinite(lag)] >= 1), "Solo información anterior al día."
        states[str(day)] = {
            "eligible": ctx.eligible(day), "age": age, "lag": lag,
            "effective_lag": np.where(np.isfinite(lag), lag, np.where(np.isfinite(age), np.maximum(age, 0), 0)),
            "decay": {t: decay[t].copy() for t in taus},
            "count": count.copy(),
        }
        events = i_later[d_later == day_int]  # revealed after the state is saved
        last[events] = day_int
        np.add.at(count, events, 1.0)
        for t in taus:
            np.add.at(decay[t], events, 1.0)
        previous = day_int
    return states


def hard_mask(state: dict, window: int, grace: int) -> np.ndarray:
    """Eligible jobs with activity in the last `window` days, or published in the last `grace`."""
    new = (state["age"] >= 0) & (state["age"] <= grace) if grace > 0 else np.zeros_like(state["eligible"], dtype=bool)
    return state["eligible"] & ((state["lag"] <= window) | new)


def soft_penalty(state: dict, tau: float, lam: float) -> np.ndarray:
    """Additive log-space penalty: `lambda * log exp(-lag / tau)`. Only lambda/tau matters."""
    return -lam * state["effective_lag"] / tau


class StateMatrix:
    """Per-query views of the daily states, as arrays, for fast grids."""

    def __init__(self, ctx: Context, states: dict):
        self.day_list = sorted(states)
        pos = {d: i for i, d in enumerate(self.day_list)}
        self.query_day = np.array([pos[k[1]] for k in ctx.keys])
        self.eligible = np.stack([states[d]["eligible"] for d in self.day_list])[self.query_day]
        self.lag = np.stack([states[d]["lag"] for d in self.day_list])[self.query_day]
        self.age = np.stack([states[d]["age"] for d in self.day_list])[self.query_day]
        self.effective_lag = np.stack([states[d]["effective_lag"] for d in self.day_list])[self.query_day]

    def hard(self, window: int, grace: int) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            new = (self.age >= 0) & (self.age <= grace) if grace > 0 else np.zeros_like(self.eligible, dtype=bool)
        return self.eligible & ((self.lag <= window) | new)

    def penalty(self, tau: float, lam: float) -> np.ndarray:
        return -lam * self.effective_lag / tau
