"""Learned P(activity): how likely a job is to get another process within H days.

Beside the fixed exponential weight of the notebook. The label measures future activity,
which is a proxy for being open: a job can stay open without processes in the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SNAPSHOT_EVERY_DAYS = 7
VOLUME_TAU = 30.0
HORIZONS = (7, 30)
LAMBDAS = (0.25, 0.5, 1.0)


def _features(lag: np.ndarray, age: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """log lag (no activity yet = its own flag), log age, log recent volume."""
    no_activity = ~np.isfinite(lag)
    known_lag = np.where(no_activity, 0.0, lag)
    return np.column_stack([np.log1p(known_lag), no_activity.astype(float),
                            np.log1p(np.maximum(age, 0)), np.log1p(volume)])


def snapshots(train_days: np.ndarray, train_jobs: np.ndarray, job_dates: np.ndarray, horizon: int,
              cutoff_day: int, every: int = SNAPSHOT_EVERY_DAYS, start_day: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Job-day rows from training data only.

    A snapshot on day t uses events before t. Its label asks for an event in [t, t + H).
    Days within H of `cutoff_day` have an incomplete label window, so they are censored.
    Jobs enter once published and with a known date. `start_day` allows a later window, used
    only to check calibration on a period the model didn't see.
    """
    n_jobs = len(job_dates)
    created = job_dates.astype("datetime64[D]").astype("int64").astype(float)
    created[np.isnat(job_dates)] = np.inf
    order = np.argsort(train_days, kind="stable")
    days, jobs = train_days[order], train_jobs[order]
    first = int(days.min()) + every if start_day is None else start_day
    last_t = cutoff_day - horizon
    xs, ys = [], []
    for t in range(first, last_t + 1, every):
        seen = days < t
        last = np.full(n_jobs, -np.inf)
        np.maximum.at(last, jobs[seen], days[seen])
        volume = np.bincount(jobs[seen], weights=np.exp(-(t - days[seen]) / VOLUME_TAU), minlength=n_jobs)
        live = created <= t
        future = (days >= t) & (days < t + horizon)
        label = np.zeros(n_jobs, dtype=bool)
        label[jobs[future]] = True
        lag = t - last
        xs.append(_features(lag[live], (t - created)[live], volume[live]))
        ys.append(label[live])
    return np.concatenate(xs), np.concatenate(ys)


def fit_activity_model(x: np.ndarray, y: np.ndarray, seed: int = 42):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    return model.fit(x, y)


def predict_log_activity(model, states: dict, volume_tau_key: int = 30) -> dict:
    """log P(activity) per job, for each day's state. Unknown ages are treated as age zero."""
    out = {}
    for day, st in states.items():
        age = np.where(np.isfinite(st["age"]), st["age"], 0.0)
        x = _features(st["lag"], age, st["decay"][volume_tau_key])
        out[day] = np.log(np.clip(model.predict_proba(x)[:, 1], 1e-15, 1))
    return out


def calibration_table(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    groups = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, bins - 1)
    return pd.DataFrame({"Predicho": prob, "Observado": y.astype(float), "Grupo": groups}).groupby("Grupo").agg(
        Filas=("Predicho", "size"), Predicho=("Predicho", "mean"), Observado=("Observado", "mean"))
