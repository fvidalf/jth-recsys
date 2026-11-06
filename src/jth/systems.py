"""Scorers for every Stage 1 system, shared by the validation and test runs.

Each system is a function from a chunk of queries (`View`) to scores and an eligibility
mask. Chunks keep memory bounded: LR's scores stay on disk and everything else is compact.
The same definition then gives fast ranks for grids (`Inputs.ranks`) and top-1000 lists for
the official check (`Inputs.lists`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from jth.clean_cf import fresh_mask
from jth.context import Context
from jth.fusion import (blend, fit_pairwise, fwls_meta, fwls_pairs, fwls_score, gate_alpha, history_bins,
                        minmax_rows, rrf_pool, sample_negatives)
from jth.ranking import TOP_K, fast_ranks, rank_pool, rank_scores, summarize_ranks

LOG_FLOOR = 1e-15
CHUNK = 500
Scorer = Callable[["SimpleNamespace"], tuple]


@dataclass
class Inputs:
    ctx: Context
    states: dict
    lr: np.ndarray  # scores on disk, one row per cache key
    lr_row: np.ndarray  # row of `lr` for each query
    similarity: sp.csr_matrix  # clean CF, one row per query
    extra: dict = field(default_factory=dict)  # name -> function(query indices) -> scores

    def __post_init__(self):
        days = sorted(self.states)
        pos = {d: i for i, d in enumerate(days)}
        self.query_day = np.array([pos[k[1]] for k in self.ctx.keys])
        stack = lambda key: np.stack([self.states[d][key] for d in days])  # noqa: E731
        self.eligible = stack("eligible")
        self.lag = stack("lag")
        self.age = stack("age")
        self.effective_lag = stack("effective_lag")
        self.decay = {t: np.stack([self.states[d]["decay"][t] for d in days]) for t in self.states[days[0]]["decay"]}
        self.count = stack("count")
        self.meta = fwls_meta(self.ctx.warm, self.ctx.history)
        self.bins = history_bins(self.ctx.history)

    def view(self, ix: np.ndarray) -> SimpleNamespace:
        d = self.query_day[ix]
        return SimpleNamespace(
            ix=ix, day=d, eligible=self.eligible[d], lag=self.lag[d], age=self.age[d],
            effective_lag=self.effective_lag[d], warm=self.ctx.warm[ix], meta=self.meta[ix],
            gold=self.ctx.gold_idx[ix], history=self.ctx.history[ix], inputs=self,
        )

    def chunks(self, mask=None):
        ix = np.arange(len(self.ctx)) if mask is None else np.flatnonzero(mask)
        for start in range(0, len(ix), CHUNK):
            yield self.view(ix[start:start + CHUNK])

    def ranks(self, scorer: Scorer, tie=None) -> np.ndarray:
        tie = self.ctx.tie if tie is None else tie
        out = np.full(len(self.ctx), np.inf)
        for v in self.chunks():
            scores, mask = scorer(v)
            out[v.ix] = fast_ranks(v.gold, scores, mask, tie)
        return out

    def lists(self, scorer: Scorer, tie=None) -> list:
        out = [None] * len(self.ctx)
        for v in self.chunks():
            scores, mask = scorer(v)
            for k, i in enumerate(v.ix):
                out[i] = rank_scores(self.ctx, scores[k], mask[k], tie)
        return out

    def lr_rows(self, ix) -> np.ndarray:
        rows = self.lr_row[ix]
        order = np.argsort(rows)
        chunk = np.empty((len(rows), self.lr.shape[1]))
        chunk[order] = np.asarray(self.lr[rows[order]])
        return chunk

    def sim_rows(self, ix) -> np.ndarray:
        return self.similarity[ix].toarray()


# ------------------------------------------------------------------ base scores
def lr_score(v) -> np.ndarray:
    return v.inputs.lr_rows(v.ix)


def log_lr(v) -> np.ndarray:
    return np.log(np.clip(lr_score(v), LOG_FLOOR, 1))


def cf_input(v) -> np.ndarray:
    """Clean CF for blending: summed similarity, only for jobs inside the official 90-day rule."""
    return v.inputs.sim_rows(v.ix) * fresh_mask(v.age)


def cf_reference(v) -> np.ndarray:
    """Clean CF as its own model: fresh jobs first, by similarity, as the official code orders."""
    from jth.clean_cf import ranked_reference
    return ranked_reference(v.inputs.sim_rows(v.ix), fresh_mask(v.age))


def popularity(v) -> np.ndarray:
    return v.inputs.count[v.day]


def recency(v) -> np.ndarray:
    return -np.where(np.isfinite(v.age), v.age, 1e6)  # an unknown date ranks as the oldest


def random_score(v) -> np.ndarray:
    return np.stack([np.random.default_rng([42, int(i)]).random(v.eligible.shape[1]) for i in v.ix])


def external(name: str) -> Callable:
    def score(v):
        return v.inputs.extra[name](v.ix)
    return score


# ------------------------------------------------------------------ filters
def plain(base) -> Scorer:
    return lambda v: (_finite(base(v), v.eligible), v.eligible)


def _finite(s: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, s, -np.inf)


def hard(base, window: int, grace: int) -> Scorer:
    def score(v):
        with np.errstate(invalid="ignore"):
            new = (v.age >= 0) & (v.age <= grace) if grace > 0 else np.zeros_like(v.eligible)
        mask = v.eligible & ((v.lag <= window) | new)
        return _finite(base(v), v.eligible), mask
    return score


def penalty(v, tau: float, lam: float) -> np.ndarray:
    return -lam * v.effective_lag / tau


def soft_lr(tau: float, lam: float) -> Scorer:
    """The original formula: log p_LR - lambda * lag / tau."""
    return lambda v: (_finite(log_lr(v) + penalty(v, tau, lam), v.eligible), v.eligible)


def soft_normalized(base, tau: float, lam: float) -> Scorer:
    """For scores that aren't probabilities: log of [eps, 1]-normalized score, same penalty."""
    return lambda v: (_finite(np.log(minmax_rows(base(v), v.eligible)) + penalty(v, tau, lam), v.eligible), v.eligible)


def lr_activity_model(log_activity: np.ndarray, lam: float) -> Scorer:
    """log p_LR + lambda * log P(activity). `log_activity` has one row per query day."""
    return lambda v: (_finite(log_lr(v) + lam * log_activity[v.day], v.eligible), v.eligible)


# ------------------------------------------------------------------ blends
def alpha_of(v, alpha, gate: bool):
    """Weight of LR per query: global, gated by history, or one per history bin (dict)."""
    if isinstance(alpha, dict):
        return np.array([alpha[int(min(h, 2))] for h in v.history], dtype=float)
    return gate_alpha(v.warm, alpha) if gate else np.full(len(v.ix), float(alpha))


def score_blend(alpha, gate: bool = False, normalize=minmax_rows) -> Scorer:
    def score(v):
        s = blend(normalize(lr_score(v), v.eligible), normalize(cf_input(v), v.eligible), alpha_of(v, alpha, gate))
        return _finite(s, v.eligible), v.eligible
    return score


def score_blend_after(alpha, gate: bool, tau: float, lam: float) -> Scorer:
    """Combine first, then penalize. Same order as the original fusion + activity."""
    def score(v):
        s = blend(minmax_rows(lr_score(v), v.eligible), minmax_rows(cf_input(v), v.eligible), alpha_of(v, alpha, gate))
        return _finite(np.log(np.maximum(s, LOG_FLOOR)) + penalty(v, tau, lam), v.eligible), v.eligible
    return score


def score_blend_before(alpha, gate: bool, tau: float, lam: float) -> Scorer:
    """Penalize each model first, then combine."""
    def score(v):
        p = penalty(v, tau, lam)
        a = minmax_rows(log_lr(v) + p, v.eligible)
        b = minmax_rows(np.log(minmax_rows(cf_input(v), v.eligible)) + p, v.eligible)
        return _finite(blend(a, b, alpha_of(v, alpha, gate)), v.eligible), v.eligible
    return score


def score_blend_hard_after(alpha, gate: bool, window: int, grace: int) -> Scorer:
    def score(v):
        s = blend(minmax_rows(lr_score(v), v.eligible), minmax_rows(cf_input(v), v.eligible), alpha_of(v, alpha, gate))
        with np.errstate(invalid="ignore"):
            new = (v.age >= 0) & (v.age <= grace) if grace > 0 else np.zeros_like(v.eligible)
        return _finite(s, v.eligible), v.eligible & ((v.lag <= window) | new)
    return score


# ------------------------------------------------------------------ rank fusion over lists
def ranks_from_lists(ctx: Context, lists: list) -> np.ndarray:
    out = np.full(len(lists), np.inf)
    for i, lst in enumerate(lists):
        hit = np.flatnonzero(lst == ctx.gold_idx[i])
        if len(hit):
            out[i] = hit[0] + 1
    return out


def rrf_lists(inp: Inputs, first: list, second: list, alpha: float, gate: bool, soft=None, hard_rule=None) -> list:
    """Weighted RRF over two lists, optionally gated, with activity applied to the merged pool."""
    ctx = inp.ctx
    out = []
    for i in range(len(ctx)):
        a = 1.0 if gate and not ctx.warm[i] else alpha
        pool, s = rrf_pool(first[i], second[i], a, ctx.n_jobs)
        day = inp.query_day[i]
        if soft is not None:
            tau, lam = soft
            s = np.log(np.maximum(s, LOG_FLOOR)) - lam * inp.effective_lag[day][pool] / tau
        elif hard_rule is not None:
            window, grace = hard_rule
            age, lag = inp.age[day][pool], inp.lag[day][pool]
            with np.errstate(invalid="ignore"):
                new = (age >= 0) & (age <= grace) if grace > 0 else np.zeros(len(pool), dtype=bool)
            keep = (lag <= window) | new
            pool, s = pool[keep], s[keep]
        out.append(rank_pool(ctx, pool, s)[:TOP_K])
    return out


# ------------------------------------------------------------------ C: FWLS
def fwls_inputs(v, with_activity: bool) -> list:
    """norm(log p_LR), norm(clean CF) and, optionally, norm(-lag).

    The log and the lag put the activity term on the same footing as the soft filter, so this
    model can express that filter exactly as a special case.
    """
    inputs = [minmax_rows(log_lr(v), v.eligible), minmax_rows(cf_input(v), v.eligible)]
    if with_activity:
        inputs.append(minmax_rows(-v.effective_lag, v.eligible))
    return inputs


def fwls_scorer(weights: np.ndarray, with_activity: bool) -> Scorer:
    return lambda v: (_finite(fwls_score(fwls_inputs(v, with_activity), v.meta, weights), v.eligible), v.eligible)


def fit_fwls(inp: Inputs, fit_mask: np.ndarray, with_activity: bool, c: float, n_uniform: int = 60, n_hard: int = 20,
             seed: int = 42) -> np.ndarray:
    """Pairs from correct jobs vs negatives: uniform eligible jobs plus LR's top jobs."""
    rows = []
    for v in inp.chunks(fit_mask):
        inputs = fwls_inputs(v, with_activity)
        negatives = sample_negatives(v.gold, v.eligible, inputs[0], n_uniform, n_hard, seed=seed + int(v.ix[0]))
        rows.append(fwls_pairs(inputs, v.meta, v.gold, negatives, np.ones(len(v.ix), dtype=bool)))
    return fit_pairwise(np.concatenate(rows), c=c, seed=seed)


# ------------------------------------------------------------------ registry of configurations
class Registry:
    """Every configuration tried, so selection and the total count stay honest."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.rows: list = []

    def add(self, family: str, config: dict, ranks: np.ndarray) -> np.ndarray:
        self.rows.append({"family": family, "config": dict(config), "ranks": ranks})
        return ranks

    def family(self, name: str) -> list:
        return [r for r in self.rows if r["family"] == name]

    def table(self, name: str, mask=None) -> pd.DataFrame:
        out = []
        for r in self.family(name):
            out.append({**summarize_ranks(self.ctx, r["ranks"], mask), **r["config"]})
        return pd.DataFrame(out)

    def best(self, name: str, mask=None) -> dict:
        """First enumerated configuration with the highest (MRR, nDCG@10), as in the original."""
        best, key = None, None
        for r in self.family(name):
            s = summarize_ranks(self.ctx, r["ranks"], mask)
            k = (s["MRR"], s["nDCG@10"])
            if key is None or k > key:
                best, key = r, k
        return best

    def __len__(self) -> int:
        return len(self.rows)
