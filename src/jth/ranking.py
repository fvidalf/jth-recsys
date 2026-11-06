"""Rankings and metrics over a Context (scripts/01_datos_y_validacion.ipynb, cell 49).

Two scoring paths exist on purpose:

- `metrics_from_lists` + `verify_official`: the exact path. It builds top-1000 lists and
  checks every number against the official evaluator. Winners always go through it.
- `fast_ranks`: a vectorized rank of the relevant job, for grids. Tested to match the exact
  path, including the top-1000 cut and tie-breaks.
"""
from __future__ import annotations

import random

import numpy as np

from jth.context import Context

TOP_K = 1000


def rank_scores(ctx: Context, scores, mask, tie=None) -> np.ndarray:
    """Top-1000 job positions among `mask`, by score, then by fixed tie priority."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return np.array([], dtype=int)
    s = np.asarray(scores, dtype=float)
    assert np.isfinite(s[idx]).all()
    tie = ctx.tie if tie is None else tie
    return idx[np.lexsort((tie[idx], -s[idx]))[:TOP_K]]


def rank_pool(ctx: Context, pool, scores) -> np.ndarray:
    return pool[np.lexsort((ctx.tie[pool], -scores))[:TOP_K]]


def lists_from_official(ctx: Context, preds: dict) -> list:
    """Official prediction dict -> eligible job positions per query. No query may be missing."""
    assert set(preds) == set(ctx.gold), "No omitir consultas."
    lists = []
    for key in ctx.keys:
        idx = ctx.job_index.get_indexer(preds[key])
        assert (idx >= 0).all() and len(idx) == len(set(idx)) and len(idx) <= TOP_K
        lists.append(idx[ctx.eligible(key[1])[idx]])
    return lists


def summarize_ranks(ctx: Context, ranks: np.ndarray, mask=None) -> dict:
    """Summary metrics from ranks (np.inf = outside the list). `mask` restricts the queries."""
    mask = np.ones(len(ranks), dtype=bool) if mask is None else mask
    r = ranks[mask]
    rr = 1 / r
    warm = ctx.warm[mask]
    return {"Consultas": len(r), "MRR": rr.mean(),
            "nDCG@10": np.where(r <= 10, 1 / np.log2(r + 1), 0.0).mean(),
            "HR@10": (r <= 10).mean(),
            "MRR warm": rr[warm].mean() if warm.any() else np.nan,
            "MRR cold": rr[~warm].mean() if (~warm).any() else np.nan,
            "Fuera top1000 (%)": 100 * np.isinf(r).mean()}


def detail_from_ranks(ranks: np.ndarray) -> dict:
    return {"puestos": ranks, "RR": 1 / ranks,
            "nDCG": np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.0),
            "HR10": (ranks <= 10).astype(float)}


def metrics_from_lists(ctx: Context, lists: list) -> tuple[dict, dict]:
    assert len(lists) == len(ctx)
    ranks = np.full(len(lists), np.inf)
    for i, lst in enumerate(lists):
        arr = np.asarray(lst, dtype=int)
        assert len(arr) <= TOP_K and len(arr) == len(set(arr))
        assert ctx.eligible(ctx.keys[i][1])[arr].all()
        hit = np.flatnonzero(arr == ctx.gold_idx[i])
        if len(hit):
            ranks[i] = hit[0] + 1
    return summarize_ranks(ctx, ranks), detail_from_ranks(ranks)


def verify_official(ctx: Context, name: str, lists: list, quiet: bool = False) -> tuple[dict, dict]:
    """Metrics from lists, asserted equal to the official evaluator overall, warm and cold."""
    preds = {k: ctx.job_ids[arr] for k, arr in zip(ctx.keys, lists)}
    random.seed(42)
    tables, _, mrr, ns = ctx.evaluator.compute_metrics(ctx.gold, preds, [10], ctx.target_dates, ctx.known_queries)
    summary, detail = metrics_from_lists(ctx, lists)
    assert ns["all"] == len(ctx)
    for group, mask in [("all", np.ones(len(lists), dtype=bool)), ("warm", ctx.warm), ("cold", ~ctx.warm)]:
        assert np.isclose(detail["RR"][mask].mean(), mrr[group], atol=1e-12, rtol=0)
        assert np.isclose(detail["nDCG"][mask].mean(), tables[group].iloc[0]["NDCG@K"], atol=1e-12, rtol=0)
    if not quiet:
        print(name, ": coincide con evaluador oficial en global/warm/cold.")
    return summary, detail


def fast_ranks(gold_idx: np.ndarray, scores: np.ndarray, masks: np.ndarray, tie: np.ndarray) -> np.ndarray:
    """Rank of the relevant job per query, vectorized over (queries x jobs) arrays.

    Equals `rank_scores` followed by a lookup: higher score first, ties by lower `tie`,
    excluded jobs never counted, and anything past position 1000 becomes np.inf.
    """
    rows = np.arange(len(gold_idx))
    gs = scores[rows, gold_idx][:, None]
    gt = tie[gold_idx][:, None]
    ahead = masks & ((scores > gs) | ((scores == gs) & (tie[None, :] < gt)))
    ranks = ahead.sum(axis=1).astype(float) + 1
    ranks[~masks[rows, gold_idx]] = np.inf
    ranks[ranks > TOP_K] = np.inf
    return ranks
