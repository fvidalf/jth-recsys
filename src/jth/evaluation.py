"""Streaming (time-ordered) evaluation, following the JTH paper's protocol.

Protocol
--------
* A *query* is a (query entity, day) pair from the test split. In ``c2j`` the
  query is a candidate and the targets are jobs; ``j2c`` is the reverse.
* Days are processed in chronological order. On day ``d`` the model scores
  every target for each query, then the interactions of day ``d`` are
  revealed to it (``model.update``). So when predicting at ``d`` the model
  knows everything strictly before ``d`` and nothing after.
* Targets created after ``d`` are not in the catalog yet and are removed
  from the ranking (same filter as the official evaluator). Targets with an
  unknown creation date stay in.
* Ties in the score are broken uniformly at random (seeded), so constant or
  sparse scores get chance-level ranks instead of id-order artifacts.
* The relevant set of a query is every target it interacted with that day
  (almost always one). A query is *warm* if its entity appears in an earlier
  interaction (train or a previous test day), otherwise *cold*.

Differences from the official ``evaluator.py``: ranks are computed over the
full catalog (no top-1000 cut), and same-day duplicates form one query with
several relevant targets instead of overwriting each other.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from jth.data import to_days

if TYPE_CHECKING:
    from jth.models import Recommender

DEFAULT_KS = (1, 5, 10, 20)


@dataclass
class Context:
    """Index space for one recommendation direction."""

    mode: str
    query_col: str
    target_col: str
    query_table: pd.DataFrame  # profiles, row i <-> query index i
    target_table: pd.DataFrame
    query_created: np.ndarray  # creation day per index (NaN if unknown)
    target_created: np.ndarray

    @property
    def n_queries(self) -> int:
        return len(self.query_table)

    @property
    def n_targets(self) -> int:
        return len(self.target_table)

    def encode(self, inter: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interactions -> (query index, target index, day) arrays."""
        q = pd.Index(self.query_table[self.query_col]).get_indexer(inter[self.query_col])
        t = pd.Index(self.target_table[self.target_col]).get_indexer(inter[self.target_col])
        if (q < 0).any() or (t < 0).any():
            raise KeyError("interaction ids missing from the entity tables")
        return q, t, to_days(inter["timestamp"]).astype(np.int64)


def make_context(cands: pd.DataFrame, jobs: pd.DataFrame, mode: str) -> Context:
    if mode == "c2j":
        q_tab, t_tab, q_col, t_col = cands, jobs, "candidate_id", "job_id"
    elif mode == "j2c":
        q_tab, t_tab, q_col, t_col = jobs, cands, "job_id", "candidate_id"
    else:
        raise ValueError(f"unknown mode {mode!r}")
    q_tab = q_tab.reset_index(drop=True)
    t_tab = t_tab.reset_index(drop=True)
    return Context(
        mode=mode,
        query_col=q_col,
        target_col=t_col,
        query_table=q_tab,
        target_table=t_tab,
        query_created=to_days(q_tab["create_date"]),
        target_created=to_days(t_tab["create_date"]),
    )


def query_metrics(ranks: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    """Metrics for one query given the (1-based) ranks of its relevant targets.

    A relevant target that cannot be ranked (not in the catalog) has rank inf.
    """
    ranks = np.sort(np.asarray(ranks, dtype=float))
    n_rel = len(ranks)
    out = {"MRR": 1.0 / ranks[0]}
    for k in ks:
        hit_ranks = ranks[ranks <= k]
        h = len(hit_ranks)
        ideal = min(n_rel, k)
        out[f"P@{k}"] = h / k
        out[f"R@{k}"] = h / n_rel
        out[f"HR@{k}"] = float(h > 0)
        dcg = np.sum(1.0 / np.log2(hit_ranks + 1))
        idcg = np.sum(1.0 / np.log2(np.arange(2, ideal + 2)))
        out[f"NDCG@{k}"] = dcg / idcg
        out[f"MAP@{k}"] = np.sum(np.arange(1, h + 1) / hit_ranks) / ideal
    return out


def evaluate(
    model: Recommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    ctx: Context,
    ks: Sequence[int] = DEFAULT_KS,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit on ``train``, stream through ``test``; one row per query."""
    rng = np.random.default_rng(seed)
    q_tr, t_tr, d_tr = ctx.encode(train)
    model.fit(q_tr, t_tr, d_tr, ctx)

    seen_q = np.zeros(ctx.n_queries, dtype=bool)
    seen_t = np.zeros(ctx.n_targets, dtype=bool)
    seen_q[q_tr] = True
    seen_t[t_tr] = True

    q_te, t_te, d_te = ctx.encode(test)
    records = []
    for day in np.unique(d_te):
        sel = d_te == day
        qs, ts = q_te[sel], t_te[sel]
        # Unique (query, target) pairs of the day; queries = unique query ids.
        pairs = np.unique(np.stack([qs, ts], axis=1), axis=0)
        uq, inv = np.unique(pairs[:, 0], return_inverse=True)

        scores = np.asarray(model.score(uq, int(day)), dtype=float)
        if scores.shape != (len(uq), ctx.n_targets):
            raise ValueError(f"{model.name}: score shape {scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError(f"{model.name}: non-finite scores")

        # NaN creation date compares False -> stays eligible.
        eligible = ~(ctx.target_created > day)
        tiebreak = rng.random(scores.shape)

        row_s, row_tb = scores[inv], tiebreak[inv]
        s_rel = row_s[np.arange(len(pairs)), pairs[:, 1]][:, None]
        tb_rel = row_tb[np.arange(len(pairs)), pairs[:, 1]][:, None]
        ahead = (row_s > s_rel) | ((row_s == s_rel) & (row_tb < tb_rel))
        ranks = 1.0 + (ahead & eligible).sum(axis=1)
        ranks[~eligible[pairs[:, 1]]] = np.inf

        for i, q in enumerate(uq):
            mask = inv == i
            rel_t = pairs[mask, 1]
            rec = {
                "day": int(day),
                "query": int(q),
                "n_rel": len(rel_t),
                "query_warm": bool(seen_q[q]),
                "target_warm": bool(seen_t[rel_t].any()),
                "first_rank": float(ranks[mask].min()),
            }
            rec.update(query_metrics(ranks[mask], ks))
            records.append(rec)

        seen_q[qs] = True
        seen_t[ts] = True
        model.update(qs, ts, int(day))

    return pd.DataFrame.from_records(records)


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics over all / warm / cold queries."""
    metric_cols = [c for c in per_query.columns if c == "MRR" or "@" in c]
    groups = {
        "all": per_query,
        "warm": per_query[per_query["query_warm"]],
        "cold": per_query[~per_query["query_warm"]],
    }
    rows = []
    for name, g in groups.items():
        row = {"group": name, "n_queries": len(g)}
        row.update(g[metric_cols].mean().to_dict())
        rows.append(row)
    return pd.DataFrame(rows)
