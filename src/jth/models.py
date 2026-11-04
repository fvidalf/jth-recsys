"""Baseline recommenders.

Every model works in index space (see ``evaluation.Context``) and in either
direction: "query" is the candidate in c2j and the job in j2c.

Interface used by ``evaluation.evaluate``:
    fit(q, t, day, ctx)   train interactions as index arrays
    score(queries, now)   (len(queries), n_targets) array, higher = better
    update(q, t, day)     interactions revealed after predicting day ``day``
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from jth.evaluation import Context
from jth.features import profile_tokens


class Recommender:
    name = "base"

    def fit(self, q: np.ndarray, t: np.ndarray, day: np.ndarray, ctx: Context) -> None:
        self.ctx = ctx

    def update(self, q: np.ndarray, t: np.ndarray, day: int) -> None:
        pass

    def score(self, queries: np.ndarray, now: int) -> np.ndarray:
        raise NotImplementedError


def _impute_mean(days: np.ndarray) -> np.ndarray:
    """Missing creation days -> mean of known ones (as the official baselines)."""
    out = days.copy()
    out[np.isnan(out)] = np.nanmean(days)
    return out


class Random(Recommender):
    name = "Random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, q, t, day, ctx):
        super().fit(q, t, day, ctx)
        self.rng = np.random.default_rng(self.seed)

    def score(self, queries, now):
        return self.rng.random((len(queries), self.ctx.n_targets))


class MostPop(Recommender):
    """Number of interactions each target has received so far."""

    name = "MostPop"

    def fit(self, q, t, day, ctx):
        super().fit(q, t, day, ctx)
        self.counts = np.bincount(t, minlength=ctx.n_targets).astype(float)

    def update(self, q, t, day):
        np.add.at(self.counts, t, 1.0)

    def score(self, queries, now):
        return np.broadcast_to(self.counts, (len(queries), self.ctx.n_targets))


class Recency(Recommender):
    """Newest targets first ("Past temporal" in the JTH paper).

    Targets not created yet are dropped by the evaluator, so ranking by
    creation date = ranking by age among what is available at ``now``.
    """

    name = "Recency"

    def fit(self, q, t, day, ctx):
        super().fit(q, t, day, ctx)
        self.created = _impute_mean(ctx.target_created)

    def score(self, queries, now):
        return np.broadcast_to(self.created, (len(queries), self.ctx.n_targets))


class PopRecency(MostPop):
    """popularity(t) * exp(-lam * age_days(t)) ("Pop. + recency" in the paper)."""

    name = "PopRecency"

    def __init__(self, lam: float = 0.015):
        self.lam = lam

    def fit(self, q, t, day, ctx):
        super().fit(q, t, day, ctx)
        self.created = _impute_mean(ctx.target_created)

    def score(self, queries, now):
        age = np.maximum(0.0, now - self.created)
        s = self.counts * np.exp(-self.lam * age)
        return np.broadcast_to(s, (len(queries), self.ctx.n_targets))


class ItemKNN(Recommender):
    """Item-based CF with cosine similarity on binary co-occurrence.

    score(q, t) = sum over targets t' the query interacted with of cos(t, t').
    The similarity matrix is rebuilt lazily after each update. Queries with no
    history (cold) get all-zero scores, i.e. a random ranking.
    """

    name = "ItemKNN"

    def fit(self, q, t, day, ctx):
        super().fit(q, t, day, ctx)
        self._q, self._t = [q], [t]
        self._dirty = True

    def update(self, q, t, day):
        self._q.append(q)
        self._t.append(t)
        self._dirty = True

    def _rebuild(self):
        q, t = np.concatenate(self._q), np.concatenate(self._t)
        X = sp.csr_matrix(
            (np.ones(len(q)), (q, t)), shape=(self.ctx.n_queries, self.ctx.n_targets)
        )
        X.data[:] = 1.0  # repeated pairs count once
        co = (X.T @ X).tocsr()
        pop = np.asarray(X.sum(axis=0)).ravel()
        inv_sqrt = np.divide(1.0, np.sqrt(pop), out=np.zeros_like(pop), where=pop > 0)
        sim = (sp.diags(inv_sqrt) @ co @ sp.diags(inv_sqrt)).tocsr()
        sim = (sim - sp.diags(sim.diagonal())).tocsr()  # no self-similarity
        sim.eliminate_zeros()
        self.X, self.sim = X, sim
        self._dirty = False

    def score(self, queries, now):
        if self._dirty:
            self._rebuild()
        return (self.X[queries] @ self.sim).toarray()


class ContentTFIDF(Recommender):
    """Cosine similarity between TF-IDF bags of skills / areas / categories.

    Uses only profile attributes, so it can score cold queries and cold
    targets, but it ignores time entirely (expired jobs rank as high as
    open ones).
    """

    name = "ContentTFIDF"

    def fit(self, q, t, day, ctx):
        super().fit(q, t, day, ctx)
        q_docs = profile_tokens(ctx.query_table)
        t_docs = profile_tokens(ctx.target_table)
        vec = TfidfVectorizer(analyzer=_identity, sublinear_tf=True)
        vec.fit(q_docs + t_docs)
        self.Q = vec.transform(q_docs)  # rows are L2-normalised
        self.T = vec.transform(t_docs).T.tocsr()

    def score(self, queries, now):
        return (self.Q[queries] @ self.T).toarray()


def _identity(doc):
    return doc


BASELINES = {
    "random": Random,
    "mostpop": MostPop,
    "recency": Recency,
    "poprecency": PopRecency,
    "itemknn": ItemKNN,
    "content": ContentTFIDF,
}
