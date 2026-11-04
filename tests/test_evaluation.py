import math

import numpy as np
import pandas as pd
import pytest

from jth.evaluation import evaluate, make_context, query_metrics, summarize
from jth.models import BASELINES, Recommender


def test_single_relevant_matches_official_formulas():
    m = query_metrics(np.array([3.0]), ks=(1, 5))
    assert m["MRR"] == pytest.approx(1 / 3)
    assert m["HR@1"] == 0 and m["P@1"] == 0
    assert m["P@5"] == pytest.approx(1 / 5)
    assert m["R@5"] == 1
    assert m["NDCG@5"] == pytest.approx(1 / math.log2(4))
    assert m["MAP@5"] == pytest.approx(1 / 3)


def test_multiple_relevant():
    m = query_metrics(np.array([3.0, 1.0]), ks=(5,))
    assert m["MRR"] == 1
    assert m["P@5"] == pytest.approx(2 / 5)
    assert m["R@5"] == 1
    assert m["MAP@5"] == pytest.approx((1 / 1 + 2 / 3) / 2)
    ideal = 1 + 1 / math.log2(3)
    assert m["NDCG@5"] == pytest.approx((1 + 1 / math.log2(4)) / ideal)


def test_miss_is_zero():
    m = query_metrics(np.array([np.inf]), ks=(1, 10))
    assert all(v == 0 for v in m.values())


def toy_data():
    cands = pd.DataFrame({
        "candidate_id": [f"c{i}" for i in range(4)],
        "create_date": pd.to_datetime(["2020-01-01"] * 4),
        "skills": ["python", "java", "python;sql", None],
    })
    jobs = pd.DataFrame({
        "job_id": [f"j{i}" for i in range(5)],
        "create_date": pd.to_datetime(
            ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01", "2020-03-01"]
        ),
        "skills": ["python", "java", "sql", "go", "python"],
    })
    inter = pd.DataFrame({
        "candidate_id": ["c0", "c1", "c2", "c0", "c3", "c1"],
        "job_id": ["j0", "j1", "j2", "j1", "j2", "j0"],
        "timestamp": pd.to_datetime(
            ["2020-01-05", "2020-01-06", "2020-01-07", "2020-02-01", "2020-02-02", "2020-02-02"]
        ),
    })
    return cands, jobs, inter.iloc[:3], inter.iloc[3:]


class Memorizer(Recommender):
    """Scores a target 1 if the query interacted with it in revealed data."""

    name = "Memorizer"

    def fit(self, q, t, day, ctx):
        super().fit(q, t, day, ctx)
        self.seen = set(zip(q.tolist(), t.tolist()))

    def update(self, q, t, day):
        self.seen |= set(zip(q.tolist(), t.tolist()))

    def score(self, queries, now):
        s = np.zeros((len(queries), self.ctx.n_targets))
        for i, qq in enumerate(queries):
            for tt in range(self.ctx.n_targets):
                s[i, tt] = (qq, tt) in self.seen
        return s


class Oracle(Recommender):
    name = "Oracle"

    def __init__(self, test_pairs):
        self.test_pairs = test_pairs

    def score(self, queries, now):
        s = np.zeros((len(queries), self.ctx.n_targets))
        for i, qq in enumerate(queries):
            for tt in self.test_pairs.get(qq, []):
                s[i, tt] = 1.0
        return s


def test_oracle_is_perfect_and_flags_are_right():
    cands, jobs, train, test = toy_data()
    ctx = make_context(cands, jobs, "c2j")
    q, t, _ = ctx.encode(test)
    pairs = {}
    for qq, tt in zip(q, t):
        pairs.setdefault(qq, []).append(tt)
    res = evaluate(Oracle(pairs), train, test, ctx, ks=(1,))
    assert (res["first_rank"] == 1).all()
    # c0 and c1 appear in train -> warm; c3 is new -> cold
    warm = dict(zip(res["query"], res["query_warm"]))
    assert warm == {0: True, 1: True, 3: False}


def test_no_same_day_leakage():
    cands, jobs, train, test = toy_data()
    ctx = make_context(cands, jobs, "c2j")

    class Spy(Recommender):
        name = "Spy"

        def fit(self, q, t, day, ctx):
            super().fit(q, t, day, ctx)
            self.latest_seen, self.calls = int(day.max()), []

        def update(self, q, t, day):
            self.latest_seen = max(self.latest_seen, day)

        def score(self, queries, now):
            self.calls.append((self.latest_seen, now))
            return np.zeros((len(queries), self.ctx.n_targets))

    spy = Spy()
    evaluate(spy, train, test, ctx, ks=(1,))
    assert spy.calls and all(seen < now for seen, now in spy.calls)


def test_revealed_interactions_are_used():
    cands, jobs, train, _ = toy_data()
    ctx = make_context(cands, jobs, "c2j")
    # the same pair again a month later: must be memorized by then
    test = pd.DataFrame({
        "candidate_id": ["c3", "c3"],
        "job_id": ["j3", "j3"],
        "timestamp": pd.to_datetime(["2020-02-01", "2020-03-01"]),
    })
    res = evaluate(Memorizer(), train, test, ctx, ks=(1,))
    assert res.sort_values("day")["first_rank"].iloc[-1] == 1


def test_future_targets_are_not_ranked():
    cands, jobs, train, test = toy_data()
    ctx = make_context(cands, jobs, "c2j")

    class FutureLover(Recommender):
        name = "FutureLover"

        def score(self, queries, now):
            s = np.zeros((len(queries), self.ctx.n_targets))
            s[:, 4] = 100.0  # j4 only exists from 2020-03-01 on
            return s

    res = evaluate(FutureLover(), train, test, ctx, ks=(1,))
    # j4 is filtered out, so it never pushes the gold down: 4 eligible jobs,
    # rank of gold is at most 4.
    assert res["first_rank"].max() <= 4


@pytest.mark.parametrize("mode", ["c2j", "j2c"])
@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_run(name, mode):
    cands, jobs, train, test = toy_data()
    ctx = make_context(cands, jobs, mode)
    res = evaluate(BASELINES[name](), train, test, ctx, ks=(1, 5))
    summ = summarize(res)
    assert summ.loc[0, "n_queries"] == len(res) > 0
    assert ((res["MRR"] >= 0) & (res["MRR"] <= 1)).all()
