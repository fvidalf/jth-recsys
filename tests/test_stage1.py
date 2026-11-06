"""Stage 1 mechanics, on invented data only."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from jth.activity import StateMatrix, daily_states, hard_mask, soft_penalty
from jth.activity_model import fit_activity_model, snapshots
from jth.clean_cf import clean_cf_scores
from jth.fusion import (best_alpha_per_bin, blend, fit_pairwise, fwls_meta, fwls_pairs, fwls_score, gate_alpha,
                        history_bins, minmax_rows, rrf_pool, sample_negatives)
from jth.ranking import detail_from_ranks, fast_ranks, metrics_from_lists, rank_scores
from jth.uncertainty import BlockBootstrap


def make_ctx(job_dates, keys, gold, warm=None, history=None, seed=42):
    dates = pd.to_datetime(job_dates).to_numpy(dtype="datetime64[D]")
    ids = np.arange(len(job_dates))
    days = np.array([np.datetime64(k[1], "D") for k in keys])
    n = len(keys)
    return SimpleNamespace(
        evaluator=SimpleNamespace(_parse_id=lambda x: int(x) if str(x).isdigit() else x), keys=keys, job_ids=ids,
        job_index=pd.Index(ids), gold_idx=np.array(gold), days=days, job_dates=dates,
        warm=np.zeros(n, dtype=bool) if warm is None else np.array(warm),
        history=np.zeros(n, dtype=int) if history is None else np.array(history),
        tie=np.random.default_rng(seed).random(len(job_dates)), n_jobs=len(job_dates),
        eligible=lambda day: dates <= np.datetime64(day, "D"), __len__=lambda self: n)


class Ctx(SimpleNamespace):
    def __len__(self):
        return len(self.keys)


def ctx_of(*args, **kwargs):
    return Ctx(**vars(make_ctx(*args, **kwargs)))


def frame(rows):
    return pd.DataFrame(rows, columns=["candidate_id", "job_id", "timestamp"]).astype({"job_id": str})


# ---------------------------------------------------------------- activity state
def test_event_of_the_day_only_affects_later_days():
    ctx = ctx_of(["2020-01-01"] * 3, [("a", "2020-01-10"), ("b", "2020-01-11")], [0, 1])
    train = frame([("x", 0, "2020-01-05")])
    target = frame([("a", 0, "2020-01-10"), ("b", 1, "2020-01-11")])
    st = daily_states(ctx, train, target)
    assert st["2020-01-10"]["lag"][0] == 5  # the event of Jan 10 isn't visible on Jan 10
    assert not np.isfinite(st["2020-01-10"]["lag"][1])
    assert st["2020-01-11"]["lag"][0] == 1
    assert not np.isfinite(st["2020-01-11"]["lag"][1])


def test_train_rows_on_the_boundary_day_are_not_visible_that_day():
    ctx = ctx_of(["2020-01-01"] * 2, [("a", "2020-01-10"), ("a", "2020-01-11")], [0, 1])
    train = frame([("x", 0, "2020-01-05"), ("y", 1, "2020-01-10")])
    target = frame([("a", 0, "2020-01-10"), ("a", 1, "2020-01-11")])
    st = daily_states(ctx, train, target)
    assert not np.isfinite(st["2020-01-10"]["lag"][1])
    assert st["2020-01-11"]["lag"][1] == 1


def test_hard_filter_grace_and_future_jobs():
    state = {"age": np.array([100.0, 2.0, 100.0, -1.0]), "lag": np.array([2.0, np.inf, np.inf, np.inf]),
             "eligible": np.array([True, True, True, False])}
    assert hard_mask(state, 7, 0).tolist() == [True, False, False, False]
    assert hard_mask(state, 7, 7).tolist() == [True, True, False, False]


def test_soft_penalty_depends_only_on_lambda_over_tau():
    state = {"effective_lag": np.array([0.0, 10.0, 30.0])}
    assert np.allclose(soft_penalty(state, 30, 0.5), soft_penalty(state, 60, 1.0))
    assert soft_penalty(state, 30, 0.0).tolist() == [0.0, 0.0, 0.0]


def test_state_matrix_matches_the_daily_states():
    ctx = ctx_of(["2020-01-01", "2020-01-09", "2020-01-20"],
                 [("a", "2020-01-10"), ("b", "2020-01-12")], [0, 1])
    st = daily_states(ctx, frame([("x", 0, "2020-01-05")]), frame([("a", 0, "2020-01-10"), ("b", 1, "2020-01-12")]))
    m = StateMatrix(ctx, st)
    for i, key in enumerate(ctx.keys):
        assert np.array_equal(m.hard(7, 7)[i], hard_mask(st[key[1]], 7, 7))
        assert np.array_equal(m.penalty(30, 0.5)[i], soft_penalty(st[key[1]], 30, 0.5))


# ---------------------------------------------------------------- ranking
def test_fast_ranks_equal_the_exact_path():
    rng = np.random.default_rng(7)
    n_jobs, n_q = 1500, 40
    keys = [(f"c{i}", "2020-01-10") for i in range(n_q)]
    gold = rng.integers(0, n_jobs, n_q)
    ctx = ctx_of(["2020-01-01"] * n_jobs, keys, gold)
    scores = np.round(rng.random((n_q, n_jobs)), 2)  # many ties on purpose
    masks = rng.random((n_q, n_jobs)) < 0.9
    lists = [rank_scores(ctx, scores[i], masks[i]) for i in range(n_q)]
    exact = np.full(n_q, np.inf)
    for i, lst in enumerate(lists):
        hit = np.flatnonzero(lst == gold[i])
        if len(hit):
            exact[i] = hit[0] + 1
    assert np.array_equal(fast_ranks(ctx.gold_idx, scores, masks, ctx.tie), exact)
    assert np.isinf(exact).any() and np.isfinite(exact).any()


def test_metrics_of_known_ranks():
    ctx = ctx_of(["2020-01-01"] * 12, [("a", "2020-01-10"), ("b", "2020-01-10")], [2, 11], warm=[True, False])
    summary, detail = metrics_from_lists(ctx, [np.array([0, 1, 2]), np.array([0])])
    assert np.isclose(summary["MRR"], (1 / 3 + 0) / 2)
    assert np.isclose(summary["nDCG@10"], 0.25)
    assert summary["HR@10"] == 0.5
    assert np.isclose(summary["MRR warm"], 1 / 3) and summary["MRR cold"] == 0
    assert summary["Fuera top1000 (%)"] == 50


# ---------------------------------------------------------------- fusion
def test_rrf_extremes_return_each_list_unchanged():
    a, b = np.array([3, 1]), np.array([2, 1])
    assert rrf_pool(a, b, 1.0, 5)[0].tolist() == [3, 1]
    assert rrf_pool(a, b, 0.0, 5)[0].tolist() == [2, 1]
    pool, s = rrf_pool(a, b, 0.5, 5)
    assert pool.tolist() == [1, 2, 3]
    assert np.isclose(s[0], 0.5 / 62 + 0.5 / 62)  # second in both lists


def test_minmax_constant_rows_and_range():
    scores = np.array([[2.0, 4.0, 3.0, 100.0], [0.0, 0.0, 0.0, 0.0]])
    masks = np.array([[True, True, True, False], [True, True, True, True]])
    norm = minmax_rows(scores, masks)
    assert np.isclose(norm[0, 0], 1e-6) and norm[0, 1] == 1
    assert np.isclose(norm[0, 2], 1e-6 + (1 - 1e-6) * 0.5)
    assert np.all(norm[1] == norm[1, 0])  # a constant row can't change any order
    assert np.isfinite(np.log(norm)).all()


def test_blend_extremes_and_constant_second_model_keep_the_first_order():
    a = np.array([[0.2, 0.9, 0.5]])
    b = np.array([[0.1, 0.1, 0.1]])
    assert np.array_equal(blend(a, b, 1.0), a)
    assert np.array_equal(blend(a, b, 0.0), b)
    assert np.argsort(-blend(a, b, 0.4)[0]).tolist() == np.argsort(-a[0]).tolist()


def test_gate_is_a_per_query_weight():
    assert gate_alpha(np.array([True, False]), 0.5).tolist() == [0.5, 1.0]
    both = blend(np.ones((2, 1)), np.zeros((2, 1)), gate_alpha(np.array([True, False]), 0.5))
    assert both.ravel().tolist() == [0.5, 1.0]


def test_bins_chosen_separately_equal_the_joint_search():
    rng = np.random.default_rng(3)
    bins = history_bins(rng.integers(0, 5, 300))
    rr = {a: rng.random(300) for a in (0.0, 0.5, 1.0)}
    alphas, chosen = best_alpha_per_bin(rr, bins)
    per_query = np.array([rr[a][i] for i, a in enumerate(alphas)]).mean()
    best_joint = -1.0
    for a0 in rr:
        for a1 in rr:
            for a2 in rr:
                pick = {0: a0, 1: a1, 2: a2}
                best_joint = max(best_joint, np.mean([rr[pick[int(b)]][i] for i, b in enumerate(bins)]))
    assert np.isclose(per_query, best_joint)
    assert set(chosen) == {0, 1, 2}


def test_history_acts_only_through_its_product_with_an_input():
    meta = fwls_meta(np.array([True, False]), np.array([3, 0]))
    assert meta.shape == (2, 3)
    a, b = np.array([[0.1, 0.9], [0.1, 0.9]]), np.array([[0.8, 0.2], [0.8, 0.2]])
    # First input always counts. The second one counts only when the candidate has history.
    w = np.array([1.0, 0.0, 0.0, 0.0, 3.0, 0.0])
    s = fwls_score([a, b], meta, w)
    assert s[0].argmax() == 0  # with history: second input dominates
    assert s[1].argmax() == 1  # without: first input only


def test_pairs_match_the_score_difference():
    rng = np.random.default_rng(0)
    a, b = rng.random((4, 6)), rng.random((4, 6))
    meta = fwls_meta(np.array([True, False, True, False]), np.array([1, 0, 4, 0]))
    gold = np.array([0, 1, 2, 3])
    negatives = [np.array([3, 4]), np.array([0]), np.array([5]), np.array([1, 2])]
    pairs = fwls_pairs([a, b], meta, gold, negatives, np.ones(4, dtype=bool))
    w = rng.random(6)
    s = fwls_score([a, b], meta, w)
    expected = np.concatenate([s[i, gold[i]] - s[i, negatives[i]] for i in range(4)])
    assert np.allclose(pairs @ w, expected)


def test_negatives_exclude_the_relevant_job_and_stay_eligible():
    masks = np.array([[True, True, True, True, False]])
    negs = sample_negatives(np.array([0]), masks, np.array([[9.0, 5.0, 4.0, 3.0, 8.0]]), 1, 1)
    assert 0 not in negs[0] and 4 not in negs[0] and 1 in negs[0]


def test_pairwise_fit_recovers_the_useful_input():
    rng = np.random.default_rng(0)
    q, j = 60, 30
    useful, noise = rng.random((q, j)), rng.random((q, j))
    gold = useful.argmax(axis=1)
    negatives = [np.setdiff1d(np.arange(j), [g]) for g in gold]
    meta = np.ones((q, 1))
    pairs = fwls_pairs([useful, noise], meta, gold, negatives, np.ones(q, dtype=bool))
    w = fit_pairwise(pairs, c=1.0)
    assert w[0] > 0 and w[0] > abs(w[1])


# ---------------------------------------------------------------- activity model
def test_snapshots_use_the_past_and_censor_the_end():
    days = np.array([0, 10, 12, 95, 100])
    jobs = np.array([0, 0, 0, 1, 0])
    dates = np.array(["1970-01-01", "1970-01-01"], dtype="datetime64[D]")
    x, y = snapshots(days, jobs, dates, horizon=30, cutoff_day=101, every=7)
    # Snapshots run on days 7, 14, ... 70 (last day with a full label window), two jobs each.
    assert len(y) == 2 * 10
    assert y[0] and not y[1]  # day 7: job 0 has events on days 10 and 12, job 1 has none


def test_activity_model_prefers_recently_active_jobs():
    rng = np.random.default_rng(1)
    days = np.sort(rng.integers(0, 400, 3000))
    jobs = rng.integers(0, 20, 3000)
    dates = np.array(["1970-01-01"] * 20, dtype="datetime64[D]")
    x, y = snapshots(days, jobs, dates, horizon=14, cutoff_day=400)
    model = fit_activity_model(x, y)
    assert 0 < model.predict_proba(x)[:, 1].mean() < 1


# ---------------------------------------------------------------- clean CF
def test_clean_cf_sums_similarity_and_keeps_jobs_out_until_they_exist():
    # Jobs 0 and 1 co-occur through candidate x. Candidate a saw job 0.
    ctx = ctx_of(["2020-01-01"] * 4, [("a", "2020-01-10"), ("n", "2020-01-10"), ("a", "2020-01-11")], [1, 2, 2])
    train = frame([("x", 0, "2020-01-01"), ("x", 1, "2020-01-02"), ("a", 0, "2020-01-03")])
    target = frame([("a", 1, "2020-01-10"), ("n", 2, "2020-01-10"), ("a", 2, "2020-01-11")])
    s = clean_cf_scores(ctx, train, target)
    assert s.shape == (3, 4)
    assert s[0, 1] > 0 and s[0, 0] == 0  # the seen job is excluded
    assert np.all(s[1] == 0)  # new candidate: CF has nothing to say
    assert s[2, 1] == 0  # job 1 was seen on Jan 10, so it's excluded on Jan 11
    # Job 2 co-occurs with nothing before Jan 11, so a day-10 process can't reveal it early.
    assert s[0, 2] == 0


# ---------------------------------------------------------------- systems
def small_inputs():
    from scipy import sparse
    from jth.systems import Inputs
    rng = np.random.default_rng(5)
    n_jobs, n_q = 1200, 30
    keys = [(f"c{i}", f"2020-02-{1 + i % 5:02d}") for i in range(n_q)]
    keys.sort(key=lambda k: k[1])
    ctx = ctx_of(["2020-01-01"] * n_jobs, keys, rng.integers(0, n_jobs, n_q), warm=rng.random(n_q) < 0.4,
                 history=rng.integers(0, 3, n_q))
    ctx.warm = ctx.history > 0
    train = frame([("x", int(j), "2020-01-15") for j in rng.integers(0, n_jobs, 200)])
    target = frame([(k[0], int(g), k[1]) for k, g in zip(keys, ctx.gold_idx)])
    states = daily_states(ctx, train, target)
    lr = rng.random((n_q, n_jobs))
    sim = sparse.csr_matrix(rng.random((n_q, n_jobs)) * (rng.random((n_q, n_jobs)) < 0.05))
    return ctx, Inputs(ctx, states, lr, np.arange(n_q), sim)


def test_fast_ranks_equal_the_lists_for_every_scorer():
    from jth.systems import (hard, lr_score, ranks_from_lists, score_blend, score_blend_after,
                             score_blend_before, soft_lr, soft_normalized, cf_input)
    ctx, inp = small_inputs()
    for scorer in [hard(lr_score, 14, 7), soft_lr(30, 0.5), soft_normalized(cf_input, 30, 0.5),
                   score_blend(0.5, gate=True), score_blend_after(0.5, True, 30, 0.5),
                   score_blend_before(0.5, False, 30, 0.5)]:
        assert np.array_equal(inp.ranks(scorer), ranks_from_lists(ctx, inp.lists(scorer)))


def test_blend_with_gate_equals_lr_for_new_candidates():
    from jth.systems import lr_score, plain, score_blend
    ctx, inp = small_inputs()
    base = inp.ranks(plain(lr_score))
    gated = inp.ranks(score_blend(0.25, gate=True))
    assert np.array_equal(base[~ctx.warm], gated[~ctx.warm])


def test_new_candidate_without_cf_signal_needs_no_gate():
    from scipy import sparse
    from jth.systems import Inputs, lr_score, plain, score_blend
    ctx, inp = small_inputs()
    sim = inp.similarity.tolil()
    sim[np.flatnonzero(~ctx.warm)] = 0
    no_signal = Inputs(ctx, inp.states, inp.lr, inp.lr_row, sparse.csr_matrix(sim))
    base = no_signal.ranks(plain(lr_score))
    free = no_signal.ranks(score_blend(0.5, gate=False))
    assert np.array_equal(base[~ctx.warm], free[~ctx.warm])


def test_fwls_fit_and_score_run_on_chunks():
    from jth.systems import fit_fwls, fwls_scorer
    ctx, inp = small_inputs()
    w = fit_fwls(inp, np.ones(len(ctx), dtype=bool), True, c=1.0, n_uniform=10, n_hard=5)
    assert w.shape == (9,)
    assert len(inp.ranks(fwls_scorer(w, True))) == len(ctx)


# ---------------------------------------------------------------- moved from the notebook
def test_tie_break_and_excluded_job():
    ctx = ctx_of(["2020-01-01"] * 4, [("a", "2020-01-10")], [0])
    ctx.tie = np.array([0.8, 0.2, 0.5, 0.1])
    order = rank_scores(ctx, np.array([0.7, 0.7, 0.0, 9.0]), np.array([True, True, True, False]))
    assert order.tolist() == [1, 0, 2]  # tie broken by priority, excluded job left out despite its score


def test_metric_contribution_of_each_position():
    d = detail_from_ranks(np.array([1.0, 3.0, 10.0, 11.0, np.inf]))
    assert np.isclose(d["RR"][1], 1 / 3) and d["RR"][4] == 0
    assert d["nDCG"][1] == 0.5 and d["nDCG"][3] == 0
    assert d["HR10"].tolist() == [1, 1, 1, 0, 0]


def official_evaluator():
    import pytest
    from pathlib import Path
    from jth.context import load_official_evaluator
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "external/JTH/code/utils/evaluator.py").exists():
        pytest.skip("submodule not initialized")
    return load_official_evaluator(repo)


def test_official_evaluator_on_invented_cases():
    import random
    ev = official_evaluator()
    gold = {("A", "2020-01-10"): "oferta_A", ("A", "2020-01-11"): "oferta_B", ("B", "2020-01-11"): "oferta_C"}
    preds = {("A", "2020-01-10"): ["oferta_futura", "oferta_C", "oferta_B", "oferta_A"], ("A", "2020-01-11"): ["oferta_B"]}
    dates = {"oferta_A": "2020-01-01", "oferta_B": "2020-01-05", "oferta_C": "2020-01-09", "oferta_futura": "2020-01-12"}
    random.seed(42)
    tables, _, mrr, n = ev.compute_metrics(gold, preds, [10], dates, set())
    # Third after the future job is dropped, first, and a query without predictions that still counts.
    assert n == {"all": 3, "warm": 1, "cold": 2}  # A moves from new to known on its second day
    assert np.isclose(mrr["all"], (1 / 3 + 1 + 0) / 3)
    assert np.isclose(tables["all"].iloc[0]["NDCG@K"], 0.5)


def test_official_reader_accepts_our_prediction_format(tmp_path):
    import gzip
    import json
    import pickle
    ev = official_evaluator()
    path = tmp_path / "toy_c2j.pkl.gz"
    with gzip.open(path, "wb") as f:
        pickle.dump({json.dumps(["candidato_demo", "2020-01-10"]): ["oferta_A", "oferta_B"]}, f)
    assert ev.load_preds(path)[("candidato_demo", "2020-01-10")].tolist() == ["oferta_A", "oferta_B"]


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_is_paired_and_covers_a_known_mean():
    days = np.datetime64("2020-01-01") + np.arange(140).astype("timedelta64[D]")
    boot = BlockBootstrap(days, n=500)
    a = np.random.default_rng(0).random(140)
    mean, low, high = boot.interval(a)
    assert low < mean < high
    assert boot.difference(a, a) == (0.0, 0.0, 0.0)


def test_detail_handles_missing_ranks():
    d = detail_from_ranks(np.array([1.0, np.inf]))
    assert d["RR"].tolist() == [1.0, 0.0] and d["HR10"].tolist() == [1.0, 0.0]


def test_normalized_activity_model_weighs_log_activity():
    from types import SimpleNamespace

    from jth import systems as sis

    view = SimpleNamespace(day=np.array([0]), eligible=np.array([[True, True, False]]))
    flat = lambda _: np.array([[2.0, 2.0, 2.0]])  # noqa: E731
    rising = lambda _: np.array([[0.0, 1.0, 9.0]])  # noqa: E731
    log_activity = np.log(np.array([[0.9, 0.1, 0.5]]))

    scores, mask = sis.normalized_activity_model(flat, log_activity, 0.5)(view)
    assert mask.tolist() == [[True, True, False]]
    assert scores[0, 0] > scores[0, 1]  # a constant base leaves only the activity term

    scores, _ = sis.normalized_activity_model(rising, log_activity, 0.0)(view)
    assert scores[0, 1] > scores[0, 0]  # lambda 0 leaves only the base
