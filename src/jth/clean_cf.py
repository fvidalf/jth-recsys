"""CF over job-job similarity, without the official catalog's leak or hash-order ties.

The official `hybrid_recency_cf_baseline.py` builds its catalog from training jobs and from
the jobs evaluated in the target period, so future relevant jobs are in the pool. It also
overwrites similarities between a candidate's past jobs and breaks zero-score ties by set
order. Those official numbers stay as a labelled reference. This version sums similarities
over the candidate's past jobs, scores every job, and leaves ties to the caller's fixed
priority. It uses the same cosine as the official model: co-occurrences / sqrt(pop_a * pop_b).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from jth.context import Context

FRESH_DAYS = 90


def clean_cf_scores(ctx: Context, train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """(queries x jobs) summed similarity, using only processes from before each query day."""
    position = {j: i for i, j in enumerate(ctx.job_ids.tolist())}
    parse = ctx.evaluator._parse_id
    query_days = np.unique(ctx.days)

    events = pd.concat([train[["candidate_id", "job_id", "timestamp"]], target[["candidate_id", "job_id", "timestamp"]]])
    event_days = pd.to_datetime(events["timestamp"]).to_numpy(dtype="datetime64[D]")
    event_cands = [parse(c) for c in events["candidate_id"]]
    event_jobs = [position[parse(j)] for j in events["job_id"]]
    order = np.argsort(event_days, kind="stable")

    cand_jobs: dict = defaultdict(set)
    job_pop: dict = defaultdict(int)
    co: dict = defaultdict(lambda: defaultdict(int))

    def add(cand, job):
        if job in cand_jobs[cand]:
            return
        for prev in cand_jobs[cand]:
            co[prev][job] += 1
            co[job][prev] += 1
        cand_jobs[cand].add(job)
        job_pop[job] += 1

    queries_by_day: dict = defaultdict(list)
    for i, day in enumerate(ctx.days):
        queries_by_day[day].append(i)

    scores = np.zeros((len(ctx), ctx.n_jobs))
    cursor = 0
    for day in query_days:
        while cursor < len(order) and event_days[order[cursor]] < day:
            k = order[cursor]
            add(event_cands[k], event_jobs[k])
            cursor += 1
        for i in queries_by_day[day]:
            seen = cand_jobs.get(ctx.keys[i][0], set())
            for prev in seen:
                for other, n in co[prev].items():
                    if other not in seen:
                        scores[i, other] += n / math.sqrt(job_pop[prev] * job_pop[other])
    return scores


def fresh_mask(age: np.ndarray, days: int = FRESH_DAYS) -> np.ndarray:
    """Jobs published in the last `days` days (the official model's 90-day rule)."""
    with np.errstate(invalid="ignore"):
        return (age >= 0) & (age <= days)


def ranked_reference(similarity: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    """Score that orders fresh jobs first, then stale, each by similarity (the official order)."""
    top = similarity.max(axis=1, keepdims=True)
    return fresh.astype(float) + similarity / (1 + np.where(top > 0, top, 1))
