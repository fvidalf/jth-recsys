"""Everything a candidate-day evaluation needs: queries, gold, eligibility, history.

Extracted from scripts/01_datos_y_validacion.ipynb (cells 31, 33 and 49), so that the
validation and test runs share one definition of the task.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

TIE_SEED = 42


def load_official_evaluator(repo: Path) -> ModuleType:
    """Import the official evaluator without writing bytecode into the submodule."""
    sys.dont_write_bytecode = True
    path = Path(repo) / "external" / "JTH" / "code" / "utils" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("jth_official_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Context:
    evaluator: ModuleType
    keys: list  # (candidate id, 'YYYY-MM-DD'), sorted by day
    gold: dict  # official gold: key -> job id
    job_ids: np.ndarray
    job_index: pd.Index
    gold_idx: np.ndarray  # job position of each query's relevant job
    days: np.ndarray  # datetime64[D], one per query
    job_dates: np.ndarray  # datetime64[D], NaT when unknown
    warm: np.ndarray  # official definition: candidate seen in train or earlier in the stream
    history: np.ndarray  # processes of the candidate known before the query
    target_dates: dict
    known_queries: set
    tie: np.ndarray  # fixed random tie-break priority per job

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def n_jobs(self) -> int:
        return len(self.job_ids)

    def eligible(self, day) -> np.ndarray:
        """Jobs created by `day`. Unknown dates stay eligible, as in the official evaluator."""
        return np.isnat(self.job_dates) | (self.job_dates <= np.datetime64(day, "D"))

    def subset_mask(self, first_day=None, last_day=None) -> np.ndarray:
        mask = np.ones(len(self), dtype=bool)
        if first_day is not None:
            mask &= self.days >= np.datetime64(first_day, "D")
        if last_day is not None:
            mask &= self.days <= np.datetime64(last_day, "D")
        return mask


def build_context(repo: Path, train_csv: Path, target_csv: Path, jobs: pd.DataFrame,
                  evaluator: ModuleType | None = None) -> Context:
    ev = evaluator or load_official_evaluator(repo)
    raw = Path(repo) / "data" / "raw"
    gold = ev.load_gold(Path(target_csv), "c2j")
    known = ev.load_initial_known_queries(Path(train_csv), "c2j")
    target_dates = ev.load_dates(raw / "jobs.csv", "job_id")

    job_ids = np.array([ev._parse_id(x) for x in jobs["job_id"]])
    job_index = pd.Index(job_ids)
    assert job_index.is_unique
    keys = sorted(gold, key=lambda k: k[1])
    gold_idx = job_index.get_indexer([gold[k] for k in keys])
    assert (gold_idx >= 0).all()
    days = np.array([pd.Timestamp(k[1]).to_datetime64() for k in keys], dtype="datetime64[D]")
    job_dates = pd.to_datetime(jobs["create_date"], errors="coerce").to_numpy(dtype="datetime64[D]")

    warm, seen = [], set(known)
    for cand, _ in keys:
        warm.append(cand in seen)
        seen.add(cand)

    # History: processes known before the query. Train rows count whole (they are what the
    # official warm definition uses); target rows count once their day has passed.
    train = pd.read_csv(train_csv, dtype=str, usecols=["candidate_id"])
    target = pd.read_csv(target_csv, dtype=str, usecols=["candidate_id", "timestamp"])
    counts = Counter(ev._parse_id(c) for c in train["candidate_id"])
    target_by_day: dict[str, list] = {}
    for c, t in zip(target["candidate_id"], target["timestamp"]):
        target_by_day.setdefault(t, []).append(ev._parse_id(c))
    history = np.zeros(len(keys), dtype=int)
    current = None
    for i, (cand, day) in enumerate(keys):
        if current is not None and day != current:
            counts.update(target_by_day.get(current, []))
        current = day
        history[i] = counts[cand]

    return Context(
        evaluator=ev, keys=keys, gold=gold, job_ids=job_ids, job_index=job_index, gold_idx=gold_idx,
        days=days, job_dates=job_dates, warm=np.array(warm, dtype=bool), history=history,
        target_dates=target_dates, known_queries=known,
        tie=np.random.default_rng(TIE_SEED).random(len(jobs)),
    )
