"""Loading the raw JTH files and building the temporal splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# Funnel stages in chronological order (columns of history.csv).
STAGE_COLS = [
    "spontaneous_application_date",
    "shortlist_date",
    "qualification_date",
    "resume_sent_to_company_date",
    "1st_interview_date",
    "2nd_interview_date",
    "3rd_interview_date",
    "4th_interview_date",
    "job_offer_proposed_date",
    "job_offer_accepted_date",
]
DATE_COLS = STAGE_COLS + ["end_of_process_date"]


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (candidates, jobs, history) with ids as str and dates parsed."""
    cands = pd.read_csv(raw_dir / "candidates.csv", dtype=str)
    jobs = pd.read_csv(raw_dir / "jobs.csv", dtype=str)
    hist = pd.read_csv(raw_dir / "history.csv", dtype=str)
    for df in (cands, jobs):
        df["create_date"] = pd.to_datetime(df["create_date"], errors="coerce")
    for col in DATE_COLS:
        hist[col] = pd.to_datetime(hist[col], errors="coerce")
    return cands, jobs, hist


def build_interactions(hist: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate-job process, stamped with its first recorded event.

    Same definition as the official ``binary_splitter.py``: the interaction
    time is the earliest date over all funnel columns.
    """
    inter = pd.DataFrame({
        "candidate_id": hist["candidate_id"],
        "job_id": hist["job_id"],
        "timestamp": hist[DATE_COLS].min(axis=1),
        "last_stage": hist["last_stage_reached"],
    })
    if inter["timestamp"].isna().any():
        raise ValueError("history rows without any date")
    return inter.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def temporal_split(inter: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at the day where the last ``test_frac`` of interactions begins.

    Unlike the official splitter (which cuts at a row index, so one day can
    land on both sides), we cut on a day boundary: everything strictly before
    the cutoff day is train, the rest is test. This avoids same-day leakage
    under streaming evaluation and changes only a handful of rows.
    """
    ts = inter["timestamp"].sort_values(kind="mergesort").to_numpy()
    cutoff = ts[int(len(ts) * (1 - test_frac))]
    train = inter[inter["timestamp"] < cutoff].reset_index(drop=True)
    test = inter[inter["timestamp"] >= cutoff].reset_index(drop=True)
    return train, test


def make_splits(inter: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1) -> dict[str, pd.DataFrame]:
    """Train / val / test, all temporal.

    ``test`` is the last ``test_frac`` of interactions (comparable with the
    paper's 80/20 split). ``val`` is carved from the end of the remaining data
    so that ``val_frac`` is a fraction of the *whole* dataset. Tune on
    (train -> val), report on (train + val -> test).
    """
    trainval, test = temporal_split(inter, test_frac)
    train, val = temporal_split(trainval, val_frac / (1 - test_frac))
    return {"train": train, "val": val, "trainval": trainval, "test": test}


def to_days(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Datetimes -> float days since epoch (NaT -> NaN)."""
    values = pd.to_datetime(pd.Series(ts)).to_numpy(dtype="datetime64[D]")
    out = values.astype("int64").astype(float)
    out[np.isnat(values)] = np.nan
    return out
