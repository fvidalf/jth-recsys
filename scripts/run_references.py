"""Run the official reference models on a named split, with a verifiable cache.

    uv run python scripts/run_references.py --split val    # reduced train -> validation
    uv run python scripts/run_references.py --split test   # official train -> test (needs v2_frozen.json)

Adapted from the runner embedded in scripts/01_datos_y_validacion.ipynb (cell 58).
The official code is called unmodified. LR's `predict_proba` is intercepted only to
keep the full score vector before its top-1000 cut.

The cache lives in the git-ignored data/cache/references/<signature>/ and is reused
only if data, code, versions, parameters and artifacts all match.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.dont_write_bytecode = True

BASELINES = "external/JTH/code/baselines"
SPLITTER = "external/JTH/code/utils/binary_splitter.py"
EVALUATOR = "external/JTH/code/utils/evaluator.py"
LR_CODE = f"{BASELINES}/feature_classifier_baseline.py"
CF_CODE = f"{BASELINES}/hybrid_recency_cf_baseline.py"
RECENCY_CODE = f"{BASELINES}/past_temporal_baseline.py"
RANDOM_CODE = f"{BASELINES}/random_baseline.py"
POPULARITY_CODE = f"{BASELINES}/online_popularity_baseline.py"
DATA_FILES = ["data/raw/candidates.csv", "data/raw/jobs.csv", "data/raw/history.csv"]
VALIDATION_FRACTION = 0.125
SEED = 42

# Expected sizes: training processes, evaluated processes, candidate-day queries.
EXPECTED = {"val": (29599, 4231, 4208), "test": (33830, 8458, 8341)}
FROZEN = "results/v2_frozen.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def sha_text(path: Path) -> str:
    """Hash of a text file with normalized newlines, comparable across platforms."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run_splitter(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with tempfile.TemporaryDirectory(prefix="jth_official_split_") as td:
        subprocess.run([sys.executable, str(repo / SPLITTER), str(repo / "data/raw/history.csv"), td],
                       check=True, capture_output=True, text=True)
        return (pd.read_csv(Path(td) / "train.csv", dtype=str),
                pd.read_csv(Path(td) / "test.csv", dtype=str))


def prepare(repo: Path, split: str) -> Path:
    code = [SPLITTER, EVALUATOR, LR_CODE, CF_CODE, RECENCY_CODE, RANDOM_CODE, POPULARITY_CODE]
    config = {
        "split": split,
        "data": {p: sha(repo / p) for p in DATA_FILES},
        "code": {p: sha_text(repo / p) for p in code},
        "runner": sha_text(Path(__file__)),
        "validation_fraction": VALIDATION_FRACTION, "seed": SEED, "pythonhashseed": str(SEED),
        "lr": {"model": "logreg", "neg_ratio": 1.0, "temporal": True, "llm": True},
        "cf_max_age": 90, "threads": 1,
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__,
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cache = repo / "data" / "cache" / "references" / f"{split}_{signature[:24]}"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if not (cache / "train.csv").exists():
        official_train, official_test = run_splitter(repo)
        if split == "val":
            days = pd.to_datetime(official_train["timestamp"]).dt.normalize()
            cutoff = days.sort_values(kind="mergesort").iloc[int(len(official_train) * (1 - VALIDATION_FRACTION))]
            train, target = official_train.loc[days < cutoff], official_train.loc[days >= cutoff]
        else:
            train, target = official_train, official_test
        train.to_csv(cache / "train.csv", index=False)
        target.to_csv(cache / "target.csv", index=False)

    train = pd.read_csv(cache / "train.csv", dtype=str)
    target = pd.read_csv(cache / "target.csv", dtype=str)
    n_train, n_target, n_queries = EXPECTED[split]
    assert len(train) == n_train and len(target) == n_target, (len(train), len(target))
    t_last, t_first = pd.to_datetime(train["timestamp"]).max(), pd.to_datetime(target["timestamp"]).min()
    # The official test shares one day with official train (row-index cut); validation does not.
    assert (t_last < t_first) if split == "val" else (t_last <= t_first)
    keys = target[["candidate_id", "timestamp"]].drop_duplicates().values.tolist()
    assert len(keys) == n_queries, len(keys)
    jobs = pd.read_csv(repo / "data/raw/jobs.csv", dtype=str)["job_id"].tolist()
    (cache / "keys.json").write_text(json.dumps(keys), encoding="utf-8")
    (cache / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    config["train_sha256"] = sha_text(cache / "train.csv")
    config["target_sha256"] = sha_text(cache / "target.csv")
    (cache / "inputs.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return cache


def lr_worker(repo: Path, cache: Path) -> None:
    """Official LR main(), keeping its full scores before the top-1000 cut."""
    from sklearn.pipeline import Pipeline

    repo, cache = Path(repo), Path(cache)
    folder = cache / "lr"
    folder.mkdir(exist_ok=True)
    source = repo / LR_CODE
    spec = importlib.util.spec_from_file_location("jth_lr_official", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys = json.loads((cache / "keys.json").read_text())
    jobs = json.loads((cache / "jobs.json").read_text())
    row_index = {tuple(k): i for i, k in enumerate(keys)}
    target = pd.read_csv(cache / "target.csv", dtype=str)
    row_calls = list(zip(target.candidate_id, target.timestamp))
    matrix = np.lib.format.open_memmap(folder / "scores_full.npy", mode="w+", dtype="float64",
                                       shape=(len(keys), len(jobs)))
    matrix[:] = np.nan
    original_predict, original_fit = Pipeline.predict_proba, Pipeline.fit
    calls = 0
    started = time.perf_counter()

    def fit_logged(self, X, y, *args, **kwargs):
        result = original_fit(self, X, y, *args, **kwargs)
        metadata = {"samples": len(y), "positives": int(np.sum(y == 1)), "negatives": int(np.sum(y == 0)),
                    "features": list(X.columns), "coefficients": self[-1].coef_[0].tolist(),
                    "n_iter": self[-1].n_iter_.tolist()}
        (folder / "fit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("LR fit completed: samples=", len(y), "features=", len(X.columns), flush=True)
        return result

    def predict_logged(self, X, *args, **kwargs):
        nonlocal calls
        probabilities = original_predict(self, X, *args, **kwargs)
        matrix[row_index[row_calls[calls]]] = probabilities[:, 1]
        calls += 1
        if calls % 250 == 0:
            matrix.flush()
            print("LR ranking rows", calls, "/", len(row_calls), flush=True)
        return probabilities

    Pipeline.fit, Pipeline.predict_proba = fit_logged, predict_logged
    os.chdir(folder)
    try:
        module.main([str(repo / "data/raw/candidates.csv"), str(repo / "data/raw/jobs.csv"),
                     str(cache / "train.csv"), str(cache / "target.csv"), str(folder / "lr_c2j.pkl.gz"),
                     "--mode", "c2j", "--neg_ratio", "1.0", "--model", "logreg",
                     "--use_temporal_features", "--use_llm_features", "--seed", str(SEED)])
    finally:
        Pipeline.fit, Pipeline.predict_proba = original_fit, original_predict
        matrix.flush()
    assert calls == len(target) and np.isfinite(matrix).all()
    (folder / "timing.json").write_text(json.dumps({"seconds": time.perf_counter() - started, "rows": calls}),
                                        encoding="utf-8")


def commands(repo: Path, cache: Path) -> dict[str, tuple[Path, list[str]]]:
    """name -> (prediction file, command). Positional order follows each official script."""
    py = [sys.executable, "-u"]
    cand, jobs = str(repo / "data/raw/candidates.csv"), str(repo / "data/raw/jobs.csv")
    train, target = str(cache / "train.csv"), str(cache / "target.csv")
    out = {name: cache / name / f"{name}_c2j.pkl.gz" for name in ("lr", "cf", "recency", "random", "popularity")}
    return {
        "lr": (out["lr"], py + [str(Path(__file__).resolve()), "--lr-worker", str(cache), "--repo", str(repo)]),
        "cf": (out["cf"], py + [str(repo / CF_CODE), train, target, cand, jobs, str(out["cf"]),
                                "--mode", "c2j", "--max_age", "90"]),
        "recency": (out["recency"], py + [str(repo / RECENCY_CODE), cand, jobs, target, str(out["recency"]),
                                          "--mode", "c2j", "--seed", str(SEED)]),
        "random": (out["random"], py + [str(repo / RANDOM_CODE), cand, jobs, target, str(out["random"]),
                                        "--mode", "c2j", "--seed", str(SEED)]),
        "popularity": (out["popularity"], py + [str(repo / POPULARITY_CODE), train, target, str(out["popularity"]),
                                                "--mode", "c2j"]),
    }


def artifacts(cache: Path) -> list[str]:
    names = [f"{n}/{n}_c2j.pkl.gz" for n in ("lr", "cf", "recency", "random", "popularity")]
    return names + ["lr/scores_full.npy", "lr/fit.json", "lr/timing.json", "keys.json", "jobs.json"]


def cache_valid(cache: Path) -> bool:
    done = cache / "completed.json"
    if not done.exists():
        return False
    info = json.loads(done.read_text())
    if info["inputs_sha256"] != sha(cache / "inputs.json"):
        return False
    return all((cache / rel).exists() and sha(cache / rel) == digest for rel, digest in info["artifacts"].items())


def execute(repo: Path, split: str) -> Path:
    repo = repo.resolve()
    if split == "test":
        frozen = repo / FROZEN
        assert frozen.is_file(), f"Test runs only after decisions are frozen in {FROZEN}."
        tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", FROZEN], capture_output=True)
        assert tracked.returncode == 0, f"Commit {FROZEN} before running test."
    cache = prepare(repo, split)
    print("CACHE:", cache.relative_to(repo), flush=True)
    if cache_valid(cache):
        print("Verified reference cache reused.", flush=True)
        return cache

    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", PYTHONHASHSEED=str(SEED), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    jobs, logs = {}, {}
    started = time.perf_counter()
    try:
        for name, (pred, command) in commands(repo, cache).items():
            (cache / name).mkdir(exist_ok=True)
            assert pred.resolve().is_relative_to(cache.resolve())
            if pred.exists():
                pred.unlink()  # an incomplete file, inside this hashed cache only
            logs[name] = (cache / name / "execution.log").open("w", encoding="utf-8")
            jobs[name] = subprocess.Popen(command, cwd=cache / name, stdout=logs[name],
                                          stderr=subprocess.STDOUT, env=env)
        while any(p.poll() is None for p in jobs.values()):
            time.sleep(30)
            tails = []
            for name, p in jobs.items():
                lines = (cache / name / "execution.log").read_text(encoding="utf-8", errors="replace").splitlines()
                state = "running" if p.poll() is None else f"exit={p.returncode}"
                tails.append(f"{name}: {state}" + (f" | {lines[-1][:60]}" if lines and p.poll() is None else ""))
            print(f"{time.perf_counter() - started:.0f}s | " + " || ".join(tails), flush=True)
        for name, p in jobs.items():
            if p.returncode:
                text = (cache / name / "execution.log").read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(f"{name} failed: {text[-4000:]}")
    finally:
        for p in jobs.values():
            if p.poll() is None:
                p.terminate()
        for stream in logs.values():
            stream.close()
    info = {"inputs_sha256": sha(cache / "inputs.json"),
            "artifacts": {rel: sha(cache / rel) for rel in artifacts(cache)},
            "seconds": time.perf_counter() - started}
    (cache / "completed.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Reference runs completed and hashed.", flush=True)
    return cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--lr-worker", type=Path)
    args = parser.parse_args()
    if args.lr_worker:
        lr_worker(args.repo, args.lr_worker)
    else:
        execute(args.repo, args.split)
