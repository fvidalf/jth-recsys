"""Run our baselines with the streaming evaluator.

    uv run python scripts/run_baselines.py                    # test split, both directions
    uv run python scripts/run_baselines.py --split val        # tune on validation
    uv run python scripts/run_baselines.py --split upstream   # official 80/20 split files

Writes results/baselines_<split>.csv (one row per model x direction x group)
and per-query results to results/runs/.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from jth.data import ROOT, build_interactions, load_raw, make_splits
from jth.evaluation import DEFAULT_KS, evaluate, make_context, summarize
from jth.models import BASELINES

RES_DIR = ROOT / "results"


def load_split(split: str, inter: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split == "upstream":
        d = ROOT / "data" / "upstream_split"
        if not (d / "test.csv").exists():
            raise SystemExit("run scripts/run_upstream.sh first (it writes the official split)")
        read = lambda f: pd.read_csv(d / f, dtype=str, parse_dates=["timestamp"])  # noqa: E731
        return read("train.csv"), read("test.csv")
    splits = make_splits(inter)
    return (splits["train"], splits["val"]) if split == "val" else (splits["trainval"], splits["test"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["test", "val", "upstream"], default="test")
    ap.add_argument("--modes", nargs="+", choices=["c2j", "j2c"], default=["c2j", "j2c"])
    ap.add_argument("--models", nargs="+", choices=sorted(BASELINES), default=list(BASELINES))
    ap.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cands, jobs, hist = load_raw()
    train, test = load_split(args.split, build_interactions(hist))
    print(f"split={args.split}: train {len(train):,} rows, test {len(test):,} rows "
          f"({test['timestamp'].min().date()} to {test['timestamp'].max().date()})")

    run_dir = RES_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in args.modes:
        ctx = make_context(cands, jobs, mode)
        for key in args.models:
            model = BASELINES[key]()
            t0 = time.perf_counter()
            per_query = evaluate(model, train, test, ctx, ks=args.ks, seed=args.seed)
            secs = time.perf_counter() - t0
            per_query.to_csv(run_dir / f"{args.split}_{mode}_{key}.csv.gz", index=False)
            summ = summarize(per_query)
            summ.insert(0, "model", model.name)
            summ.insert(1, "mode", mode)
            summ["seconds"] = round(secs, 1)
            rows.append(summ)
            a = summ.iloc[0]
            print(f"{mode} {model.name:<13} MRR {a['MRR']:.4f}  NDCG@10 {a['NDCG@10']:.4f}  "
                  f"R@10 {a['R@10']:.4f}  ({secs:.0f}s)")

    out = pd.concat(rows, ignore_index=True)
    path = RES_DIR / f"baselines_{args.split}.csv"
    out.to_csv(path, index=False, float_format="%.5f")
    print("summary ->", path)

    cols = ["model", "mode", "group", "n_queries", "MRR"] + [f"{m}@{k}" for m in ("HR", "NDCG") for k in args.ks]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
