"""TF-IDF with the fixed soft filter, using subsets of its profile fields. Validation only.

    uv run python scripts/tfidf_campos.py

Needs the validation cache from `scripts/run_references.py --split val`. Settings are chosen on
P1 and scored on P2, as in notebook v2, and intervals use the same 7-day block bootstrap.
Writes results/v2_tfidf_campos.csv. It never reads test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from jth import systems as sis
from jth.activity import SOFT_LAMBDAS, SOFT_TAUS, daily_states
from jth.context import build_context, load_official_evaluator
from jth.features import TOKEN_FIELDS, profile_tokens
from jth.ranking import summarize_ranks
from jth.uncertainty import BlockBootstrap

ROOT = Path(__file__).resolve().parents[1]
V2_START = "2023-12-14"
ALL_FIELDS = "Todos los campos"
RECRUITER_CHANGED = {"job_category", "expertise_area"}
VARIANTS = {
    ALL_FIELDS: dict(TOKEN_FIELDS),
    "Sin job_category ni expertise_area": {k: v for k, v in TOKEN_FIELDS.items() if k not in RECRUITER_CHANGED},
    "Solo campos LLM": {k: v for k, v in TOKEN_FIELDS.items() if k.startswith("llm_")},
    "Solo campos de reclutador": {k: v for k, v in TOKEN_FIELDS.items() if not k.startswith("llm_")},
}


def identity(tokens: list[str]) -> list[str]:
    return tokens


def main() -> None:
    caches = sorted((ROOT / "data" / "cache" / "references").glob("val_*"))
    if len(caches) != 1:
        raise SystemExit(f"Expected one validation cache, found {len(caches)}. Run scripts/run_references.py --split val.")
    cache = caches[0]
    evaluator = load_official_evaluator(ROOT)
    candidates = pd.read_csv(ROOT / "data" / "raw" / "candidates.csv", dtype=str)
    jobs = pd.read_csv(ROOT / "data" / "raw" / "jobs.csv", dtype=str)
    train = pd.read_csv(cache / "train.csv", dtype=str)
    target = pd.read_csv(cache / "target.csv", dtype=str)

    ctx = build_context(ROOT, cache / "train.csv", cache / "target.csv", jobs, evaluator)
    states = daily_states(ctx, train, target)
    keys = [(evaluator._parse_id(k[0]), k[1]) for k in json.loads((cache / "keys.json").read_text())]
    lr_rows = pd.Index(keys).get_indexer(ctx.keys)
    lr_scores = np.load(cache / "lr" / "scores_full.npy", mmap_mode="r")
    candidate_rows = pd.Index([evaluator._parse_id(x) for x in candidates["candidate_id"]]).get_indexer([k[0] for k in ctx.keys])
    assert (lr_rows >= 0).all() and (candidate_rows >= 0).all()
    no_cf = sp.csr_matrix((len(ctx), ctx.n_jobs))  # these scorers don't use CF

    in_v2 = np.array([str(day) >= V2_START for day in ctx.days])
    in_v1 = ~in_v2
    bootstrap = BlockBootstrap(ctx.days)
    train_candidates, train_jobs = set(train["candidate_id"]), set(train["job_id"])

    ranks_lr, ranks_v1_choice, rows = None, {}, []
    for name, fields in VARIANTS.items():
        corpus = (profile_tokens(candidates[candidates["candidate_id"].isin(train_candidates)], fields)
                  + profile_tokens(jobs[jobs["job_id"].isin(train_jobs)], fields))
        vectorizer = TfidfVectorizer(analyzer=identity, sublinear_tf=True, norm="l2").fit(corpus)
        query = vectorizer.transform(profile_tokens(candidates, fields))
        document = vectorizer.transform(profile_tokens(jobs, fields)).T.tocsr()

        def tfidf(ix: np.ndarray, query=query, document=document) -> np.ndarray:
            return (query[candidate_rows[ix]] @ document).toarray()

        inputs = sis.Inputs(ctx, states, lr_scores, lr_rows, no_cf, {"tfidf": tfidf})
        if ranks_lr is None:
            ranks_lr = inputs.ranks(sis.plain(sis.log_lr))
        base = inputs.ranks(sis.plain(sis.external("tfidf")))
        grid = {(tau, lam): inputs.ranks(sis.soft_normalized(sis.external("tfidf"), tau, lam))
                for tau in SOFT_TAUS for lam in SOFT_LAMBDAS}
        best_all = max(grid, key=lambda k: (1 / grid[k]).mean())
        best_v1 = max(grid, key=lambda k: (1 / grid[k][in_v1]).mean())
        ranks_v1_choice[name] = grid[best_v1]
        summary = summarize_ranks(ctx, grid[best_v1], in_v2)
        delta, low, high = bootstrap.difference(1 / grid[best_v1], 1 / ranks_lr, in_v2)
        rows.append({
            "Variante": name, "Campos": ";".join(fields), "Vocabulario": len(vectorizer.vocabulary_),
            "MRR base (validación)": (1 / base).mean(),
            "MRR suave fijo (validación)": (1 / grid[best_all]).mean(), "τ, λ (validación)": str(best_all),
            "τ, λ (P1)": str(best_v1), "MRR P2": summary["MRR"], "MRR P2 con historial": summary["MRR warm"],
            "MRR P2 nuevos": summary["MRR cold"], "ΔMRR P2 vs LR": delta, "IC95 vs LR inferior": low, "IC95 vs LR superior": high,
        })

    reference = ranks_v1_choice[ALL_FIELDS]
    for row in rows:
        delta, low, high = bootstrap.difference(1 / ranks_v1_choice[row["Variante"]], 1 / reference, in_v2)
        row.update({"ΔMRR P2 vs todos": delta, "IC95 vs todos inferior": low, "IC95 vs todos superior": high})
    table = pd.DataFrame(rows)
    out = ROOT / "results" / "v2_tfidf_campos.csv"
    table.to_csv(out, index=False)
    print(f"LR en P2: {(1 / ranks_lr[in_v2]).mean():.5f}")
    print(table.drop(columns="Campos").round(4).to_string(index=False))
    print(f"Escrito: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
