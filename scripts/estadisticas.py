"""Descriptive statistics for Section 3 (and the numbers quoted in Section 2).

    uv run python scripts/estadisticas.py   # -> results/tabla_datos.tex

Uses the official 80/20 chronological split (data/upstream_split, written by
scripts/run_upstream.sh), the same one behind the baseline table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from jth.data import ROOT, load_raw

SPLIT = ROOT / "data" / "upstream_split"


def read_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(SPLIT / f"{name}.csv", dtype=str, parse_dates=["timestamp"])
    df["day"] = df["timestamp"].dt.normalize()
    return df


def describe(df: pd.DataFrame) -> dict[str, str]:
    pc = df.groupby("candidate_id").size()
    pj = df.groupby("job_id").size()
    n_c, n_j = len(pc), len(pj)
    return {
        "Número de interacciones": f"{len(df):,}",
        "Número de candidatos": f"{n_c:,}",
        "Número de ofertas": f"{n_j:,}",
        "Interacciones por candidato (prom.)": f"{pc.mean():.2f}",
        "Interacciones por candidato (mín/med/máx)": f"{pc.min()}/{pc.median():.0f}/{pc.max()}",
        "Interacciones por oferta (prom.)": f"{pj.mean():.2f}",
        "Interacciones por oferta (mín/med/máx)": f"{pj.min()}/{pj.median():.0f}/{pj.max()}",
        "Densidad": f"{len(df) / (n_c * n_j):.3%}",
    }


def to_latex_num(s: str) -> str:
    """Spanish number style: 1.234,56 and escaped %."""
    s = s.replace(",", "@").replace(".", ",").replace("@", ".")
    return s.replace("%", r"\%")


def main() -> None:
    cands, jobs, hist = load_raw()
    train, test = read_split("train"), read_split("test")
    total = pd.concat([train, test])

    cols = {"Total": describe(total), "Entrenamiento": describe(train), "Prueba": describe(test)}
    table = pd.DataFrame(cols)
    new_c = ~test["candidate_id"].drop_duplicates().isin(train["candidate_id"])
    new_j = ~test["job_id"].drop_duplicates().isin(train["job_id"])
    table.loc["Candidatos nuevos (sin entrenamiento)"] = ["", "", f"{new_c.mean():.1%}"]
    table.loc["Ofertas nuevas (sin entrenamiento)"] = ["", "", f"{new_j.mean():.1%}"]
    print(table.to_string())

    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Estadística & Total & Entrenamiento & Prueba \\", r"\midrule"]
    for name, row in table.iterrows():
        lines.append(f"{name} & " + " & ".join(to_latex_num(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Estadísticas de las interacciones en los conjuntos de entrenamiento y prueba.}",
              r"\label{tab:datos}", r"\end{table}"]
    (ROOT / "results" / "tabla_datos.tex").write_text("\n".join(lines) + "\n")
    print("table ->", ROOT / "results" / "tabla_datos.tex")

    # ---- context numbers quoted in Section 2 (official split) --------------
    allv = total.sort_values("day", kind="mergesort").reset_index(drop=True)
    for col, name in [("job_id", "job_prior"), ("candidate_id", "cand_prior")]:
        g = allv.groupby([col, "day"]).size().rename("n").reset_index().sort_values([col, "day"])
        g[name] = g.groupby(col)["n"].cumsum() - g["n"]
        allv = allv.merge(g[[col, "day", name]], on=[col, "day"], how="left")
    t = allv[allv.set_index(["candidate_id", "job_id", "timestamp"]).index.isin(
        test.set_index(["candidate_id", "job_id", "timestamp"]).index)]
    jc = jobs.set_index("job_id")["create_date"]
    hits, pool_sizes = [], []
    for d, grp in t.groupby("day"):
        pool = set(allv.loc[(allv["day"] < d) & (allv["day"] >= d - pd.Timedelta(days=30)), "job_id"])
        hits.append(grp["job_id"].isin(pool).to_numpy())
        pool_sizes.append(len(pool))
    print("\nSection 2 numbers (official split, test period):")
    print(f"  correct job had interactions in previous 30 d: {np.concatenate(hits).mean():.1%}")
    print(f"  size of that 30-day pool: median {np.median(pool_sizes):.0f} jobs; "
          f"jobs published by then: median {np.median([(jc <= d).sum() for d in t['day'].unique()]):,.0f}")
    print(f"  correct job already had interactions before the day: {(t['job_prior'] > 0).mean():.1%}")
    print(f"  candidate already had interactions before the day: {(t['cand_prior'] > 0).mean():.1%}")

    # ---- raw file shapes -----------------------------------------------------
    print("\nFiles:", {n: df.shape for n, df in [("candidates", cands), ("jobs", jobs), ("history", hist)]})
    print("candidate LLM columns:", sum(c.startswith("llm_") for c in cands.columns),
          "| job LLM columns:", sum(c.startswith("llm_") for c in jobs.columns))
    print(hist["last_stage_reached"].value_counts().to_string())


if __name__ == "__main__":
    main()
