"""LaTeX table of the reference models (Section 4), from the official-code re-run.

    uv run python scripts/make_table.py   # -> results/tabla_baselines.tex

Needs results/upstream/results.csv (WITH_SLOW=1 bash scripts/run_upstream.sh, for the LogReg row).
Uses booktabs (\\usepackage{booktabs}).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from jth.data import ROOT

ROWS = [  # prediction file -> label in the report
    ("random_c2j", r"\textit{Random}"),
    ("popularity_c2j", r"\textit{Most Popular}"),
    ("pasttemporal_c2j", "Recencia"),
    ("cfrecency_c2j", "CF + recencia"),
    ("logreg_c2j", "Regresión logística"),
]
COLS = [("mrr", "MRR"), ("NDCG@10", "NDCG@10"),
        ("mrr-warm", "MRR con historial"), ("mrr-cold", "MRR nuevos")]


def fmt(v: float, best: bool) -> str:
    s = f"{v:.3f}".replace(".", ",")
    return f"\\textbf{{{s}}}" if best else s


def main() -> None:
    res = pd.read_csv(ROOT / "results" / "upstream" / "results.csv")
    res["key"] = res["file"].map(lambda f: Path(f).name.removesuffix(".pkl.gz"))
    res = res.drop_duplicates("key", keep="last").set_index("key")
    table = res.loc[[k for k, _ in ROWS], [c for c, _ in COLS]]
    best = table.max()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(COLS) + "}",
        r"\toprule",
        "Modelo & " + " & ".join(h for _, h in COLS) + r" \\",
        r"\midrule",
    ]
    for key, label in ROWS:
        cells = [fmt(table.at[key, c], table.at[key, c] == best[c]) for c, _ in COLS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Modelos de referencia en la recomendación de ofertas a candidatos.}",
        r"\label{tab:baselines}",
        r"\end{table}",
    ]
    out = ROOT / "results" / "tabla_baselines.tex"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("table ->", out)


if __name__ == "__main__":
    main()
