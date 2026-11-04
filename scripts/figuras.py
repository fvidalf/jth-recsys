"""Figures and coverage table for Section 3 (Análisis de datos).

    uv run python scripts/figuras.py

Writes figures/fig_etapas.png, figures/fig_interacciones.png,
figures/fig_temporal.png and results/tabla_cobertura.tex.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jth.data import ROOT, build_interactions, load_raw

FIG = ROOT / "figures"
MILES = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))  # 2.500
BLUE, GRAY = "#2a78d6", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

STAGES = {  # last_stage_reached -> Spanish label, in funnel order
    "Application Made": "Postulación",
    "Shortlist": "Preselección",
    "Qualification": "Calificación",
    "Resume Sent": "CV enviado",
    "1st Interview": "1.ª entrevista",
    "2nd Interview": "2.ª entrevista",
    "3rd Interview": "3.ª entrevista",
    "4th Interview": "4.ª entrevista",
    "Offer Received": "Oferta recibida",
    "Offer Accepted": "Oferta aceptada",
}


def label_bars(ax, bars, horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v, b.get_y() + b.get_height() / 2, f" {int(v):,}".replace(",", "."), va="center", fontsize=8)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}".replace(",", "."), ha="center", va="bottom", fontsize=7)


def fig_etapas(hist):
    counts = hist["last_stage_reached"].value_counts().reindex(list(STAGES)).fillna(0)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    y = np.arange(len(counts))[::-1]
    bars = ax.barh(y, counts.to_numpy(), color=BLUE, height=0.65)
    label_bars(ax, bars, horizontal=True)
    ax.set_yticks(y, [STAGES[s] for s in counts.index])
    ax.set_xlim(0, counts.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(MILES)
    ax.set_xlabel("Número de procesos")
    ax.set_ylabel("Etapa máxima alcanzada")
    ax.set_title("Distribución de la etapa máxima alcanzada por proceso")
    fig.savefig(FIG / "fig_etapas.png")
    plt.close(fig)


def fig_interacciones(inter):
    per_c = inter.groupby("candidate_id").size()
    per_j = inter.groupby("job_id").size()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # candidates: discrete counts 1..9 and 10+
    k = np.minimum(per_c.to_numpy(), 10)
    vals = np.bincount(k, minlength=11)[1:]
    bars = a1.bar(np.arange(1, 11), vals, color=BLUE, width=0.8)
    label_bars(a1, bars)
    a1.set_xticks(np.arange(1, 11), [str(i) for i in range(1, 10)] + ["10+"])
    a1.set_ylim(0, vals.max() * 1.12)
    a1.grid(axis="x", visible=False)
    a1.set_xlabel("Número de interacciones")
    a1.yaxis.set_major_formatter(MILES)
    a1.set_ylabel("Número de candidatos")
    a1.set_title("Interacciones por candidato")

    # jobs: bins of 5, last bin 100+
    edges = np.arange(0, 105, 5)
    clipped = np.minimum(per_j.to_numpy(), 100)
    a2.hist(clipped, bins=np.append(edges, 105) + 0.5, color=BLUE, edgecolor="white", linewidth=0.6)
    a2.set_xticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100+"])
    a2.grid(axis="x", visible=False)
    a2.set_xlabel("Número de interacciones")
    a2.yaxis.set_major_formatter(MILES)
    a2.set_ylabel("Número de ofertas")
    a2.set_title("Interacciones por oferta")
    fig.tight_layout()
    fig.savefig(FIG / "fig_interacciones.png")
    plt.close(fig)


def fig_temporal(inter, jobs, test_start):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # interactions per month; first (03/2018) and last (04/2025) months are partial -> dropped
    monthly = inter.set_index("timestamp").resample("MS").size().iloc[1:-1]
    a1.plot(monthly.index, monthly.to_numpy(), color=BLUE, lw=1.6)
    a1.axvline(test_start, color=GRAY, lw=1, ls="--")
    a1.set_ylim(bottom=0)
    a1.text(test_start, 25, " prueba", color=GRAY, va="bottom", fontsize=8)
    a1.set_xlabel("Mes")
    a1.yaxis.set_major_formatter(MILES)
    a1.set_ylabel("Número de interacciones")
    a1.set_title("Interacciones por mes")

    # job age at interaction (days since publication), weekly bins up to 26 weeks
    created = inter["job_id"].map(jobs.set_index("job_id")["create_date"])
    age = (inter["timestamp"] - created).dt.days.to_numpy()
    cap = 182
    a2.hist(np.minimum(age, cap), bins=np.arange(0, cap + 8, 7), color=BLUE, edgecolor="white", linewidth=0.6)
    med = float(np.median(age))
    a2.axvline(med, color=GRAY, lw=1, ls="--")
    a2.text(med, a2.get_ylim()[1] * 0.97, f" mediana: {med:.0f} días", color=GRAY, va="top", fontsize=8)
    a2.set_xticks([0, 28, 56, 84, 112, 140, 182], ["0", "28", "56", "84", "112", "140", "182+"])
    a2.grid(axis="x", visible=False)
    a2.set_xlabel("Días desde la publicación de la oferta")
    a2.yaxis.set_major_formatter(MILES)
    a2.set_ylabel("Número de interacciones")
    a2.set_title("Antigüedad de la oferta al interactuar")
    fig.tight_layout()
    fig.savefig(FIG / "fig_temporal.png")
    plt.close(fig)
    return med, float(np.quantile(age, 0.9))


def tabla_cobertura(cands, jobs):
    rows = [  # attribute, candidate (human, llm), job (human, llm)
        ("Habilidades", ("skills", "llm_hard_skills"), ("skills", "llm_hard_skills")),
        ("Área de experiencia", ("expertise_area", "llm_expertise_area"), ("expertise_area", "llm_expertise_area")),
        ("Categoría de cargo", ("job_category", "llm_job_category"), ("job_category", "llm_job_category")),
        ("Años de experiencia", ("years_experience", "llm_years_of_work_experience"),
         ("years_experience", "llm_required_years_of_work_experience")),
        ("Salario", ("actual_salary", "llm_estimated_salary_eur_per_year"),
         ("salary", "llm_salary_estimation_eur_per_year")),
    ]
    pct = lambda df, c: f"{df[c].notna().mean():.0%}".replace("%", r"\%")  # noqa: E731
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{Candidatos} & \multicolumn{2}{c}{Ofertas} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
             r"Atributo & Reclutador & LLM & Reclutador & LLM \\", r"\midrule"]
    for name, (ch, cl), (jh, jl) in rows:
        lines.append(f"{name} & {pct(cands, ch)} & {pct(cands, cl)} & {pct(jobs, jh)} & {pct(jobs, jl)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Porcentaje de perfiles con cada atributo, según si fue ingresado por un reclutador o inferido por un LLM.}",
              r"\label{tab:cobertura}", r"\end{table}"]
    out = ROOT / "results" / "tabla_cobertura.tex"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    FIG.mkdir(exist_ok=True)
    cands, jobs, hist = load_raw()
    inter = build_interactions(hist)
    test_start = pd.read_csv(ROOT / "data" / "upstream_split" / "test.csv", parse_dates=["timestamp"])["timestamp"].min()
    fig_etapas(hist)
    fig_interacciones(inter)
    med, p90 = fig_temporal(inter, jobs, test_start)
    print(f"job age at interaction: median {med:.0f} d, p90 {p90:.0f} d; test starts {test_start:%Y-%m-%d}")
    tabla_cobertura(cands, jobs)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
