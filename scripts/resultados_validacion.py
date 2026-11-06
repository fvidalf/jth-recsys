"""Tablas del informe intermedio, extraídas del notebook de validación.

    uv run python scripts/resultados_validacion.py

Fuente: ``scripts/01_datos_y_validacion.ipynb``, cuyos outputs están versionados en
git. El notebook es la fuente de verdad y este script no vuelve a correr los modelos:
copia las cuatro tablas que imprime su Sección 15 (comparación final, aportes,
intervalos frente a LR oficial e intervalos pareados) a ``results/*.csv`` y arma
``results/tabla_componentes.tex`` y ``results/tabla_aportes.tex``.

Los valores son los que el notebook imprime, redondeados a seis decimales. Si el
notebook se vuelve a ejecutar, correr esto otra vez para refrescar los CSV y las
tablas. Solo usa la biblioteca estándar, así que no necesita los datos ni el
entorno de uv.

Convenciones del informe (ver docs/report_style.md): coma decimal, miles con punto,
booktabs y flotantes [htbp].
"""
from __future__ import annotations

import csv
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "scripts" / "01_datos_y_validacion.ipynb"

# Celda que imprime la comparación final: es la que manda sobre las tablas parciales
# de las secciones anteriores, que traen las mismas columnas con menos modelos.
MARCADOR_FINAL = "tabla_final_modelos"

# Controles que sostienen el argumento del informe. Si el notebook cambia de forma
# inesperada conviene enterarse acá y no en el PDF.
CONSULTAS_ESPERADAS = 4208.0
GANADOR_ESPERADO = "Fusión + actividad"
CONTROL_ESPERADO = "LR oficial"


class _TablaHTML(HTMLParser):
    """Junta las celdas de la primera ``<table>`` de un fragmento HTML.

    Los ``display()`` de pandas emiten una tabla plana, así que basta con acumular
    filas y celdas. ``convert_charrefs`` resuelve las entidades.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.filas: list[list[str]] = []
        self._celda: list[str] | None = None
        self._en_tabla = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._en_tabla += 1
        elif tag == "tr" and self._en_tabla:
            self.filas.append([])
        elif tag in ("th", "td") and self._en_tabla:
            self._celda = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._en_tabla -= 1
        elif tag in ("th", "td") and self._celda is not None:
            self.filas[-1].append("".join(self._celda).strip())
            self._celda = None

    def handle_data(self, data: str) -> None:
        if self._celda is not None:
            self._celda.append(data)


def _tabla_de_html(fragmento: str) -> tuple[list[str], list[list[str]]]:
    parser = _TablaHTML()
    parser.feed(fragmento)
    filas = [fila for fila in parser.filas if fila]
    if len(filas) < 2:
        raise SystemExit("fragmento HTML sin cabecera o sin filas")
    cabecera, cuerpo = filas[0], filas[1:]
    # Un índice con nombre sale en una segunda fila de <thead> donde solo la primera
    # celda trae texto. Se promueve a nombre de la columna del índice.
    if (
        len(cabecera) > 1
        and not cabecera[0]
        and cuerpo
        and cuerpo[0][0]
        and not any(cuerpo[0][1:])
    ):
        cabecera = [cuerpo[0][0]] + cabecera[1:]
        cuerpo = cuerpo[1:]
    return cabecera, cuerpo


def tablas_del_notebook(ruta: Path = NOTEBOOK) -> list[tuple[int, str, list[str], list[list[str]]]]:
    """Tablas que imprime el notebook, con el número y la fuente de la celda."""
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    salida = []
    for numero, celda in enumerate(nb["cells"], start=1):
        if celda.get("cell_type") != "code":
            continue
        fuente = "".join(celda.get("source", []))
        for output in celda.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if html:
                cabecera, filas = _tabla_de_html("".join(html))
                salida.append((numero, fuente, cabecera, filas))
    return salida


def buscar(tablas, columnas: set[str], marca: str | None = None):
    """Primera tabla con todas esas columnas, opcionalmente de una celda marcada."""
    for numero, fuente, cabecera, filas in tablas:
        if marca is not None and marca not in fuente:
            continue
        if columnas <= set(cabecera):
            return numero, cabecera, filas
    raise SystemExit(f"no encontré una tabla con las columnas {sorted(columnas)} y marca {marca!r}")


def sin_indice_anonimo(
    cabecera: list[str], filas: list[list[str]]
) -> tuple[list[str], list[list[str]]]:
    """Quita la columna del índice cuando pandas no le dio nombre (0, 1, 2...)."""
    if cabecera and cabecera[0] == "":
        return cabecera[1:], [fila[1:] for fila in filas]
    return cabecera, filas


def escribir_csv(
    destino: Path, cabecera: list[str], filas: list[list[str]], indice: str | None = None
) -> None:
    if indice is not None and cabecera and cabecera[0] == "":
        cabecera = [indice] + cabecera[1:]
    with destino.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(cabecera)
        escritor.writerows(filas)
    print(f"csv    -> {destino.relative_to(ROOT)}  ({len(filas)} filas)")


def num(texto: str, cifras: int = 3) -> str:
    """Número con coma decimal; ``--`` si el notebook no imprime valor."""
    if texto in ("", "NaN", "nan", "None"):
        return "--"
    return f"{float(texto):.{cifras}f}".replace(".", ",")


def diferencia(texto: str) -> str:
    """Diferencia con signo y cuatro decimales, en modo matemático de LaTeX."""
    if texto in ("", "NaN", "nan", "None"):
        return "--"
    return f"${float(texto):+.4f}$".replace(".", ",")


def tabla_componentes(cabecera, filas, intervalos, destino: Path) -> None:
    """Una fila por modelo, con su diferencia pareada frente a LR oficial."""
    posicion = {columna: i for i, columna in enumerate(cabecera)}
    metricas = ["MRR", "nDCG@10", "MRR warm", "MRR cold"]
    maximos = {m: max(float(fila[posicion[m]]) for fila in filas) for m in metricas}

    lineas = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Modelo & MRR & NDCG@10 & MRR con historial & MRR nuevos & "
        r"$\Delta$MRR vs LR oficial (IC 95\%) \\",
        r"\midrule",
    ]
    for fila in filas:
        nombre = fila[0]
        celdas = []
        for metrica in metricas:
            valor = float(fila[posicion[metrica]])
            texto = num(fila[posicion[metrica]])
            if valor == maximos[metrica]:
                texto = f"\\textbf{{{texto}}}"
            celdas.append(texto)
        if nombre in intervalos:
            delta = diferencia(intervalos[nombre]["delta"])
            bajo = num(intervalos[nombre]["bajo"], 4)
            alto = num(intervalos[nombre]["alto"], 4)
            celdas.append(f"{delta} $[{bajo}; {alto}]$")
        else:
            celdas.append("--")
        lineas.append(f"{nombre} & " + " & ".join(celdas) + r" \\")
    lineas += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Comparación final en validación. La última columna es la diferencia "
        r"pareada de MRR frente a la regresión logística oficial, con un intervalo de "
        r"confianza del 95\% por \textit{bootstrap} de bloques de siete días.}",
        r"\label{tab:componentes}",
        r"\end{table}",
    ]
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"tabla  -> {destino.relative_to(ROOT)}")


def tabla_aportes(cabecera, aportes, cab_pares, pares, destino: Path) -> None:
    """Una fila por comparación: qué agrega cada pieza sobre su control."""
    pos = {columna: i for i, columna in enumerate(cabecera)}
    pos_par = {columna: i for i, columna in enumerate(cab_pares)}
    clave = lambda fila: (fila[pos["Variante"]], fila[pos["Control"]])  # noqa: E731
    por_clave = {clave(fila): fila for fila in aportes}
    por_par = {(fila[pos_par["Variante"]], fila[pos_par["Control"]]): fila for fila in pares}
    claves = list(por_clave)
    claves += [k for k in por_par if k not in por_clave]

    lineas = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Variante & Control & $\Delta$MRR & $\Delta$MRR con historial & "
        r"$\Delta$MRR nuevos & IC 95\% del $\Delta$MRR \\",
        r"\midrule",
    ]
    for comparacion in claves:
        fila = por_clave.get(comparacion)
        par = por_par.get(comparacion)
        if fila is not None:
            variante, control = fila[pos["Variante"]], fila[pos["Control"]]
            dmrr = diferencia(fila[pos["ΔMRR"]])
            dwarm = diferencia(fila[pos["ΔMRR warm"]])
            dcold = diferencia(fila[pos["ΔMRR cold"]])
        else:  # comparación que solo existe en la tabla de intervalos pareados
            variante, control = par[pos_par["Variante"]], par[pos_par["Control"]]
            dmrr, dwarm, dcold = diferencia(par[pos_par["ΔMRR"]]), "--", "--"
        intervalo = (
            f"$[{num(par[pos_par['IC95 inferior']], 4)}; "
            f"{num(par[pos_par['IC95 superior']], 4)}]$"
            if par is not None
            else "--"
        )
        lineas.append(
            f"{variante} & {control} & {dmrr} & {dwarm} & {dcold} & {intervalo} \\\\"
        )
    lineas += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Aporte de cada componente frente a su control, en validación.}",
        r"\label{tab:aportes}",
        r"\end{table}",
    ]
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"tabla  -> {destino.relative_to(ROOT)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # consola cp1252 en Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tablas = tablas_del_notebook()
    _, cab_modelos, fil_modelos = buscar(
        tablas,
        {"Consultas", "MRR", "nDCG@10", "HR@10", "MRR warm", "MRR cold"},
        marca=MARCADOR_FINAL,
    )
    _, cab_aportes, fil_aportes = buscar(
        tablas, {"Variante", "Control", "ΔMRR", "ΔMRR warm", "ΔMRR cold"}, marca=MARCADOR_FINAL
    )
    _, cab_intervalos, fil_intervalos = buscar(
        tablas,
        {"Modelo", "ΔMRR vs LR oficial", "IC95 inferior", "IC95 superior"},
        marca=MARCADOR_FINAL,
    )
    _, cab_pares, fil_pares = buscar(
        tablas,
        {"Variante", "Control", "ΔMRR", "IC95 inferior", "IC95 superior"},
        marca=MARCADOR_FINAL,
    )
    cab_aportes, fil_aportes = sin_indice_anonimo(cab_aportes, fil_aportes)
    cab_pares, fil_pares = sin_indice_anonimo(cab_pares, fil_pares)

    consultas = {float(fila[cab_modelos.index("Consultas")]) for fila in fil_modelos}
    assert consultas == {CONSULTAS_ESPERADAS}, f"consultas inesperadas: {consultas}"
    mejor = max(fil_modelos, key=lambda fila: float(fila[cab_modelos.index("MRR")]))
    assert mejor[0] == GANADOR_ESPERADO, f"el ganador por MRR cambió: {mejor[0]}"

    pos = {columna: i for i, columna in enumerate(cab_intervalos)}
    intervalos = {
        fila[pos["Modelo"]]: {
            "delta": fila[pos["ΔMRR vs LR oficial"]],
            "bajo": fila[pos["IC95 inferior"]],
            "alto": fila[pos["IC95 superior"]],
        }
        for fila in fil_intervalos
    }
    assert CONTROL_ESPERADO in intervalos, "falta el intervalo del control"
    assert abs(float(intervalos[CONTROL_ESPERADO]["delta"])) < 1e-6

    escribir_csv(ROOT / "results" / "validacion_modelos.csv", cab_modelos, fil_modelos, "Modelo")
    escribir_csv(ROOT / "results" / "validacion_aportes.csv", cab_aportes, fil_aportes)
    escribir_csv(
        ROOT / "results" / "validacion_intervalos.csv", cab_intervalos, fil_intervalos, "Modelo"
    )
    escribir_csv(
        ROOT / "results" / "validacion_intervalos_pares.csv", cab_pares, fil_pares
    )
    tabla_componentes(
        cab_modelos, fil_modelos, intervalos, ROOT / "results" / "tabla_componentes.tex"
    )
    tabla_aportes(
        cab_aportes, fil_aportes, cab_pares, fil_pares, ROOT / "results" / "tabla_aportes.tex"
    )
    mrr_ganador = num(mejor[cab_modelos.index("MRR")], 4)
    print(f"ganador en validación: {GANADOR_ESPERADO}, MRR {mrr_ganador}")


if __name__ == "__main__":
    main()
