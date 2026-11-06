"""Figuras del informe intermedio, extraídas del notebook de validación.

    uv run python scripts/figuras_notebook.py

Las figuras quedan en ``figures/`` (no versionado, ver .gitignore) y el manifiesto
``results/figuras_notebook.csv`` registra de qué celda del notebook sale cada una y
qué afirmación del informe respalda, para que cualquier figura se pueda rastrear y
regenerar. Solo usa la biblioteca estándar.

Las figuras se buscan por un texto que aparece en el código de la celda, no por
número, así que siguen saliendo si el notebook se reordena.
"""
from __future__ import annotations

import base64
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "scripts" / "01_datos_y_validacion.ipynb"

# (nombre del archivo, texto que identifica la celda, qué respalda en el informe)
FIGURAS = [
    (
        "validacion_timeline",
        "Línea de tiempo de los datos",
        "Bloques de entrenamiento reducido, validación y test, y volumen mensual de procesos",
    ),
    (
        "antiguedad_positivo",
        "Aciertos top-10 por antigüedad",
        "El acierto se concentra en ofertas recientes, así que la vigencia domina la señal",
    ),
    (
        "filtro_actividad",
        "Filtro duro sobre LR",
        "MRR y cobertura del filtro por actividad según la ventana, frente al peso suave",
    ),
    (
        "comparacion_modelos",
        "Referencias y mejores variantes",
        "MRR y nDCG@10 de cada modelo y su corte entre candidatos con y sin historial",
    ),
    (
        "contenido_vs_recencia",
        "Temporalidad frente a",
        "El baseline propio de contenido frente a Recencia, antes de la regresión logística",
    ),
]


def imagenes_de_celda(celda: dict) -> list[str]:
    salida = []
    for output in celda.get("outputs", []):
        png = output.get("data", {}).get("image/png")
        if png:
            salida.append("".join(png) if isinstance(png, list) else png)
    return salida


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # consola cp1252 en Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    carpetas = ROOT / "figures"
    carpetas.mkdir(exist_ok=True)
    manifiesto = []
    faltantes = []

    for nombre, marcador, afirmacion in FIGURAS:
        celda = next(
            (
                (numero, c)
                for numero, c in enumerate(nb["cells"], start=1)
                if c.get("cell_type") == "code" and marcador in "".join(c.get("source", []))
            ),
            None,
        )
        if celda is None:
            faltantes.append((nombre, marcador))
            continue
        numero, cuerpo = celda
        imagenes = imagenes_de_celda(cuerpo)
        if not imagenes:
            faltantes.append((nombre, marcador))
            continue
        for orden, png in enumerate(imagenes, start=1):
            sufijo = "" if orden == 1 else f"_{orden}"
            archivo = carpetas / f"fig_{nombre}{sufijo}.png"
            archivo.write_bytes(base64.b64decode(png))
            manifiesto.append(
                {
                    "archivo": archivo.relative_to(ROOT).as_posix(),
                    "celda": numero,
                    "afirmacion": afirmacion,
                    "marcador": marcador,
                }
            )
            print(f"png    -> {archivo.relative_to(ROOT)}  (celda {numero})")

    destino = ROOT / "results" / "figuras_notebook.csv"
    with destino.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo, fieldnames=["archivo", "celda", "afirmacion", "marcador"], lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(manifiesto)
    print(f"csv    -> {destino.relative_to(ROOT)}  ({len(manifiesto)} figuras)")

    if faltantes:
        for nombre, marcador in faltantes:
            print(f"falta: {nombre} (marcador {marcador!r})")
        raise SystemExit("no encontré todas las figuras en el notebook")


if __name__ == "__main__":
    main()
