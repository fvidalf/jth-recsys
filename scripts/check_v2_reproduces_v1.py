"""Compare the saved outputs of notebook v2 with those of the original, cell by cell.

    uv run python scripts/check_v2_reproduces_v1.py

Every v2 cell carried over records its origin in metadata.jth.v1. Text outputs must match
exactly, apart from the volatile items listed below. Figures aren't compared byte by byte.
Exits with status 1 on any difference.

Gate modes:
  full      all text outputs must match.
  displays  tables and markdown must match; printed lines are ignored.
  no_markdown  printed lines and tables must match. Its markdown output became markdown cells
            with the same numbers (cells 40, 55, 70, 74 and 76).
  skip      not compared. Environment table (cell 3), the run-information table without
            its timings (cell 33, which asserts the same counts) and the findings table
            (cell 116, reworded, with its computed evidence unchanged).

Tables are compared by column name, so moving a column doesn't count as a change.

Code cells of the original left out of v2 on purpose are listed in REMOVED. Only cells 14 and 64
used real data: 14 counted test rows and 64 built the state now in src/jth/activity.py. The invented-data checks moved to tests/test_stage1.py.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts" / "01_datos_y_validacion.ipynb"
V2 = ROOT / "scripts" / "01_datos_y_validacion_v2.ipynb"

REMOVED = {
    14: "invented example and a count of test rows against queries, trimmed by hand",
    64: "daily state code, moved to src/jth/activity.py; its status line asserted nothing",
    27: "invented Recency example (tests/test_stage1.py)",
    29: "invented metric contributions (tests/test_stage1.py)",
    31: "invented evaluator check (tests/test_stage1.py)",
    44: "invented prediction format check (tests/test_stage1.py)",
    46: "file hashes and a status table from a past working session",
    51: "invented tie-break check (tests/test_stage1.py)",
    58: "reference runner, moved to scripts/run_references.py",
}

IGNORED_COLUMNS = {"Orden", "Perfiles con valor"}  # removed from cell 66, added to cell 95

VOLATILE = [
    (re.compile(r'"tipo": "suave"'), '"tipo": "suave fijo"'),  # renamed in v2
    (re.compile(r"Fusión LR-CF"), "Fusión de rangos (RRF)"),  # renamed in v2
    (re.compile(r"LR \+ suave(?! fijo)"), "LR + suave fijo"),  # renamed in v2
    (re.compile(r"<td>suave</td>"), "<td>suave fijo</td>"),  # renamed in v2
    (re.compile(r"Actividad propia"), "Popularidad reciente"),  # renamed in v2
    (re.compile(r'"cache_referencias": "[^"]*"'), '"cache_referencias": "X"'),  # machine-specific path
    (re.compile(r"<Figure size [^>]*>"), "<Figure>"),
    (re.compile(r"<Axes[^>]*>"), "<Axes>"),
    (re.compile(r"[ \t]+\n"), "\n"),
]


def clean(text: str) -> str:
    for pattern, new in VOLATILE:
        text = pattern.sub(new, text)
    return text.strip()


def table_by_name(text: str) -> str:
    """A simple HTML table as text, with columns sorted by name, so order doesn't matter."""
    if "</thead>" not in text:
        return text
    head, body = text.split("</thead>", 1)
    if head.count("<tr") != 1:
        return text
    names = re.findall(r"<th[^>]*>([^<]*)</th>", head)[1:]
    rows = [re.findall(r"<td>([^<]*)</td>", r) for r in re.findall(r"<tr>\s*<th>[^<]*</th>(.*?)</tr>", body, re.S)]
    if not rows or any(len(r) != len(names) for r in rows):
        return text
    columns = sorted((n, tuple(r[k] for r in rows)) for k, n in enumerate(names) if n not in IGNORED_COLUMNS)
    return "\n".join(f"{n}: {' | '.join(v)}" for n, v in columns)


def outputs(cell: dict, mode: str) -> list[str]:
    out = []
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream":
            if mode in ("full", "no_markdown") and o.get("name") == "stdout":
                out.append(clean("".join(o["text"])))
        elif o["output_type"] in ("display_data", "execute_result"):
            data = o.get("data", {})
            if "image/png" in data:
                continue
            if mode == "no_markdown" and "text/markdown" in data:
                continue
            for key in ("text/markdown", "text/html", "text/plain"):  # html: tables without line wrapping
                if key in data:
                    text = clean("".join(data[key]))
                    out.append(table_by_name(text) if key == "text/html" else text)
                    break
        elif o["output_type"] == "error":
            out.append(f"ERROR {o.get('ename')}: {o.get('evalue')}")
    return out


def main() -> int:
    v1 = json.loads(V1.read_text(encoding="utf-8"))["cells"]
    v2 = json.loads(V2.read_text(encoding="utf-8"))["cells"]
    compared, skipped, failures = 0, [], []
    errors = [c["id"] for c in v2 if any(o["output_type"] == "error" for o in c.get("outputs", []))]
    seen = set()
    for cell in v2:
        info = cell.get("metadata", {}).get("jth", {})
        if cell["cell_type"] != "code" or "v1" not in info:
            continue
        i, mode = info["v1"], info.get("gate", "full")
        seen.add(i)
        if mode == "skip":
            skipped.append(i)
            continue
        a, b = outputs(v1[i], mode), outputs(cell, mode)
        compared += 1
        if a != b:
            failures.append((i, a, b))
    missing = [i for i, c in enumerate(v1) if c["cell_type"] == "code" and i not in seen and i not in REMOVED]
    unexpected = sorted(seen & set(REMOVED))

    print(f"Code cells compared: {compared}. Not compared by design: {skipped}. Missing from v2: {missing or 'none'}.")
    print(f"Removed on purpose: {sorted(REMOVED)}.")
    missing += unexpected
    print(f"Cells with errors: {errors or 'none'}.")
    for i, a, b in failures:
        print(f"\n=== cell {i}: outputs differ ({len(a)} in v1, {len(b)} in v2)")
        for x, y in zip(a, b):
            if x != y:
                xl, yl = x.splitlines(), y.splitlines()
                shown = 0
                for lx, ly in zip(xl, yl):
                    if lx != ly and shown < 6:
                        print(f"  v1: {lx[:170]}\n  v2: {ly[:170]}")
                        shown += 1
                if len(xl) != len(yl):
                    print(f"  line counts: v1 {len(xl)}, v2 {len(yl)}")
    if failures or missing or errors:
        print(f"\nGATE FAILED: {len(failures)} cells differ.")
        return 1
    print("\nGATE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
