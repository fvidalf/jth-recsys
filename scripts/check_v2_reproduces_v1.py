"""Compare the saved outputs of notebook v2 with those of the original, cell by cell.

    uv run python scripts/check_v2_reproduces_v1.py

Every v2 cell carried over records its origin in metadata.jth.v1. Text outputs must match
exactly, apart from the volatile items listed below. Figures aren't compared byte by byte.
Exits with status 1 on any difference.

Gate modes:
  full      all text outputs must match.
  displays  tables and markdown must match; printed lines are ignored.
  skip      not compared. Environment table (cell 3), the run-information table without its
            timings (cell 33, which asserts the same counts), file hashes (cell 46, code files
            differ only by CRLF), and the runner cell (58), whose job moved to a script.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "scripts" / "01_datos_y_validacion.ipynb"
V2 = ROOT / "scripts" / "01_datos_y_validacion_v2.ipynb"

VOLATILE = [
    (re.compile(r"tardó \*\*[\d.,]+ s\*\*"), "tardó X s"),  # timing
    (re.compile(r'"cache_referencias": "[^"]*"'), '"cache_referencias": "X"'),  # machine-specific path
    (re.compile(r"<Figure size [^>]*>"), "<Figure>"),
    (re.compile(r"<Axes[^>]*>"), "<Axes>"),
    (re.compile(r"[ \t]+\n"), "\n"),
]


def clean(text: str) -> str:
    for pattern, new in VOLATILE:
        text = pattern.sub(new, text)
    return text.strip()


def outputs(cell: dict, mode: str) -> list[str]:
    out = []
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream":
            if mode == "full" and o.get("name") == "stdout":
                out.append(clean("".join(o["text"])))
        elif o["output_type"] in ("display_data", "execute_result"):
            data = o.get("data", {})
            if "image/png" in data:
                continue
            for key in ("text/markdown", "text/plain"):
                if key in data:
                    out.append(clean("".join(data[key])))
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
    missing = [i for i, c in enumerate(v1) if c["cell_type"] == "code" and i not in seen]

    print(f"Code cells compared: {compared}. Not compared by design: {skipped}. Missing from v2: {missing}.")
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
