"""Download the JTH dataset from Zenodo into data/raw/ and verify checksums.

    uv run python scripts/download_data.py
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

RECORD = "https://zenodo.org/api/records/21390581/files/{name}/content"
FILES = {  # md5 published by Zenodo
    "candidates.csv": "277c43a8f683726764a76dea1704be67",
    "jobs.csv": "3c586b3e87b95ed4efac3bcce1a68140",
    "history.csv": "c2e0b7626f1c010d3eeb12567a7e37f2",
    "dataset_card.md": "5b54471b4d0d5775f1f282d1871a01ee",
}
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        dest = RAW_DIR / name
        if dest.exists() and md5(dest) == expected:
            print(f"ok    {name}")
            continue
        print(f"fetch {name}")
        urllib.request.urlretrieve(RECORD.format(name=name), dest)
        if md5(dest) != expected:
            dest.unlink()
            raise SystemExit(f"checksum mismatch for {name}")
    print("data ->", RAW_DIR)


if __name__ == "__main__":
    main()
