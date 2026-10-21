#!/usr/bin/env python3
"""Download the project's public benchmark datasets into the expected data/ paths.

Datasets are intentionally not versioned in Git. Run this script locally when needed.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NCSRD_RECORD = "13900057"
DATA4CYBER_RECORD = "19965384"  # project decisions are pinned to this release
NCSRD_FILENAME = "amari_ue_data_classic_tabular.csv"
DATA4CYBER_FILENAME = "data4cyber_dataset.zip"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "CN-project-dataset-fetcher/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\rDownloading {destination.name}: {done / total:.0%}", end="", flush=True)
    print()


def record_file_url(record_id: str, filename: str) -> str:
    url = f"https://zenodo.org/api/records/{record_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "CN-project-dataset-fetcher/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        record = json.load(response)
    for item in record.get("files", []):
        if item.get("key") == filename:
            return item["links"]["self"]
    names = ", ".join(item.get("key", "?ෙ") for item in record.get("files", []))
    raise RuntimeError(f"Could not find {filename!r} in Zenodo record {record_id}. Available: {names}")


def install_ncsrd(force: bool = False) -> None:
    dest = ROOT / "data/ncsrd/raw" / NCSRD_FILENAME
    if dest.exists() and not force:
        print(f"Already present: {dest}")
        return
    url = record_file_url(NCSRD_RECORD, NCSRD_FILENAME)
    print(f"Downloading NCSRD-DS-5GDDoS from https://zenodo.org/records/{NCSRD_RECORD}")
    download(url, dest)


def install_data4cyber(force: bool = False) -> None:
    root = ROOT / "data/data4cyber/raw"
    marker = list(root.glob("*/dataset.csv"))
    if marker and not force:
        print(f"Data4Cyber scenario CSVs already present ({len(marker)} found); skipping.")
        return
    with tempfile.TemporaryDirectory(prefix="data4cyber-") as temp:
        archive = Path(temp) / DATA4CYBER_FILENAME
        url = record_file_url(DATA4CYBER_RECORD, DATA4CYBER_FILENAME)
        print(f"Downloading Data4Cyber from https://zenodo.org/records/{DATA4CYBER_RECORD}")
        download(url, archive)
        extracted = Path(temp) / "extract"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as zf:
            # Only dataset.csv files are required by this project's data pipeline;
            # omit packet captures and other large ancillary artifacts.
            matches = [n for n in zf.namelist() if Path(n).name == "dataset.csv" and not n.startswith("/") and ".." not in Path(n).parts]
            if not matches:
                raise RuntimeError("No scenario dataset.csv files found in the Data4Cyber archive.")
            for name in matches:
                zf.extract(name, extracted)
        csvs = list(extracted.rglob("dataset.csv"))
        if not csvs:
            raise RuntimeError("Data4Cyber archive extraction produced no dataset.csv files.")
        root.mkdir(parents=True, exist_ok=True)
        for source in csvs:
            # Preserve scenario directory names while avoiding an extra archive-root folder.
            parent = source.parent
            if parent.name.lower() in {"data4cyber", "dataset", "data4cyber_dataset"}:
                continue
            target = root / parent.name / "dataset.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        installed = list(root.glob("*/dataset.csv"))
        if not installed:
            raise RuntimeError("Could not map extracted scenario CSVs into data/data4cyber/raw/<scenario>/dataset.csv")
        print(f"Installed {len(installed)} scenario CSVs under {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncsrd-only", action="store_true", help="download only NCSRD")
    parser.add_argument("--data4cyber-only", action="store_true", help="download only Data4Cyber")
    parser.add_argument("--force", action="store_true", help="replace/re-download existing data")
    args = parser.parse_args()
    try:
        if not args.data4cyber_only:
            install_ncsrd(args.force)
        if not args.ncsrd_only:
            install_data4cyber(args.force)
    except Exception as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        print("Check your internet connection and the Zenodo record/file availability.", file=sys.stderr)
        return 1
    print("Dataset download/setup finished. Review each dataset's citation and terms before redistribution.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
