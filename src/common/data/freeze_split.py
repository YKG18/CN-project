"""Freeze the NCSRD project-standard split so every method trains on identical rows.

    python src/common/data/freeze_split.py          # write it
    python src/common/data/freeze_split.py --check  # verify without writing

Writes `config.SPLIT_INDEX_FILE`. Run it once; commit nothing (the file lives
under the git-ignored `data/`), and everyone regenerates it — the split is
fully determined by `config.PROJECT_STANDARD_SPLIT` and `config.SEED`, so every
machine produces byte-identical indices.

Base, Saurabh and Proposed must all load it rather than calling `split()`
themselves:

    ad = NetworkDataAdapter.for_feature_set("saurabh49").load_split(
             config.SPLIT_INDEX_FILE)

`base38` and `saurabh49` have identical rows in identical order, so one index
file covers both feature sets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config  # noqa: E402
from common.data.ncsrd_adapter import NetworkDataAdapter  # noqa: E402


def build(feature_set: str = "saurabh49", verbose: bool = True) -> NetworkDataAdapter:
    ad = NetworkDataAdapter.for_feature_set(feature_set, verbose=verbose)
    ad.split(**config.PROJECT_STANDARD_SPLIT)
    return ad


def summarize(ad: NetworkDataAdapter) -> None:
    sp = ad._split
    print(f"\n  policy: {config.PROJECT_STANDARD_SPLIT}")
    print(f"  {'split':<7}{'rows':>10}{'attack':>9}{'rate':>9}{'blocks':>9}")
    for name in ("train", "val", "test"):
        idx = sp[name]
        y = ad.y[idx]
        nb = len(np.unique(ad._blocks[idx])) if ad._blocks is not None else 0
        print(f"  {name:<7}{len(y):>10,}{int(y.sum()):>9,}{y.mean() * 100:>8.2f}%{nb:>9}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="rebuild and compare against the stored file")
    p.add_argument("--feature-set", default="saurabh49",
                   help="which CSV to read rows from; both give the same indices")
    a = p.parse_args(argv)

    data = config.NCSRD_PROCESSED / f"ncsrd_{a.feature_set}.csv"
    if not data.exists():
        print(f"error: {data} not found. Run src/common/data/ncsrd_prep.py first.",
              file=sys.stderr)
        return 2

    ad = build(a.feature_set, verbose=not a.check)
    summarize(ad)

    if a.check:
        if not config.SPLIT_INDEX_FILE.exists():
            print(f"\n  MISSING: {config.SPLIT_INDEX_FILE}", file=sys.stderr)
            return 1
        z = np.load(config.SPLIT_INDEX_FILE, allow_pickle=False)
        same = all(np.array_equal(z[k], ad._split[k])
                   for k in ("train", "val", "test"))
        print(f"\n  stored file matches a fresh build: {same}")
        return 0 if same else 1

    ad.save_split(config.SPLIT_INDEX_FILE)
    print(f"\n  frozen -> {config.SPLIT_INDEX_FILE}")
    print("  every method should now load_split() this file, not call split().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
