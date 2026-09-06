"""Proposed-method deep dive on NCSRD — Member 3.

    python experiments/ncsrd/run_proposed.py                 # P0-P7 ablation
    python experiments/ncsrd/run_proposed.py --config P6     # one configuration
    python experiments/ncsrd/run_proposed.py --dataset data4cyber

Matches the pattern of `run_base.py` and `run_saurabh.py`: the 3x2 matrix cell
lives in `run_experiment.py`, while this script is the per-method deep dive —
here, the P0-P7 ablation from Implementation.md.

It deliberately holds no model logic. Each configuration is executed by
`run_experiment.run_proposed_experiment`, so there is exactly one
implementation of the proposed pipeline and these numbers cannot drift from
the ones the matrix reports.

Results are printed. Nothing is written unless you pass `--save`, because
final result collection belongs to Member 5.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import config  # noqa: E402

CONFIGS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")


def _runner():
    """Load run_experiment.py by path.

    Both experiment folders already contain a `run_base.py`, so importing
    top-level scripts by name is unreliable here; loading by path is explicit.
    """
    path = ROOT / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("cn_run_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cn_run_experiment"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("ncsrd", "data4cyber"), default="ncsrd")
    p.add_argument("--config", choices=CONFIGS, default=None,
                   help="run one configuration (default: the whole P0-P7 ablation)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--save", action="store_true",
                   help="write D6 rows and detail JSON under results/raw/")
    p.add_argument("--output-dir", default="results/raw")
    a = p.parse_args(argv)

    run = _runner()
    names = [a.config] if a.config else list(CONFIGS)
    rows = []

    for i, cfg in enumerate(names, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(names)}] proposed {cfg} x {a.dataset}\n{'=' * 78}")
        cell = argparse.Namespace(
            dataset=a.dataset, method="proposed", config=cfg,
            split_policy="project_standard", seed=a.seed,
            save=a.save, output_dir=a.output_dir)
        try:
            res = run.run_proposed_experiment(cell)
            rows.append(res["d6_row"])
            if a.save:
                run.save_result(cell, res)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append({"config": cfg, "FAILED": True})

    print(f"\n{'=' * 90}")
    print(f"PROPOSED ABLATION — {a.dataset}, project-standard split, attack class")
    print("=" * 90)
    print(f"  {'config':<8}{'P':>9}{'R':>9}{'F1':>9}{'FPR':>9}{'acc':>9}"
          f"{'AUC':>9}{'tau':>8}{'size/KB':>10}")
    for r in rows:
        if r.get("FAILED"):
            print(f"  {r['config']:<8}{'FAILED':>9}")
            continue
        print(f"  {r['config']:<8}{r['precision']:>9.4f}{r['recall']:>9.4f}"
              f"{r['f1']:>9.4f}{r['fpr']:>9.4f}{r['accuracy']:>9.4f}"
              f"{r['roc_auc']:>9.4f}{r['threshold']:>8.3f}{r['model_size_kb']:>10.1f}")
    print("=" * 90)
    ok = sum(1 for r in rows if not r.get("FAILED"))
    print(f"{ok}/{len(rows)} configurations completed.")
    if not a.save:
        print("(no files written; pass --save to store them under results/)")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
