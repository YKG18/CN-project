"""Base-paper XGBoost methodology applied to Data4Cyber — Member 1, dataset 2.

    python experiments/data4cyber/run_base.py                  # primary (block)
    python experiments/data4cyber/run_base.py --split scenario  # secondary only
    python experiments/data4cyber/run_base.py --split both
    python experiments/data4cyber/run_base.py --no-profile      # sensitivity check

The point here is NOT to reproduce the NCSRD Base-paper score. Data4Cyber is a
smart-grid / ICS testbed, a different cyber-physical domain (D7). The goal is to
apply the same *methodology* after the minimum justified dataset-specific
adaptation, and establish a clean Base baseline for the later 3x2 comparison.

What carries over from the NCSRD base-paper configuration
---------------------------------------------------------
Everything about the model and the protocol, reused from `src/base/base_xgboost.py`:

* undersample the majority class while retaining every minority sample, with
  the ratio selected on validation;
* `scale_pos_weight` derived from the post-resampling counts;
* `logloss` as the evaluation metric;
* early stopping monitored on validation;
* decision threshold chosen on validation, never on test;
* attack-class precision / recall / F1, plus accuracy, FPR, ROC-AUC and
  weighted F1 (D6).

What is necessarily different, and why
--------------------------------------
* **Features.** The base paper's 38 NCSRD columns are 5G radio and bearer
  counters. They simply do not exist here. Data4Cyber's equivalent is its
  electrical telemetry, taken from the shared pipeline exactly as
  `data4cyber_prep.py` produces it -- no separate preprocessing (D7: each
  dataset keeps its own adapter, only the method transfers).
* **Imbalance direction.** NCSRD is 6.28 % attack, so the majority is benign.
  Data4Cyber is ~61 % attack in train, so the *majority is the attack class* and
  the paper's "reduce the majority, keep all minority" rule thins attacks
  instead. `undersample_majority()` decides from the data.
* **Split.** D3's `block` policy, produced by the shared pipeline, not the
  NCSRD block split.
* **No Table III comparison.** There is no published Data4Cyber number to
  reproduce; these are new baseline results.

Results are printed, not written to `results/` -- Member 5 owns result
collection. Use `--save-json PATH` if you want a copy for your own notes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from base.base_xgboost import (  # noqa: E402
    BASE_PAPER_PARAMS, BaseXGBoost, evaluate, select_threshold,
)
from common import config  # noqa: E402

CONFIG_NAME = "base_paper"
DATASET = "data4cyber"
#: D3: `block` is the primary within-dataset experiment; `scenario` holds out
#: whole scenarios and is a secondary novel-attack robustness check only.
SPLIT_MODES = ("block", "scenario")
PRIMARY_MODE = "block"


def load_split(mode: str, name: str) -> dict:
    """Row-level arrays written by `src/common/data/data4cyber_prep.py`."""
    path = config.DATA4CYBER_PROCESSED / mode / f"{name}_rows.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n"
            f"    python src/common/data/data4cyber_prep.py")
    z = np.load(path, allow_pickle=False)
    return {"X": z["X"], "y": z["y"].astype(int), "scenario": z["scenario"],
            "block": z["block"], "timestamp": z["timestamp"]}


def candidate_ratios(y: np.ndarray) -> tuple[float, ...]:
    """Majority:minority ratios worth trying, given the actual class balance.

    Anything at or above the natural ratio is a no-op (the majority class is
    already scarcer than the request), so the grid is clipped and the untouched
    distribution is always included as the last candidate.
    """
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    natural = max(n_pos, n_neg) / max(1, min(n_pos, n_neg))
    grid = [r for r in (1.0, 1.25, 1.5, 2.0, 5.0, 10.0) if r < natural - 1e-9]
    return tuple(grid + [round(natural, 4)])


def tune_ratio(tr, va, seed: int, verbose: bool = True):
    """Select the undersampling ratio on VALIDATION."""
    trials = []
    for ratio in candidate_ratios(tr["y"]):
        m = BaseXGBoost(undersample_ratio=ratio, seed=seed)
        m.fit(tr["X"], tr["y"], va["X"], va["y"])
        p = m.predict_proba(va["X"])
        tau = select_threshold(va["y"], p)
        v = evaluate(va["y"], p, tau)
        trials.append({"ratio": ratio, "val_f1": v["f1"],
                       "val_precision": v["precision"], "val_recall": v["recall"],
                       "threshold": tau, "train_counts": m.train_counts_,
                       "n_trees": m.n_trees})
        if verbose:
            tag = "  (untouched)" if ratio == candidate_ratios(tr["y"])[-1] else ""
            print(f"    ratio {ratio:>7.4f}  val F1={v['f1']:.4f}  "
                  f"P={v['precision']:.4f}  R={v['recall']:.4f}  "
                  f"tau={tau:.4f}  trees={m.n_trees}{tag}")
    best = max(trials, key=lambda t: t["val_f1"])
    if verbose:
        print(f"    -> selected ratio {best['ratio']} "
              f"(validation F1 {best['val_f1']:.4f})")
    return best["ratio"], trials


def run_one(mode: str, seed: int, drop_profile: bool, verbose: bool = True) -> dict:
    role = "PRIMARY — within-dataset" if mode == PRIMARY_MODE else \
           "SECONDARY — unseen-attack robustness, report separately"
    if verbose:
        print(f"\n{'=' * 74}\n{CONFIG_NAME} / {DATASET} / split={mode}\n{role}\n{'=' * 74}")

    tr, va, te = (load_split(mode, s) for s in ("train", "validation", "test"))
    manifest = json.loads((config.DATA4CYBER_PROCESSED / mode / "manifest.json")
                          .read_text(encoding="utf-8"))
    features = list(manifest["features"])

    if drop_profile:
        # Sensitivity check: Profile.* are simulator driving inputs (irradiance,
        # load and PV setpoints) rather than measured telemetry, and they track
        # time of day. Dropping them shows how much the result leans on them.
        keep = np.array([not f.startswith("Profile.") for f in features])
        for d in (tr, va, te):
            d["X"] = d["X"][:, keep]
        features = [f for f, k in zip(features, keep) if k]
        if verbose:
            print(f"  --no-profile: dropped {int((~keep).sum())} Profile.* features")

    if verbose:
        print(f"\n  features: {len(features)}")
        print("  class distribution (attack is the MAJORITY class here)")
        for name, d in (("train", tr), ("val", va), ("test", te)):
            y = d["y"]
            print(f"    {name:<6} n={len(y):>6,}  benign={int((y == 0).sum()):>6,}  "
                  f"attack={int((y == 1).sum()):>6,}  ({y.mean() * 100:.2f}% attack)  "
                  f"blocks={len(np.unique(d['block']))}  "
                  f"scenarios={len(np.unique(d['scenario']))}")

    for name, d in (("train", tr), ("val", va), ("test", te)):
        if len(np.unique(d["y"])) < 2:
            raise RuntimeError(f"{name} split is single-class; cannot proceed")

    if verbose:
        print("\n  selecting the undersampling ratio on validation")
    ratio, trials = tune_ratio(tr, va, seed, verbose)

    if verbose:
        print("\n  final fit")
    t0 = time.perf_counter()
    model = BaseXGBoost(undersample_ratio=ratio, seed=seed)
    model.fit(tr["X"], tr["y"], va["X"], va["y"])
    fit_s = time.perf_counter() - t0

    p_val = model.predict_proba(va["X"])
    tau = select_threshold(va["y"], p_val)
    model.threshold = tau

    t0 = time.perf_counter()
    p_test = model.predict_proba(te["X"])
    infer_ms = (time.perf_counter() - t0) * 1000

    test = evaluate(te["y"], p_test, tau)
    val = evaluate(va["y"], p_val, tau)
    at_half = evaluate(te["y"], p_test, 0.5)

    if verbose:
        print(f"    resampled train: {model.train_counts_}  "
              f"scale_pos_weight={model.scale_pos_weight_:.4f}")
        print(f"    trees={model.n_trees} (early stopping)  fit={fit_s:.1f}s")
        print(f"    threshold selected on validation: {tau:.4f}")
        print(f"\n  TEST @ tau={tau:.4f}   ({mode} split)")
        for k in ("precision", "recall", "f1", "accuracy", "fpr", "roc_auc",
                  "weighted_f1"):
            print(f"    {k:<14} {test[k]:.4f}")
        print(f"    confusion      TN={test['tn']:,} FP={test['fp']:,} "
              f"FN={test['fn']:,} TP={test['tp']:,}")

    return {
        "dataset": DATASET, "method": "base", "config": CONFIG_NAME,
        "split_policy": mode,
        "split_role": "primary" if mode == PRIMARY_MODE else "secondary_unseen_attack",
        "feature_set": f"data4cyber_{len(features)}"
                       + ("_no_profile" if drop_profile else ""),
        "n_features": len(features), "seed": seed,
        "class_distribution": {
            n: {"n": int(len(d["y"])), "benign": int((d["y"] == 0).sum()),
                "attack": int((d["y"] == 1).sum()),
                "attack_rate": round(float(d["y"].mean()), 6),
                "blocks": int(len(np.unique(d["block"]))),
                "scenarios": sorted(set(map(str, d["scenario"])))}
            for n, d in (("train", tr), ("validation", va), ("test", te))},
        "imbalance": {
            "strategy": "random undersampling of the majority class, all "
                        "minority retained (base paper IV.D); the majority "
                        "here is the ATTACK class",
            "ratio_majority_to_minority": ratio,
            "ratio_selected_on": "validation",
            "ratios_tried": [t["ratio"] for t in trials], "trials": trials,
            "resampled_train_counts": model.train_counts_,
            "scale_pos_weight": round(model.scale_pos_weight_, 6)},
        "model": {**BASE_PAPER_PARAMS, "random_state": seed,
                  "early_stopping_rounds": model.early_stopping_rounds,
                  "trees_after_early_stopping": model.n_trees},
        "threshold": {"value": round(tau, 6), "selected_on": "validation",
                      "criterion": "max F1 on the attack class"},
        "metrics": {"test": test, "validation": val, "test_at_0.5": at_half},
        "timing": {"fit_seconds": round(fit_s, 2),
                   "test_inference_ms": round(infer_ms, 3)},
        "source": str((config.DATA4CYBER_PROCESSED / mode).relative_to(ROOT)),
        "features": features,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=[*SPLIT_MODES, "both"], default=PRIMARY_MODE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-profile", action="store_true",
                   help="drop the Profile.* simulator inputs (sensitivity check)")
    p.add_argument("--save-json", type=Path, default=None,
                   help="optional copy of the full record; results/ is Member 5's")
    a = p.parse_args(argv)

    modes = list(SPLIT_MODES) if a.split == "both" else [a.split]
    records = [run_one(m, a.seed, a.no_profile) for m in modes]

    print(f"\n{'=' * 74}\nsummary (attack class)\n{'=' * 74}")
    print(f"  {'split':<10}{'role':<26}{'P':>8}{'R':>8}{'F1':>8}"
          f"{'acc':>8}{'FPR':>8}{'AUC':>8}")
    for r in records:
        m = r["metrics"]["test"]
        print(f"  {r['split_policy']:<10}{r['split_role']:<26}"
              f"{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}"
              f"{m['accuracy']:>8.4f}{m['fpr']:>8.4f}{m['roc_auc']:>8.4f}")
    if len(records) > 1:
        print("\n  The two rows answer different questions and must not be "
              "combined or averaged (D3).")

    if a.save_json:
        a.save_json.parent.mkdir(parents=True, exist_ok=True)
        a.save_json.write_text(json.dumps(records, indent=2, default=float),
                               encoding="utf-8")
        print(f"\n  saved -> {a.save_json}")
    else:
        print("\n  (no files written; pass --save-json PATH if you want a copy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
