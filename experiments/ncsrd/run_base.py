"""Run the base-paper XGBoost reproduction on NCSRD-DS-5GDDoS.

    python experiments/ncsrd/run_base.py                    # both split policies
    python experiments/ncsrd/run_base.py --split reference  # just the reproduction

Two split policies, reported separately and never conflated (D2):

`reference`  the Base-paper reproduction. Random stratified 70/10/20, seed 42,
             scaling as `ncsrd_prep` fit it. This is what gets compared against
             the published Table III numbers. Optimistic by construction: rows
             are 5-second samples of the same 7 UEs, so a random split puts
             near-duplicates on both sides.

`project_standard`  the shared comparison setting (D2). Stratified contiguous
             20-minute blocks, 70/10/20, train-only scaling, loaded from the
             frozen index so Base, Saurabh and Proposed use identical rows.
             Stricter; reported alongside, never instead of, the reproduction.

Results are printed. Nothing is written unless you pass `--save`, because
final result collection belongs to Member 5. With `--save` you get:

    results/raw/base_ncsrd_results.csv     one row per run, D6 schema
    results/raw/base_ncsrd_<split>.json    full configuration + metrics
    results/tables/base_ncsrd_table3.md    side-by-side with the paper
    results/plots/base_ncsrd_<split>_*.png confusion matrix, ROC, PR curve
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "common" / "data"))

from base.base_xgboost import (  # noqa: E402
    BASE_PAPER_PARAMS, UNDERSAMPLE_RATIOS, BaseXGBoost, evaluate,
    select_threshold,
)
from common import config  # noqa: E402
from ncsrd_adapter import NetworkDataAdapter  # noqa: E402

FEATURE_SET = "base38"
CONFIG_NAME = "base_paper"

#: Both policies live in `src/common/config.py` so every method uses the same
#: values. `reference` is the Base-paper reproduction (random stratified, seed
#: 42, scaling as prep fit it, plus the validation split the paper's early
#: stopping needs). `project_standard` is the shared stratified-block policy.
SPLIT_POLICIES = {
    "reference": config.BASE_REFERENCE_SPLIT,
    "project_standard": config.PROJECT_STANDARD_SPLIT,
}

#: Table III of the base paper, attack class unless noted.
PAPER_TARGETS = {
    "accuracy": 0.996, "precision_0": 1.00, "recall_0": 1.00, "f1_0": 1.00,
    "precision": 0.96, "recall": 0.98, "f1": 0.97, "weighted_f1": 1.00,
}

RESULT_COLUMNS = [
    "dataset", "method", "config", "feature_set", "split_policy", "seed",
    "precision", "recall", "f1", "fpr", "accuracy", "weighted_f1", "roc_auc",
    "tp", "fp", "fn", "tn", "threshold", "threshold_selected_on",
    "n_train", "n_test", "inference_latency_ms", "model_size_kb", "notes",
]


def tune_undersample_ratio(b, seed: int) -> tuple[float, list[dict]]:
    """Pick the "custom sampling strategy" ratio on VALIDATION, not test."""
    trials = []
    for ratio in UNDERSAMPLE_RATIOS:
        m = BaseXGBoost(undersample_ratio=ratio, seed=seed)
        m.fit(b.X_train, b.y_train, b.X_val, b.y_val)
        p_val = m.predict_proba(b.X_val)
        tau = select_threshold(b.y_val, p_val)
        val = evaluate(b.y_val, p_val, tau)
        trials.append({"ratio": ratio, "val_f1": val["f1"],
                       "val_precision": val["precision"],
                       "val_recall": val["recall"], "threshold": tau,
                       "train_counts": m.train_counts_, "n_trees": m.n_trees})
        print(f"    ratio {ratio:>5.1f} (maj:min)  val F1={val['f1']:.4f}  "
              f"P={val['precision']:.4f}  R={val['recall']:.4f}  "
              f"tau={tau:.4f}  trees={m.n_trees}")
    best = max(trials, key=lambda t: t["val_f1"])
    print(f"    -> selected ratio {best['ratio']} (validation F1 {best['val_f1']:.4f})")
    return best["ratio"], trials


def make_plots(y_test, p_test, tau, out_dir: Path, tag: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                                 RocCurveDisplay, confusion_matrix)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    cm = confusion_matrix(y_test, (p_test >= tau).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["benign", "attack"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Base XGBoost — {tag}\nconfusion matrix @ tau={tau:.4f}")
    fig.tight_layout()
    for name, fig_, plot in [
        ("confusion_matrix", fig, None),
        ("roc_curve", None, lambda a: RocCurveDisplay.from_predictions(
            y_test, p_test, ax=a, name="base_paper")),
        ("pr_curve", None, lambda a: PrecisionRecallDisplay.from_predictions(
            y_test, p_test, ax=a, name="base_paper")),
    ]:
        if plot is not None:
            fig_, a = plt.subplots(figsize=(5, 4))
            plot(a)
            a.set_title(f"Base XGBoost — {tag}")
            a.grid(alpha=.3)
            fig_.tight_layout()
        path = out_dir / f"base_ncsrd_{tag}_{name}.png"
        fig_.savefig(path, dpi=150)
        plt.close(fig_)
        written.append(path.name)
    return written


def run_one(split_name: str, seed: int, quick: bool, save: bool = False) -> dict:
    print(f"\n{'=' * 72}\n{CONFIG_NAME} / {FEATURE_SET} / split={split_name}\n{'=' * 72}")

    ad = NetworkDataAdapter.for_feature_set(FEATURE_SET)
    if split_name == "project_standard" and config.SPLIT_INDEX_FILE.exists():
        # Load the frozen indices so Base, Saurabh and Proposed train and test
        # on byte-identical rows. Rebuild with src/common/data/freeze_split.py.
        ad.load_split(config.SPLIT_INDEX_FILE)
        print(f"  loaded frozen split <- {config.SPLIT_INDEX_FILE.name}")
    else:
        ad.split(**SPLIT_POLICIES[split_name])
    b = ad.for_xgboost(balance="none")        # base paper undersamples, not SMOTE
    assert b.X_train.shape[1] == 38, f"expected 38 features, got {b.X_train.shape[1]}"

    print("\n  class distribution")
    for name, y in (("train", b.y_train), ("val", b.y_val), ("test", b.y_test)):
        print(f"    {name:<6} n={len(y):>7,}  benign={int((y == 0).sum()):>7,}  "
              f"attack={int((y == 1).sum()):>6,}  ({y.mean() * 100:.2f}% attack)")

    # A threshold cannot be chosen on a validation split that holds only one
    # class, and D5 forbids falling back to test labels. Refuse rather than
    # emit meaningless numbers.
    if len(np.unique(b.y_val)) < 2:
        counts = {"benign": int((b.y_val == 0).sum()),
                  "attack": int((b.y_val == 1).sum())}
        msg = ("validation split is single-class "
               f"({counts}), so no threshold can be selected on it")
        print()
        print(f"  *** SKIPPED: {msg}.")
        print("      NCSRD has five short, widely separated attack windows, so a")
        print("      contiguous chronological band can miss them entirely. That is")
        print("      a property of the split policy, not of this model. See the")
        print("      report; not writing results for this split.")
        if save:
            raw = ROOT / "results" / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            (raw / f"base_ncsrd_{split_name}.blocked.json").write_text(
                json.dumps({
                "config": CONFIG_NAME, "feature_set": FEATURE_SET,
                "split_policy": split_name, "status": "blocked", "reason": msg,
                "validation_class_counts": counts,
                "class_distribution": {
                    name: {"n": int(len(y)), "attack": int((y == 1).sum())}
                        for name, y in (("train", b.y_train), ("val", b.y_val),
                                        ("test", b.y_test))},
                }, indent=2), encoding="utf-8")
        return None

    print("\n  selecting the undersampling ratio on validation")
    if quick:
        ratio, trials = 1.0, []
        print("    --quick: skipping the search, using 1:1")
    else:
        ratio, trials = tune_undersample_ratio(b, seed)

    print("\n  final fit")
    t0 = time.perf_counter()
    model = BaseXGBoost(undersample_ratio=ratio, seed=seed)
    model.fit(b.X_train, b.y_train, b.X_val, b.y_val)
    train_s = time.perf_counter() - t0
    print(f"    resampled train: {model.train_counts_}  "
          f"scale_pos_weight={model.scale_pos_weight_:.4f}")
    print(f"    trees={model.n_trees} (early stopping)  fit={train_s:.1f}s")

    # Threshold on VALIDATION only.
    p_val = model.predict_proba(b.X_val)
    tau = select_threshold(b.y_val, p_val)
    model.threshold = tau
    print(f"    threshold selected on validation: {tau:.4f}")

    t0 = time.perf_counter()
    p_test = model.predict_proba(b.X_test)
    infer_ms = (time.perf_counter() - t0) * 1000

    test = evaluate(b.y_test, p_test, tau)
    at_half = evaluate(b.y_test, p_test, 0.5)
    val = evaluate(b.y_val, p_val, tau)

    print(f"\n  TEST @ tau={tau:.4f}")
    for k in ("precision", "recall", "f1", "precision_0", "recall_0", "f1_0",
              "accuracy", "weighted_f1", "roc_auc", "fpr"):
        tgt = PAPER_TARGETS.get(k)
        flag = f"   (paper {tgt:.3f})" if tgt is not None else ""
        print(f"    {k:<14} {test[k]:.4f}{flag}")
    print(f"    confusion      TN={test['tn']:,} FP={test['fp']:,} "
          f"FN={test['fn']:,} TP={test['tp']:,}")

    booster_kb = len(model.model.get_booster().save_raw()) / 1024

    record = {
        "dataset": "ncsrd", "method": "base", "config": CONFIG_NAME,
        "feature_set": FEATURE_SET, "split_policy": split_name, "seed": seed,
        **{k: round(float(test[k]), 6) for k in
           ("precision", "recall", "f1", "fpr", "accuracy", "weighted_f1", "roc_auc")},
        "tp": test["tp"], "fp": test["fp"], "fn": test["fn"], "tn": test["tn"],
        "threshold": round(tau, 6), "threshold_selected_on": "validation",
        "n_train": int(len(b.y_train)), "n_test": int(len(b.y_test)),
        "inference_latency_ms": round(infer_ms, 3),
        "model_size_kb": round(booster_kb, 1),
        "notes": f"undersample_ratio={ratio}; trees={model.n_trees}",
    }

    detail = {
        "config": CONFIG_NAME,
        "feature_set": FEATURE_SET,
        "n_features": int(b.X_train.shape[1]),
        "feature_names": list(b.feature_names),
        "split_policy": split_name,
        "split": dict(SPLIT_POLICIES[split_name]),
        "split_source": ("frozen: " + config.SPLIT_INDEX_FILE.name
                         if split_name == "project_standard"
                         and config.SPLIT_INDEX_FILE.exists() else "built in-run"),
        "seed": seed,
        "class_distribution": {
            name: {"n": int(len(y)), "benign": int((y == 0).sum()),
                   "attack": int((y == 1).sum()),
                   "attack_rate": round(float(y.mean()), 6)}
            for name, y in (("train", b.y_train), ("val", b.y_val), ("test", b.y_test))
        },
        "n_test_blocks": (int(len(np.unique(b.test_block)))
                          if b.test_block is not None else None),
        "imbalance": {
            "strategy": "random undersampling of the majority class, all "
                        "minority retained (base paper IV.D)",
            "undersample_ratio_majority_to_minority": ratio,
            "ratio_selected_on": "validation",
            "ratios_tried": [t["ratio"] for t in trials] or [ratio],
            "trials": trials,
            "resampled_train_counts": model.train_counts_,
            "scale_pos_weight": round(model.scale_pos_weight_, 6),
        },
        "model": {**BASE_PAPER_PARAMS, "random_state": seed,
                  "early_stopping_rounds": model.early_stopping_rounds,
                  "trees_after_early_stopping": model.n_trees,
                  "model_size_kb": round(booster_kb, 1)},
        "threshold": {"value": round(tau, 6), "selected_on": "validation",
                      "criterion": "max F1 on the attack class"},
        "metrics": {"test": test, "test_at_0.5": at_half, "validation": val},
        "paper_targets": PAPER_TARGETS,
        "timing": {"fit_seconds": round(train_s, 2),
                   "test_inference_ms": round(infer_ms, 3)},
    }

    if save:
        detail["plots"] = make_plots(b.y_test, p_test, tau,
                                     ROOT / "results" / "plots", split_name)
        raw = ROOT / "results" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        # This split produced real results, so drop any stale "blocked" marker.
        (raw / f"base_ncsrd_{split_name}.blocked.json").unlink(missing_ok=True)
        (raw / f"base_ncsrd_{split_name}.json").write_text(
            json.dumps(detail, indent=2, default=float), encoding="utf-8")
    record["_detail"] = detail
    return record


def write_csv(records: list[dict]) -> Path:
    path = ROOT / "results" / "raw" / "base_ncsrd_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {(r["config"], r["split_policy"]) for r in records}
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f)
                        if (r.get("config"), r.get("split_policy")) not in keep]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for r in existing + records:
            w.writerow({k: r.get(k, "") for k in RESULT_COLUMNS})
    return path


def write_table(records: list[dict]) -> Path:
    ref = next((r for r in records if r["split_policy"] == "reference"), None)
    detail = {r["split_policy"]: r["_detail"] for r in records if "_detail" in r}

    rows = [("Accuracy", "accuracy"), ("Precision (0)", "precision_0"),
            ("Recall (0)", "recall_0"), ("F1 (0)", "f1_0"),
            ("Precision (1)", "precision"), ("Recall (1)", "recall"),
            ("F1 (1)", "f1"), ("Weighted F1", "weighted_f1")]

    lines = [
        "# Base paper XGBoost — NCSRD reproduction",
        "",
        "`base_paper` configuration, `base38` feature set. Generated by",
        "`experiments/ncsrd/run_base.py`; do not edit by hand.",
        "",
        "## Table III comparison",
        "",
        "**reference** is the Base-paper reproduction and the only column to",
        "compare against the paper: random stratified 70/10/20, seed 42, scaling",
        "as `ncsrd_prep` fit it.",
        "",
        "**project_standard** is the shared D2 policy used for the Base vs Saurabh",
        "vs Proposed comparison: stratified contiguous 20-minute blocks, 70/10/20,",
        "train-only scaling, loaded from the frozen index. It is reported",
        "alongside — never instead of — the reproduction. The gap between the two",
        "columns is what the random split's near-duplicate rows were worth.",
        "",
        "| Metric | Paper | reference | project_standard |",
        "|---|---:|---:|---:|",
    ]
    for label, key in rows:
        cells = []
        for sp in ("reference", "project_standard"):
            d = detail.get(sp)
            cells.append(f"{d['metrics']['test'][key]:.4f}" if d else "—")
        lines.append(f"| {label} | {PAPER_TARGETS[key]:.3f} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Secondary metrics", "",
              "| Metric | reference | project_standard |", "|---|---:|---:|"]
    for label, key in [("FPR", "fpr"), ("ROC-AUC", "roc_auc"),
                       ("Threshold", "threshold")]:
        cells = []
        for sp in ("reference", "project_standard"):
            d = detail.get(sp)
            cells.append(f"{d['metrics']['test'][key]:.4f}" if d else "—")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")

    lines += ["", "## Confusion matrices (test)", "",
              "| Split | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|"]
    for sp in ("reference", "project_standard"):
        d = detail.get(sp)
        if d:
            m = d["metrics"]["test"]
            lines.append(f"| {sp} | {m['tn']:,} | {m['fp']:,} | "
                         f"{m['fn']:,} | {m['tp']:,} |")

    if ref:
        lines += ["", "## Run configuration (reference)", ""]
        d = detail.get("reference", {})
        imb = d.get("imbalance", {})
        lines += [
            f"* features: {d.get('n_features')} (`base38`)",
            f"* split: random stratified 70/10/20, seed {ref['seed']}, "
            f"scaling as prep fit it",
            f"* imbalance: majority undersampled to {imb.get('undersample_ratio_majority_to_minority')}:1, "
            f"all {imb.get('resampled_train_counts', {}).get('attack', '?'):,} attack rows kept; "
            f"scale_pos_weight={imb.get('scale_pos_weight')}",
            f"* trees after early stopping: "
            f"{d.get('model', {}).get('trees_after_early_stopping')}",
            f"* threshold {ref['threshold']} selected on **validation** "
            f"(max attack-class F1)",
        ]

    ps = next((r for r in records if r["split_policy"] == "project_standard"), None)
    if ps:
        d = detail.get("project_standard", {})
        imb = d.get("imbalance", {})
        cd = d.get("class_distribution", {})
        lines += ["", "## Run configuration (project_standard)", "",
                  f"* features: {d.get('n_features')} (`base38`) — same 38 as above",
                  f"* split: stratified contiguous "
                  f"{d.get('split', {}).get('block_minutes')}-minute blocks, "
                  f"70/10/20, seed {ps['seed']}, train-only scaling",
                  f"* split source: {d.get('split_source')}",
                  f"* test blocks: {d.get('n_test_blocks')}",
                  "* class balance: " + ", ".join(
                      f"{k} {v['attack_rate'] * 100:.2f}% attack"
                      for k, v in cd.items()),
                  f"* imbalance: majority undersampled to "
                  f"{imb.get('undersample_ratio_majority_to_minority')}:1; "
                  f"scale_pos_weight={imb.get('scale_pos_weight')}",
                  f"* trees after early stopping: "
                  f"{d.get('model', {}).get('trees_after_early_stopping')}",
                  f"* threshold {ps['threshold']} selected on **validation**",
                  ]
    lines.append("")

    path = ROOT / "results" / "tables" / "base_ncsrd_table3.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=[*SPLIT_POLICIES, "both"], default="both")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--quick", action="store_true",
                   help="skip the undersampling-ratio search (smoke test)")
    p.add_argument("--save", action="store_true",
                   help="write CSV, JSON, table and plots under results/ "
                        "(off by default: Member 5 owns result collection)")
    a = p.parse_args(argv)

    data = config.NCSRD_PROCESSED / f"ncsrd_{FEATURE_SET}.csv"
    if not data.exists():
        print(f"error: {data} not found.\n"
              f"Build it first:  python src/common/data/ncsrd_prep.py",
              file=sys.stderr)
        return 2

    names = list(SPLIT_POLICIES) if a.split == "both" else [a.split]
    records = [r for r in (run_one(n, a.seed, a.quick, a.save) for n in names) if r]
    if not records:
        print("\nno usable results produced", file=sys.stderr)
        return 1

    print(f"\n{'=' * 72}\nsummary (attack class)\n{'=' * 72}")
    print(f"  {'split':<18}{'P':>9}{'R':>9}{'F1':>9}{'acc':>9}{'FPR':>9}{'AUC':>9}")
    for r in records:
        print(f"  {r['split_policy']:<18}{r['precision']:>9.4f}{r['recall']:>9.4f}"
              f"{r['f1']:>9.4f}{r['accuracy']:>9.4f}{r['fpr']:>9.4f}"
              f"{r['roc_auc']:>9.4f}")

    if a.save:
        csv_path = write_csv(records)
        tbl_path = write_table(records)
        print(f"\nresults -> {csv_path.relative_to(ROOT)}")
        print(f"table   -> {tbl_path.relative_to(ROOT)}")
        print(f"details -> results/raw/base_ncsrd_<split>.json")
    else:
        print("\n  (no files written; pass --save if you want them under results/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
