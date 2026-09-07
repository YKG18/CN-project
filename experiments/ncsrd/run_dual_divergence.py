"""Dual-divergence experiment (P8) — Member 3, step C.

    python experiments/ncsrd/run_dual_divergence.py
    python experiments/ncsrd/run_dual_divergence.py --dataset data4cyber

Answers one question: does an adapting correlation baseline add signal beyond
the fixed-reference divergence the model already has?

Three rows, on the shared project-standard split:

    P6 (reference)     the shipped proposed model, for context
    P8-control         [base | d_frozen]                -- 50 columns
    P8-dual EWMA z     [base | d_frozen | z_adaptive]   -- 51, stateful
    P8-dual lagged     [base | d_frozen | d_lagged]     -- 51, stateless

Each dual row differs from the control in exactly one column, so each gap is
that adaptive feature's doing and nothing else. The control is the comparison
that matters; the P6 row is there only to confirm the control is a fair
stand-in for the shipped model.

The two dual rows separate *adaptiveness* from *statefulness*: both use a
reference that moves with recent traffic, but only the EWMA one accumulates
state across the stream.

Results are printed. Nothing is written unless you pass `--save`, because final
result collection belongs to Member 5. This configuration is experimental and
is deliberately absent from the 3x2 matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import config  # noqa: E402
from proposed.dual_divergence import DualDivergenceXGBoost  # noqa: E402
from proposed.proposed_xgboost import ProposedXGBoost  # noqa: E402


def _bundle(dataset: str):
    """Load the shared frozen split for either dataset."""

    if dataset == "ncsrd":
        from common.data.ncsrd_adapter import NetworkDataAdapter

        adapter = NetworkDataAdapter.for_feature_set(
            "saurabh49", verbose=False
        ).load_split(config.SPLIT_INDEX_FILE)

        return adapter.for_xgboost(balance="class_weight")

    from common.data.data4cyber_adapter import Data4CyberAdapter

    bundle = Data4CyberAdapter(
        split_mode="block", verbose=False
    ).for_xgboost(balance="class_weight")

    # Data4Cyber carries block ids in `meta`; NCSRD exposes them as attributes.
    # Normalise so the rest of this script does not care which dataset it has.
    # Same convention as run_experiment.load_bundle_for_dataset.
    if isinstance(getattr(bundle, "meta", None), dict):
        bundle.train_block = bundle.meta.get("train_block")
        bundle.val_block = bundle.meta.get("val_block")
        bundle.test_block = bundle.meta.get("test_block")

    return bundle


def _score(name: str, y_true, y_pred, p_score, latency_ms, extra=None):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    auc = roc_auc_score(y_true, p_score) if len(set(y_true)) > 1 else float("nan")

    row = {
        "config": name,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "roc_auc": round(float(auc), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "latency_ms_per_sample": round(float(latency_ms), 6),
    }

    if extra:
        row.update(extra)

    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("ncsrd", "data4cyber"), default="ncsrd")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--skip-p6", action="store_true",
                   help="skip the P6 reference row (it is only for context)")
    p.add_argument("--window", type=int, default=None,
                   help="correlation window size; defaults to 500 (ncsrd) / "
                        "120 (data4cyber). The lagged variant needs at least "
                        "two windows per block to carry any signal, so on "
                        "Data4Cyber (120-row blocks) the default makes it "
                        "constant -- pass a smaller value to exercise it.")
    p.add_argument("--save", action="store_true",
                   help="write results/raw/proposed_dual_divergence.json")
    a = p.parse_args(argv)

    print(f"Loading the shared frozen split ({a.dataset}) ...")
    b = _bundle(a.dataset)
    window = a.window or (500 if a.dataset == "ncsrd" else 120)
    print(f"  train {b.X_train.shape}  val {b.X_val.shape}  test {b.X_test.shape}")

    rows = []

    if not a.skip_p6:
        print("\nFitting P6 (reference) ...")
        p6 = ProposedXGBoost.for_config(
            config_name="P6", seed=a.seed, correlation_window_size=window)
        p6.fit(b.X_train, b.y_train, b.X_val, b.y_val,
               train_block_ids=b.train_block, val_block_ids=b.val_block)

        t0 = time.perf_counter()
        p_score = p6.predict_proba(b.X_test, block_ids=b.test_block)
        latency = (time.perf_counter() - t0) / len(b.X_test) * 1000
        y_pred = (p_score >= p6.selected_threshold_).astype(int)
        rows.append(_score("P6 (reference)", b.y_test, y_pred, p_score, latency,
                           {"threshold": round(float(p6.selected_threshold_), 4),
                            "n_features": int(p6.n_model_features_)}))

    variants = (
        ("P8-control (frozen only)", False, "ewma_z"),
        ("P8-dual (EWMA z, stateful)", True, "ewma_z"),
        ("P8-dual (lagged, stateless)", True, "lagged"),
    )

    for label, adaptive, kind in variants:
        print(f"\nFitting {label} ...")
        model = DualDivergenceXGBoost(
            seed=a.seed,
            correlation_window_size=window,
            use_adaptive_feature=adaptive,
            adaptive_kind=kind,
        )
        model.fit(b.X_train, b.y_train, b.X_val, b.y_val,
                  train_block_ids=b.train_block, val_block_ids=b.val_block)

        t0 = time.perf_counter()
        p_score = model.predict_proba(b.X_test, block_ids=b.test_block)
        latency = (time.perf_counter() - t0) / len(b.X_test) * 1000
        y_pred = (p_score >= model.selected_threshold_).astype(int)

        gain = model.feature_gain()
        extra = {
            "threshold": round(float(model.selected_threshold_), 4),
            "n_features": int(model.n_model_features_),
            "gain_frozen": round(gain["frozen_divergence"], 4),
            "ewma_updates": int(model.train_ewma_updates_),
            "cusum_events": int(model.train_drift_events_),
        }

        if adaptive:
            extra["gain_adaptive"] = round(gain["adaptive_divergence"], 4)

        rows.append(_score(label, b.y_test, y_pred, p_score, latency, extra))

    bar = "=" * 104
    print(f"\n{bar}")
    print(f"DUAL DIVERGENCE (step C) — {a.dataset}, project-standard split, "
          f"attack class")
    print(bar)
    print(f"  {'config':<30}{'P':>9}{'R':>9}{'F1':>9}{'FPR':>9}{'AUC':>9}"
          f"{'tau':>7}{'feat':>6}{'gain frz':>10}{'gain adp':>10}")

    for r in rows:
        print(f"  {r['config']:<30}{r['precision']:>9.4f}{r['recall']:>9.4f}"
              f"{r['f1']:>9.4f}{r['fpr']:>9.4f}{r['roc_auc']:>9.4f}"
              f"{r.get('threshold', float('nan')):>7.2f}"
              f"{r.get('n_features', 0):>6}"
              f"{r.get('gain_frozen', float('nan')):>10.4f}"
              f"{r.get('gain_adaptive', float('nan')):>10.4f}")

    print(bar)

    control = next((r for r in rows if r["config"].startswith("P8-control")), None)

    if control:
        for r in rows:
            if not r["config"].startswith("P8-dual"):
                continue
            print(f"  {r['config']:<30} dF1 {r['f1'] - control['f1']:+.4f}   "
                  f"dFPR {r['fpr'] - control['fpr']:+.4f}   "
                  f"dAUC {r['roc_auc'] - control['roc_auc']:+.4f}   "
                  f"gain {r.get('gain_adaptive', 0.0):.4f}")
        print("  (each dual row differs from the control in exactly one "
              "column, so each gap is that adaptive feature)")

    if a.save:
        out = ROOT / "results" / "raw" / "proposed_dual_divergence.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"dataset": a.dataset, "rows": rows}, indent=2),
                       encoding="utf-8")
        print(f"saved -> {out.relative_to(ROOT)}")
    else:
        print("  (no files written; pass --save to store them under results/)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
