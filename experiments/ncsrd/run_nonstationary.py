"""Non-stationary benign traffic — Member 3, faculty direction 5.

    python experiments/ncsrd/run_nonstationary.py
    python experiments/ncsrd/run_nonstationary.py --profile bursty_mmtc

Generates synthetic benign traffic variations (bursty mMTC, periodic URLLC,
gradual drift) and pushes them through three detectors to see which keeps its
false-alarm rate stable when normal traffic changes shape:

    saurabh            static benign correlation baseline
    proposed_frozen    EWMA baseline frozen at the end of training (the default)
    proposed_streaming EWMA baseline fed back in as a classifier *feature*
    proposed_adaptive  frozen features, EWMA/CUSUM drive the *threshold* (step B)
    proposed_local_tau step B's per-window variant (brief's literal reading)

Step B was measured and FAILED: the pooled channel is inert and the per-window
variant trades recall 0.9728 -> 0.6047 for its FPR gain. Both rows are kept so
the negative result is reproducible rather than asserted. See
src/proposed/adaptive_threshold.py.

Every row is benign, so every alarm is a false alarm and FPR is measured
directly. This is the experiment that decides whether the online baseline
updater earns its place: a static baseline should drift out of date as benign
traffic changes, an adapting one should not.

Reports FPR mean/variance/max across 1000+ sliding windows plus per-sample
detection latency, as the brief asks.

Results are printed. Nothing is written unless you pass `--save`.
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

from common import config  # noqa: E402
from common.data.ncsrd_adapter import NetworkDataAdapter  # noqa: E402
from proposed.adaptive_threshold import DriftAdaptiveThreshold  # noqa: E402
from proposed.nonstationary import PROFILES, fpr_stability, generate  # noqa: E402
from proposed.proposed_xgboost import ProposedXGBoost  # noqa: E402
from saurabh.saurabh_xgboost import SaurabhXGBoost  # noqa: E402

WINDOW = config.THRESHOLD_WINDOW      # 500
TARGET_WINDOWS = 1000                 # the brief asks for 1000+


def _detection_check(pro, adapt, adapt_local, b):
    """Static vs adaptive threshold on the real labelled test split.

    The profiles above are all benign, so they can only measure false alarms.
    This confirms the adaptive channel does not break normal detection --
    the "not breaking the classifier" half of the success criterion.

    Labels are used to *score* the output, never to produce it.
    """
    from sklearn.metrics import confusion_matrix

    bar = "=" * 92
    print(f"\n{bar}")
    print("Detection on the real NCSRD test split (labels used only to score)")
    print(bar)

    p_test = pro.predict_proba(b.X_test, block_ids=b.test_block)
    tau = float(pro.selected_threshold_)

    outputs = {
        "static tau (P6 default)": (p_test >= tau).astype(int),
        "adaptive tau (step B)": adapt.predict(b.X_test, block_ids=b.test_block),
        "window-local tau": adapt_local.predict(b.X_test, block_ids=b.test_block),
    }
    stats = adapt.summary()

    print(f"  {'decision rule':<26}{'P':>9}{'R':>9}{'F1':>9}{'FPR':>9}"
          f"{'TP':>7}{'FP':>7}{'FN':>7}{'TN':>8}")
    rows = []
    for label, y_pred in outputs.items():
        tn, fp, fn, tp = confusion_matrix(
            b.y_test, y_pred, labels=[0, 1]).ravel()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        print(f"  {label:<26}{pr:>9.4f}{rc:>9.4f}{f1:>9.4f}{fpr:>9.4f}"
              f"{tp:>7}{fp:>7}{fn:>7}{tn:>8}")
        rows.append({"split": "ncsrd_test", "decision_rule": label,
                     "precision": round(float(pr), 4), "recall": round(float(rc), 4),
                     "f1": round(float(f1), 4), "fpr": round(float(fpr), 4),
                     "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})

    print(f"  adaptive threshold moved {stats['n_threshold_changes']} times, "
          f"range [{stats['threshold_min']:.2f}, {stats['threshold_max']:.2f}]")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=[*PROFILES, "all"], default="all")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--strength", type=float, default=0.6,
                   help="peak relative change in the modulated features")
    p.add_argument("--save", action="store_true",
                   help="write results/raw/proposed_nonstationary.json")
    a = p.parse_args(argv)

    print("Loading the shared frozen split ...")
    ad = NetworkDataAdapter.for_feature_set("saurabh49", verbose=False).load_split(
        config.SPLIT_INDEX_FILE)
    b = ad.for_xgboost(balance="class_weight")

    # Benign test rows only: every alarm on this stream is a false alarm.
    benign = b.y_test == 0
    X_benign = b.X_test[benign]
    blocks = b.test_block[benign]
    print(f"  benign test stream: {X_benign.shape[0]:,} rows x {X_benign.shape[1]} features")

    print("Fitting detectors on the shared split ...")
    sau = SaurabhXGBoost(seed=a.seed, use_correlation_graph=True,
                         use_dynamic_threshold=True, use_shap_drift=False)
    sau.fit(b.X_train, b.y_train, b.X_val, b.y_val,
            train_block_ids=b.train_block, val_block_ids=b.val_block)

    pro = ProposedXGBoost.for_config(config_name="P6", seed=a.seed)
    pro.fit(b.X_train, b.y_train, b.X_val, b.y_val,
            train_block_ids=b.train_block, val_block_ids=b.val_block)

    adapt = DriftAdaptiveThreshold(pro)
    adapt_local = DriftAdaptiveThreshold(pro, window_local=True)

    tau_s = float(sau.dynamic_threshold.global_threshold_)
    tau_p = float(pro.selected_threshold_)
    print(f"  saurabh tau={tau_s:.4f}   proposed tau={tau_p:.4f}")

    detection_rows = _detection_check(pro, adapt, adapt_local, b)

    names = list(PROFILES) if a.profile == "all" else [a.profile]
    records = []

    for name in names:
        prof = generate(X_benign, name, seed=a.seed, strength=a.strength)
        X = prof.X.astype(np.float32)
        print(f"\n{'=' * 92}\n{name}  —  {prof.description}\n{'=' * 92}")

        detectors = []

        t0 = time.perf_counter()
        y = sau.predict(X, block_ids=blocks)
        ms = (time.perf_counter() - t0) / len(X) * 1000
        detectors.append(("saurabh (static baseline)", y, ms, None, None))

        t0 = time.perf_counter()
        p_f = pro.predict_proba(X, block_ids=blocks)
        ms_f = (time.perf_counter() - t0) / len(X) * 1000
        detectors.append(("proposed_frozen", (p_f >= tau_p).astype(int), ms_f,
                          None, None))

        t0 = time.perf_counter()
        p_s = pro.predict_proba_streaming(X, block_ids=blocks)
        ms_s = (time.perf_counter() - t0) / len(X) * 1000
        detectors.append(("proposed_streaming", (p_s >= tau_p).astype(int), ms_s,
                          pro.stream_ewma_updates_, pro.stream_drift_events_))

        t0 = time.perf_counter()
        y_a = adapt.predict(X, block_ids=blocks)
        ms_a = (time.perf_counter() - t0) / len(X) * 1000
        tau_stats = adapt.summary()
        detectors.append(("proposed_adaptive_tau", y_a, ms_a,
                          adapt.n_accepted_, adapt.n_change_points_))

        t0 = time.perf_counter()
        y_l = adapt_local.predict(X, block_ids=blocks)
        ms_l = (time.perf_counter() - t0) / len(X) * 1000
        local_stats = adapt_local.summary()
        detectors.append(("proposed_local_tau", y_l, ms_l,
                          adapt_local.n_accepted_, adapt_local.n_change_points_))

        print(f"  {'detector':<28}{'windows':>9}{'FPR mean':>11}{'FPR var':>12}"
              f"{'FPR max':>10}{'lat ms/sample':>15}{'EWMA':>7}{'CUSUM':>7}")
        for label, y_pred, ms, upd, drift in detectors:
            st = fpr_stability(y_pred, WINDOW, TARGET_WINDOWS)
            print(f"  {label:<28}{st['n_windows']:>9}{st['fpr_mean']:>11.4f}"
                  f"{st['fpr_var']:>12.6f}{st['fpr_max']:>10.4f}{ms:>15.5f}"
                  f"{('-' if upd is None else upd):>7}"
                  f"{('-' if drift is None else drift):>7}")
            rec = {"profile": name, "description": prof.description,
                   "detector": label, **st,
                   "latency_ms_per_sample": round(ms, 6),
                   "ewma_updates": upd, "cusum_events": drift}
            stats = {"proposed_adaptive_tau": tau_stats,
                     "proposed_local_tau": local_stats}.get(label)
            if stats is not None:
                rec["threshold"] = {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in stats.items()}
            records.append(rec)

        print(f"  threshold (adaptive): base {tau_stats['base_threshold']:.2f}  "
              f"mean {tau_stats['threshold_mean']:.4f}  "
              f"range [{tau_stats['threshold_min']:.2f}, {tau_stats['threshold_max']:.2f}]  "
              f"var {tau_stats['threshold_var']:.6f}  "
              f"moves {tau_stats['n_threshold_changes']}  "
              f"accepted {tau_stats['n_accepted']}/{tau_stats['n_windows']} windows")
        print(f"  threshold (per-window):        "
              f"mean {local_stats['threshold_mean']:.4f}  "
              f"range [{local_stats['threshold_min']:.2f}, {local_stats['threshold_max']:.2f}]  "
              f"var {local_stats['threshold_var']:.6f}  "
              f"moves {local_stats['n_threshold_changes']}")

    print(f"\n{'=' * 92}")
    print("All rows are benign, so FPR is the false-alarm rate directly.")
    print("Low FPR variance under a changing profile = the baseline kept up.")
    print("STEP B FAILED: proposed_adaptive_tau is inert (identical to static);")
    print("proposed_local_tau buys FPR stability at recall 0.9728 -> 0.6047.")
    print("=" * 92)

    if a.save:
        out = ROOT / "results" / "raw" / "proposed_nonstationary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"profiles": records, "detection": detection_rows},
            indent=2), encoding="utf-8")
        print(f"saved -> {out.relative_to(ROOT)}")
    else:
        print("(no files written; pass --save to store them under results/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
