"""Non-stationary benign traffic — Member 3, faculty direction 5.

    python experiments/ncsrd/run_nonstationary.py
    python experiments/ncsrd/run_nonstationary.py --profile bursty_mmtc

Generates synthetic benign traffic variations (bursty mMTC, periodic URLLC,
gradual drift) and pushes them through three detectors to see which keeps its
false-alarm rate stable when normal traffic changes shape:

    saurabh            static benign correlation baseline
    proposed_frozen    EWMA baseline frozen at the end of training (the default)
    proposed_streaming EWMA baseline that keeps adapting as the stream arrives

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
from proposed.nonstationary import PROFILES, fpr_stability, generate  # noqa: E402
from proposed.proposed_xgboost import ProposedXGBoost  # noqa: E402
from saurabh.saurabh_xgboost import SaurabhXGBoost  # noqa: E402

WINDOW = config.THRESHOLD_WINDOW      # 500
TARGET_WINDOWS = 1000                 # the brief asks for 1000+


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

    tau_s = float(sau.dynamic_threshold.global_threshold_)
    tau_p = float(pro.selected_threshold_)
    print(f"  saurabh tau={tau_s:.4f}   proposed tau={tau_p:.4f}")

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

        print(f"  {'detector':<28}{'windows':>9}{'FPR mean':>11}{'FPR var':>12}"
              f"{'FPR max':>10}{'lat ms/sample':>15}{'EWMA':>7}{'CUSUM':>7}")
        for label, y_pred, ms, upd, drift in detectors:
            st = fpr_stability(y_pred, WINDOW, TARGET_WINDOWS)
            print(f"  {label:<28}{st['n_windows']:>9}{st['fpr_mean']:>11.4f}"
                  f"{st['fpr_var']:>12.6f}{st['fpr_max']:>10.4f}{ms:>15.5f}"
                  f"{('-' if upd is None else upd):>7}"
                  f"{('-' if drift is None else drift):>7}")
            records.append({"profile": name, "description": prof.description,
                            "detector": label, **st,
                            "latency_ms_per_sample": round(ms, 6),
                            "ewma_updates": upd, "cusum_events": drift})

    print(f"\n{'=' * 92}")
    print("All rows are benign, so FPR is the false-alarm rate directly.")
    print("Low FPR variance under a changing profile = the baseline kept up.")
    print("=" * 92)

    if a.save:
        out = ROOT / "results" / "raw" / "proposed_nonstationary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"saved -> {out.relative_to(ROOT)}")
    else:
        print("(no files written; pass --save to store them under results/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
