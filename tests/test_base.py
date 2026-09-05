"""Checks for the base-paper XGBoost configuration (Member 1).

    python tests/test_base.py

Needs no real data — everything runs on a small synthetic set. Verifies the
methodology constraints the base paper and PROJECT_DECISIONS impose, not the
score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "common" / "data"))
sys.path.insert(0, str(ROOT / "experiments" / "ncsrd"))

import ncsrd_prep  # noqa: E402
from base.base_xgboost import (  # noqa: E402
    BASE_PAPER_PARAMS, BaseXGBoost, evaluate, scale_pos_weight_of,
    select_threshold, undersample_majority,
)

PASS, FAIL = [], []


def check(name, fn):
    try:
        detail = fn()
        PASS.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}\n        {e}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}\n        {type(e).__name__}: {e}")


def toy(n=4000, f=38, rate=0.0628, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    X = rng.normal(size=(n, f)).astype(np.float32)
    X[y == 1] += 1.4
    return X, y


def main() -> int:
    print("=" * 70)
    print("Base-paper XGBoost checks")
    print("=" * 70)
    X, y = toy()

    # --- imbalance handling: the paper's "custom sampling strategy" ---------
    def _keeps_all_minority():
        for ratio in (1.0, 2.0, 5.0, 10.0):
            _, yr = undersample_majority(X, y, ratio, seed=42)
            assert int((yr == 1).sum()) == int((y == 1).sum()), (
                f"ratio {ratio} dropped minority samples: "
                f"{int((yr == 1).sum())} != {int((y == 1).sum())}")
        return "all attack rows retained at every ratio"
    check("undersampling keeps every minority sample", _keeps_all_minority)

    def _hits_ratio():
        n_pos = int((y == 1).sum())
        for ratio in (1.0, 2.0, 5.0):
            _, yr = undersample_majority(X, y, ratio, seed=42)
            got = int((yr == 0).sum()) / n_pos
            assert abs(got - ratio) < 0.02, f"ratio {ratio}: got {got:.3f}"
        return "majority:minority ratio honoured"
    check("undersampling reaches the requested ratio", _hits_ratio)

    def _no_smote():
        Xr, yr = undersample_majority(X, y, 1.0, seed=42)
        assert len(yr) < len(y), "undersampling should shrink the training set"
        rows = {tuple(np.round(r, 6)) for r in X}
        assert all(tuple(np.round(r, 6)) in rows for r in Xr), (
            "resampled rows are not all original rows — this must be "
            "undersampling, never SMOTE (the paper uses SMOTE only for "
            "CNN/LSTM/MLP)")
        return "resampled rows are all real, none synthetic"
    check("imbalance handling is undersampling, not SMOTE", _no_smote)

    def _deterministic():
        a = undersample_majority(X, y, 2.0, seed=42)[1]
        b = undersample_majority(X, y, 2.0, seed=42)[1]
        c = undersample_majority(X, y, 2.0, seed=7)[1]
        assert np.array_equal(a, b), "same seed gave different resampling"
        assert not np.array_equal(a, c), "different seeds gave identical resampling"
        return "seeded and reproducible"
    check("undersampling is reproducible from the seed", _deterministic)

    def _spw():
        assert abs(scale_pos_weight_of(np.array([0] * 90 + [1] * 10)) - 9.0) < 1e-9
        return "n_neg / n_pos"
    check("scale_pos_weight matches the XGBoost definition", _spw)

    # --- threshold selection ------------------------------------------------
    def _threshold_on_val():
        rng = np.random.default_rng(1)
        yv = (rng.random(2000) < 0.1).astype(int)
        pv = np.clip(rng.normal(0.2, 0.15, 2000) + yv * 0.55, 0, 1)
        tau = select_threshold(yv, pv)
        assert 0.0 < tau < 1.0, f"threshold out of range: {tau}"
        f1 = evaluate(yv, pv, tau)["f1"]
        for other in np.linspace(0.05, 0.95, 19):
            assert f1 >= evaluate(yv, pv, other)["f1"] - 1e-9, (
                f"tau={tau:.3f} is not F1-optimal; {other:.2f} beats it")
        return f"tau={tau:.4f} is F1-optimal on the given set"
    check("threshold selection maximizes attack-class F1", _threshold_on_val)

    # --- metric definitions (D6) -------------------------------------------
    def _metrics():
        yt = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p = np.array([0.1, 0.2, 0.9, 0.4, 0.8, 0.7, 0.3, 0.95])
        m = evaluate(yt, p, 0.5)
        assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 3), (
            f"confusion wrong: {m['tp'], m['fp'], m['fn'], m['tn']}")
        assert abs(m["precision"] - 3 / 4) < 1e-9
        assert abs(m["recall"] - 3 / 4) < 1e-9
        assert abs(m["fpr"] - 1 / 4) < 1e-9, "FPR must be FP/(FP+TN)"
        assert abs(m["accuracy"] - 6 / 8) < 1e-9
        return "attack-class P/R/F1 and FPR = FP/(FP+TN)"
    check("metrics follow the D6 definitions", _metrics)

    def _reports_both_classes():
        m = evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.8, 0.2]), 0.5)
        for k in ("precision", "recall", "f1", "precision_0", "recall_0", "f1_0",
                  "accuracy", "weighted_f1", "roc_auc", "fpr",
                  "tp", "fp", "fn", "tn", "threshold"):
            assert k in m, f"evaluate() is missing {k}, needed for Table III"
        return "every Table III field present"
    check("evaluate() emits every reported field", _reports_both_classes)

    # --- model wiring -------------------------------------------------------
    def _fit_predict():
        Xtr, ytr = toy(3000, 38, seed=1)
        Xv, yv = toy(800, 38, seed=2)
        Xte, yte = toy(800, 38, seed=3)
        m = BaseXGBoost(undersample_ratio=2.0, seed=42).fit(Xtr, ytr, Xv, yv)
        assert abs(m.scale_pos_weight_ - 2.0) < 0.05, (
            f"scale_pos_weight should follow the resampled counts, got "
            f"{m.scale_pos_weight_}")
        p = m.predict_proba(Xte)
        assert p.shape == (len(yte),), f"predict_proba shape {p.shape}"
        assert p.min() >= 0 and p.max() <= 1, "probabilities out of [0, 1]"
        m.threshold = select_threshold(yv, m.predict_proba(Xv))
        pred = m.predict(Xte)
        assert set(np.unique(pred)) <= {0, 1}, "predict() must return 0/1"
        assert np.array_equal(pred, (p >= m.threshold).astype(int)), (
            "predict() does not honour the configured threshold")
        return f"fit/predict_proba/predict wired; {m.n_trees} trees"
    check("BaseXGBoost exposes the shared model interface", _fit_predict)

    def _early_stopping():
        Xtr, ytr = toy(3000, 38, seed=1)
        Xv, yv = toy(800, 38, seed=2)
        m = BaseXGBoost(seed=42, early_stopping_rounds=10).fit(Xtr, ytr, Xv, yv)
        assert m.n_trees < BASE_PAPER_PARAMS["n_estimators"], (
            "early stopping did not trigger; n_estimators is meant to be an "
            "upper bound")
        return f"stopped at {m.n_trees} of {BASE_PAPER_PARAMS['n_estimators']}"
    check("early stopping on validation is active", _early_stopping)

    def _seeded():
        Xtr, ytr = toy(2000, 38, seed=1)
        Xv, yv = toy(600, 38, seed=2)
        a = BaseXGBoost(seed=42).fit(Xtr, ytr, Xv, yv).predict_proba(Xv)
        b = BaseXGBoost(seed=42).fit(Xtr, ytr, Xv, yv).predict_proba(Xv)
        assert np.allclose(a, b), "same seed produced different predictions"
        return "identical predictions across runs"
    check("the experiment is reproducible from the seed", _seeded)

    def _logloss():
        assert BASE_PAPER_PARAMS["eval_metric"] == "logloss", (
            "the paper trains with logloss as the evaluation metric")
        return "eval_metric=logloss"
    check("training uses the paper's evaluation metric", _logloss)

    # --- lineage separation (D1) -------------------------------------------
    def _uses_base38():
        import run_base
        assert run_base.FEATURE_SET == "base38", (
            f"the base-paper runner must use base38, not {run_base.FEATURE_SET}")
        assert run_base.CONFIG_NAME == "base_paper"
        assert len(ncsrd_prep.BASE38_FEATURES) == 38
        return "runner is pinned to base38 / base_paper"
    check("base-paper runner uses the 38-feature lineage", _uses_base38)

    def _no_saurabh_mixing():
        import run_base
        b38 = set(ncsrd_prep.BASE38_FEATURES)
        s49 = {"ul_retx_max", "dl_retx_max", "ran_ue_id", "cell_1_cqi", "cell_3_cqi"}
        assert not (b38 & s49), f"saurabh49-only columns leaked into base38: {b38 & s49}"
        assert "saurabh" not in run_base.FEATURE_SET
        return "no saurabh49 columns in the base lineage"
    check("base38 and saurabh49 stay separate", _no_saurabh_mixing)

    def _paper_targets():
        import run_base
        t = run_base.PAPER_TARGETS
        assert t["accuracy"] == 0.996 and t["precision"] == 0.96
        assert t["recall"] == 0.98 and t["f1"] == 0.97
        return "Table III targets recorded, not hard-coded as results"
    check("paper targets are stored for comparison only", _paper_targets)

    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for n, e in FAIL:
        print(f"  FAILED: {n} -- {e}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
