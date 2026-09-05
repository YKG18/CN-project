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


def _load(name: str, path: Path):
    """Import a runner by file path.

    Both experiment folders contain a `run_base.py`; putting both on sys.path
    would make one shadow the other.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

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



def data4cyber_checks() -> None:
    """Data4Cyber base pipeline. Skipped when the processed data is absent."""
    import json

    from common import config as cfg
    proc = cfg.DATA4CYBER_PROCESSED
    if not (proc / "block" / "manifest.json").exists():
        print("\n[Data4Cyber] SKIPPED - run src/common/data/data4cyber_prep.py first")
        return

    print("\n[Data4Cyber] base pipeline")
    d4c_run = _load("d4c_run_base",
                    ROOT / "experiments" / "data4cyber" / "run_base.py")

    def _primary_is_block():
        assert d4c_run.PRIMARY_MODE == "block", (
            f"D3 makes `block` the primary split, got {d4c_run.PRIMARY_MODE}")
        assert set(d4c_run.SPLIT_MODES) == {"block", "scenario"}
        return "block primary, scenario secondary"
    check("Data4Cyber uses the D3 primary split", _primary_is_block)

    def _loads_and_shapes():
        man = json.loads((proc / "block" / "manifest.json").read_text())
        nf = len(man["features"])
        for name in ("train", "validation", "test"):
            d = d4c_run.load_split("block", name)
            assert d["X"].shape[1] == nf, (
                f"{name} has {d['X'].shape[1]} features, manifest says {nf}")
            assert len(d["y"]) == len(d["X"]) == len(d["block"]) == len(d["scenario"])
            assert np.isfinite(d["X"]).all(), f"{name} contains NaN/inf"
        return f"row arrays load with {nf} features and aligned metadata"
    check("Data4Cyber row arrays load consistently", _loads_and_shapes)

    def _non_degenerate():
        rates = {}
        for name in ("train", "validation", "test"):
            d = d4c_run.load_split("block", name)
            assert len(np.unique(d["y"])) == 2, (
                f"{name} split is single-class; no threshold could be selected")
            rates[name] = round(float(d["y"].mean()), 4)
        return f"both classes in every split; attack rates {rates}"
    check("Data4Cyber splits are non-degenerate", _non_degenerate)

    def _no_leaky_features():
        man = json.loads((proc / "block" / "manifest.json").read_text())
        f = man["features"]
        bad = [c for c in f
               if c.endswith((".realtime", ".timestamp"))
               or "attack" in c.lower() or "scenario" in c.lower()
               or c.lower() in ("timestamp", "block")]
        assert not bad, f"leaky/identifying columns in the feature set: {bad}"
        return f"{len(f)} features, no clocks, attacker fields or scenario ids"
    check("Data4Cyber features carry no scenario leakage", _no_leaky_features)

    def _blocks_disjoint():
        seen = {}
        for name in ("train", "validation", "test"):
            d = d4c_run.load_split("block", name)
            for b in np.unique(d["block"]):
                assert b not in seen, (
                    f"block {b} is in both {seen[b]} and {name}")
                seen[b] = name
        return f"{len(seen)} blocks, each entirely on one side"
    check("no Data4Cyber block spans two splits", _blocks_disjoint)

    def _ratio_grid():
        # attack-majority: candidates must stop at the natural ratio
        y_maj_attack = np.array([1] * 610 + [0] * 390)
        c = d4c_run.candidate_ratios(y_maj_attack)
        nat = 610 / 390
        assert abs(c[-1] - round(nat, 4)) < 1e-6, (
            f"last candidate should be the untouched ratio {nat:.4f}, got {c[-1]}")
        assert all(r <= nat + 1e-9 for r in c), f"candidates exceed natural ratio: {c}"
        assert len(set(c)) == len(c), "duplicate candidates"
        # benign-majority still gets a wide grid
        c2 = d4c_run.candidate_ratios(np.array([0] * 940 + [1] * 60))
        assert len(c2) > len(c), "benign-majority grid should be wider"
        return f"attack-majority grid {c}"
    check("undersampling grid adapts to the class balance", _ratio_grid)

    def _end_to_end():
        rec = d4c_run.run_one("block", cfg.SEED, drop_profile=False, verbose=False)
        assert rec["split_role"] == "primary"
        assert rec["threshold"]["selected_on"] == "validation", (
            "threshold must come from validation, never test")
        m = rec["metrics"]["test"]
        for k in ("precision", "recall", "f1", "accuracy", "fpr", "roc_auc",
                  "weighted_f1"):
            assert k in m, f"missing primary metric {k}"
            assert 0.0 <= m[k] <= 1.0, f"{k} out of range: {m[k]}"
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == rec[
            "class_distribution"]["test"]["n"], "confusion counts do not sum to test n"
        assert rec["model"]["eval_metric"] == "logloss"
        assert rec["model"]["trees_after_early_stopping"] < \
            rec["model"]["n_estimators"], "early stopping did not trigger"
        return (f"F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
                f"AUC={m['roc_auc']:.4f}")
    check("Data4Cyber base run completes with valid metrics", _end_to_end)

    def _reproducible():
        a = d4c_run.run_one("block", cfg.SEED, drop_profile=False, verbose=False)
        b_ = d4c_run.run_one("block", cfg.SEED, drop_profile=False, verbose=False)
        assert a["metrics"]["test"] == b_["metrics"]["test"], (
            "two identical runs disagree - the experiment is not reproducible")
        assert a["threshold"]["value"] == b_["threshold"]["value"]
        return "identical metrics across runs at the project seed"
    check("Data4Cyber base run is reproducible", _reproducible)

    def _primary_secondary_separate():
        sec = d4c_run.run_one("scenario", cfg.SEED, drop_profile=False, verbose=False)
        assert sec["split_role"] == "secondary_unseen_attack"
        pri = d4c_run.run_one("block", cfg.SEED, drop_profile=False, verbose=False)
        assert pri["split_policy"] != sec["split_policy"]
        assert pri["metrics"]["test"] != sec["metrics"]["test"], (
            "primary and secondary produced identical metrics - are they really "
            "different splits?")
        return "block and scenario reported as distinct experiments"
    check("primary and secondary evaluations stay separate",
          _primary_secondary_separate)

    def _writes_nothing():
        before = {p_ for p_ in (ROOT / "results").rglob("*") if p_.is_file()}
        d4c_run.run_one("block", cfg.SEED, drop_profile=False, verbose=False)
        after = {p_ for p_ in (ROOT / "results").rglob("*") if p_.is_file()}
        assert before == after, (
            f"run_one() wrote into results/: {sorted(after - before)} - "
            f"result collection belongs to Member 5")
        return "no artifacts created under results/"
    check("the Data4Cyber run writes nothing to results/", _writes_nothing)


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

    def _undersample_inverts():
        # Data4Cyber-shaped: the ATTACK class is the majority. The paper's rule
        # is "reduce the majority, keep all minority", so attacks must be thinned
        # and every benign row kept.
        rng = np.random.default_rng(5)
        y2 = (rng.random(6000) < 0.61).astype(int)
        X2 = rng.normal(size=(6000, 8)).astype(np.float32)
        n_ben = int((y2 == 0).sum())
        _, yr = undersample_majority(X2, y2, 1.0, seed=42)
        assert int((yr == 0).sum()) == n_ben, (
            "benign is the minority here and must be kept in full")
        assert int((yr == 1).sum()) < int((y2 == 1).sum()), (
            "attack is the majority here and must be thinned")
        assert int((yr == 1).sum()) == n_ben, "ratio 1.0 should balance the set"
        return "majority detected from the data, not hard-coded"
    check("undersampling thins whichever class is the majority",
          _undersample_inverts)

    def _ncsrd_direction_unchanged():
        # Regression guard: on a benign-majority set the behaviour must be
        # exactly the historical one (keep all attacks, subsample benign).
        rg = np.random.default_rng(42)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        k = min(len(neg), int(round(len(pos) * 2.0)))
        want = np.concatenate([pos, rg.choice(neg, size=k, replace=False)])
        rg.shuffle(want)
        Xr, yr = undersample_majority(X, y, 2.0, seed=42)
        assert np.array_equal(yr, y[want]) and np.array_equal(Xr, X[want]), (
            "benign-majority resampling changed - NCSRD results would move")
        return "NCSRD (benign-majority) resampling is bit-identical"
    check("generalising undersampling did not move the NCSRD path",
          _ncsrd_direction_unchanged)

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
        run_base = _load("ncsrd_run_base", ROOT / "experiments" / "ncsrd" / "run_base.py")
        assert run_base.FEATURE_SET == "base38", (
            f"the base-paper runner must use base38, not {run_base.FEATURE_SET}")
        assert run_base.CONFIG_NAME == "base_paper"
        assert len(ncsrd_prep.BASE38_FEATURES) == 38
        return "runner is pinned to base38 / base_paper"
    check("base-paper runner uses the 38-feature lineage", _uses_base38)

    def _no_saurabh_mixing():
        run_base = _load("ncsrd_run_base", ROOT / "experiments" / "ncsrd" / "run_base.py")
        b38 = set(ncsrd_prep.BASE38_FEATURES)
        s49 = {"ul_retx_max", "dl_retx_max", "ran_ue_id", "cell_1_cqi", "cell_3_cqi"}
        assert not (b38 & s49), f"saurabh49-only columns leaked into base38: {b38 & s49}"
        assert "saurabh" not in run_base.FEATURE_SET
        return "no saurabh49 columns in the base lineage"
    check("base38 and saurabh49 stay separate", _no_saurabh_mixing)

    def _paper_targets():
        run_base = _load("ncsrd_run_base", ROOT / "experiments" / "ncsrd" / "run_base.py")
        t = run_base.PAPER_TARGETS
        assert t["accuracy"] == 0.996 and t["precision"] == 0.96
        assert t["recall"] == 0.98 and t["f1"] == 0.97
        return "Table III targets recorded, not hard-coded as results"
    check("paper targets are stored for comparison only", _paper_targets)

    data4cyber_checks()

    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for n, e in FAIL:
        print(f"  FAILED: {n} -- {e}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
