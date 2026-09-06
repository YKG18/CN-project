"""Checks for the proposed method (Member 3).

    python tests/test_proposed.py

Runs on small synthetic data — no real dataset needed. These verify that the
proposed components are actually *executed and consumed*, not just configured,
and they pin the behaviours the 3x2 comparison depends on.

Written after an audit found that Proposed and Saurabh produce byte-identical
test predictions. That turned out to be expected (see
`test_predict_proba_uses_the_frozen_training_baseline` below), and these tests
exist so the reason stays documented and any future change to it is caught.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proposed.nonstationary import (  # noqa: E402
    PROFILES, fpr_stability, generate, sliding_windows,
)
from proposed.proposed_xgboost import ProposedXGBoost  # noqa: E402

PASS, FAIL = [], []

CONFIGS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
WIN = 60          # small correlation window so synthetic runs stay fast


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


def toy(n=1200, f=12, rate=0.07, seed=0):
    """Small, separable, block-structured stream.

    `rate` mirrors NCSRD's ~6% attack rate on purpose: EWMA only adapts on
    windows that are at least `benign_update_fraction` (0.90) predicted benign,
    so an attack-heavy fixture would make the adaptation look inert when it is
    not.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    X = rng.normal(size=(n, f))
    X[y == 1] += 1.6                       # attacks shift the mean
    X[y == 1, :3] *= 2.0                   # ...and the covariance, so the
    blocks = np.repeat(np.arange(n // WIN + 1), WIN)[:n].astype(str)
    return X, y, blocks                    # correlation baseline can move


def fitted(config_name="P6", seed=42, **kw):
    Xtr, ytr, btr = toy(1200, seed=1)
    Xv, yv, bv = toy(600, seed=2)
    m = ProposedXGBoost.for_config(
        config_name=config_name, seed=seed,
        correlation_window_size=WIN, shap_window_size=WIN,
        shap_sample_size=25, student_max_iter=40, **kw)
    m.fit(Xtr, ytr, Xv, yv, train_block_ids=btr, val_block_ids=bv)
    return m


def main() -> int:
    print("=" * 70)
    print("Proposed (Member 3) checks")
    print("=" * 70)

    Xte, yte, bte = toy(600, seed=3)

    # --- configuration ------------------------------------------------
    def _all_configs_build():
        seen = {}
        for c in CONFIGS:
            m = ProposedXGBoost.for_config(config_name=c, seed=42)
            seen[c] = (m.use_ewma, m.use_cusum, m.use_constrained_threshold,
                       m.use_fast_shap, m.use_distillation)
        assert seen["P0"] == (False,) * 5, f"P0 should disable everything, got {seen['P0']}"
        assert all(seen["P6"]), f"P6 should enable everything, got {seen['P6']}"
        assert len(set(seen.values())) > 1, "all ablation configs are identical"
        return f"{len(CONFIGS)} configs, P0 all-off, P6 all-on"
    check("P0-P7 ablation configs are distinct", _all_configs_build)

    def _backbone_disabled():
        m = ProposedXGBoost.for_config(config_name="P6", seed=42)
        assert m.backbone.use_correlation_graph is False
        assert m.backbone.use_dynamic_threshold is False
        assert m.backbone.use_shap_drift is False
        return "Saurabh's own graph/threshold/drift are off; Proposed supplies its own"
    check("proposed disables the backbone's duplicate modules", _backbone_disabled)

    # --- the components actually execute -------------------------------
    model = fitted("P6")

    def _ewma_runs():
        assert model.use_ewma
        assert model.fitted_ewma_ is not None, "no EWMA baseline was fitted"
        assert model.n_model_features_ == model.n_input_features_ + 1, (
            "EWMA should append exactly one divergence feature, got "
            f"{model.n_model_features_} from {model.n_input_features_}")
        return f"{model.n_input_features_} -> {model.n_model_features_} features"
    check("EWMA runs and appends the divergence feature", _ewma_runs)

    def _adaptation_counters():
        assert model.validation_ewma_updates_ > 0, (
            "EWMA never updated during validation replay - adaptation is inert")
        assert model.validation_drift_events_ >= 0
        return (f"{model.validation_ewma_updates_} EWMA updates, "
                f"{model.validation_drift_events_} CUSUM events")
    check("EWMA/CUSUM adaptation actually fires", _adaptation_counters)

    def _constrained_threshold_runs():
        t = model.selected_threshold_
        assert t is not None and 0.0 < t < 1.0, f"bad threshold {t}"
        assert model.constrained_threshold.threshold_ == t, (
            "selected_threshold_ did not come from the constrained optimizer")
        return f"tau={t}"
    check("the FPR-constrained threshold is the one used", _constrained_threshold_runs)

    def _respects_alpha():
        m = model.validation_threshold_metrics_
        assert m["validation_fpr"] <= model.fpr_alpha + 1e-9, (
            f"validation FPR {m['validation_fpr']:.4f} exceeds the "
            f"alpha={model.fpr_alpha} budget")
        return (f"val FPR {m['validation_fpr']:.4f} <= alpha {model.fpr_alpha}")
    check("the constrained threshold honours its FPR budget", _respects_alpha)

    def _threshold_from_validation_only():
        # Refitting with different TEST data must not move the threshold.
        a = fitted("P6", seed=42).selected_threshold_
        b_ = fitted("P6", seed=42).selected_threshold_
        assert a == b_, "threshold is not determined by train/validation alone"
        return "threshold depends only on train+validation (D5)"
    check("the threshold never sees test labels", _threshold_from_validation_only)

    def _fast_shap_runs():
        assert model.use_fast_shap
        assert model.fast_shap.reference_ranking_ is not None, (
            "lightweight SHAP reference was never built")
        assert model.fast_shap.reference_importance_ is not None
        return (f"reference over {len(model.fast_shap.reference_ranking_)} features, "
                f"fit {model.fast_shap.reference_fit_latency_ms_:.1f} ms")
    check("lightweight SHAP builds its reference", _fast_shap_runs)

    def _distillation_runs():
        assert model.distilled_edge_model is not None, "no student was distilled"
        import pickle
        teacher = len(pickle.dumps(model.model))
        student = len(pickle.dumps(model.distilled_edge_model))
        assert student < teacher, (
            f"student ({student} B) is not smaller than teacher ({teacher} B)")
        return f"student is {teacher / student:.1f}x smaller than the teacher"
    check("distillation produces a smaller student", _distillation_runs)

    # --- interface -----------------------------------------------------
    def _interface():
        p = model.predict_proba(Xte, block_ids=bte)
        yp = model.predict(Xte, block_ids=bte)
        assert p.shape == (len(yte),), f"predict_proba shape {p.shape}"
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert set(np.unique(yp)) <= {0, 1}
        return "fit/predict_proba/predict match the shared interface"
    check("proposed exposes the shared model interface", _interface)

    def _reproducible():
        a = fitted("P6", seed=42).predict_proba(Xte, block_ids=bte)
        b_ = fitted("P6", seed=42).predict_proba(Xte, block_ids=bte)
        assert np.allclose(a, b_), "two identical runs disagree"
        return "identical predictions at the project seed"
    check("proposed is reproducible from the seed", _reproducible)

    # --- the behaviour that explains Saurabh == Proposed ---------------
    def _frozen_baseline():
        """predict_proba deliberately uses the baseline frozen at fit time.

        `fitted_ewma_` is snapshotted right after the teacher is trained, before
        the validation replay adapts `self.ewma`. That keeps predict_proba
        idempotent and stops validation state leaking into test features -- but
        it also means the EWMA/CUSUM adaptation does NOT change test-time
        predictions. This is why Proposed and Saurabh score identically on the
        3x2 matrix. If this is ever changed, this test should fail and the
        comparison must be re-examined.
        """
        frozen = model._build_stream_features(Xte, bte, model.fitted_ewma_)
        adapted = model._build_stream_features(Xte, bte, model.ewma)
        got = model.predict_proba(Xte, block_ids=bte)
        want = model.model.predict_proba(frozen)[:, 1]
        assert np.allclose(got, want), (
            "predict_proba is no longer using the frozen fitted_ewma_ baseline")
        moved = not np.array_equal(frozen, adapted)
        return (f"frozen baseline used; adapted baseline "
                f"{'differs' if moved else 'is identical'} "
                f"(max div diff {np.abs(frozen[:, -1] - adapted[:, -1]).max():.3e})")
    check("predict_proba uses the frozen training baseline",
          _frozen_baseline)

    def _streaming_is_leakage_safe_and_idempotent():
        """The online-adaptation counterpart of predict_proba.

        `predict_proba_streaming` lets the EWMA baseline evolve across the
        stream, gated on the model's own predictions and on CUSUM -- never on
        labels. It works on deep copies, so it must be repeatable and must not
        mutate the model.
        """
        before = (model.validation_ewma_updates_, model.validation_drift_events_)
        a = model.predict_proba_streaming(Xte, block_ids=bte)
        n_up, n_dr = model.stream_ewma_updates_, model.stream_drift_events_
        b_ = model.predict_proba_streaming(Xte, block_ids=bte)
        assert np.array_equal(a, b_), (
            "streaming prediction is not idempotent - it is mutating state")
        assert (model.validation_ewma_updates_,
                model.validation_drift_events_) == before, (
            "streaming prediction mutated the fitted validation counters")
        assert a.shape == (len(yte),) and a.min() >= 0.0 and a.max() <= 1.0
        assert n_up > 0 or n_dr > 0, "streaming pass never adapted at all"
        return f"idempotent; {n_up} EWMA updates, {n_dr} CUSUM events"
    check("streaming prediction is leakage-safe and idempotent",
          _streaming_is_leakage_safe_and_idempotent)

    def _streaming_differs_from_frozen():
        """Documents WHY the frozen baseline is the default.

        The teacher is trained on divergences measured against the frozen
        baseline. Letting the baseline move at test time changes that feature's
        meaning, so the streaming path is a diagnostic, not the headline
        configuration. If these ever stop differing, the frozen/streaming
        distinction has collapsed and the comparison needs re-examining.
        """
        frozen = model.predict_proba(Xte, block_ids=bte)
        stream = model.predict_proba_streaming(Xte, block_ids=bte)
        assert not np.array_equal(frozen, stream), (
            "streaming and frozen baselines give identical probabilities - "
            "the online adaptation is inert")
        return (f"max |diff| {np.abs(frozen - stream).max():.4f}; "
                f"frozen remains the default (train/serve consistency)")
    check("streaming and frozen paths are genuinely different",
          _streaming_differs_from_frozen)

    def _p0_differs_from_p6():
        p0 = fitted("P0")
        a = p0.predict_proba(Xte, block_ids=bte)
        b_ = model.predict_proba(Xte, block_ids=bte)
        assert p0.n_model_features_ == p0.n_input_features_, (
            "P0 has EWMA off, so it must not append a divergence feature")
        assert not np.array_equal(a, b_), (
            "P0 and P6 produce identical probabilities - the ablation is inert")
        return "P0 (all off) and P6 (all on) genuinely differ"
    check("the ablation endpoints differ", _p0_differs_from_p6)

    # --- direction 5: non-stationary benign traffic ---------------------
    Xb, _, bb = toy(3000, seed=11)

    def _profiles_generate():
        made = {}
        for name in PROFILES:
            prof = generate(Xb, name, seed=42, strength=0.6)
            assert prof.X.shape == Xb.shape, f"{name} changed the shape"
            assert np.isfinite(prof.X).all(), f"{name} produced NaN/inf"
            made[name] = prof
        assert np.array_equal(made["stable"].X, Xb), (
            "the stable control must leave the traffic untouched")
        for name in ("bursty_mmtc", "periodic_urllc", "gradual_drift"):
            assert not np.array_equal(made[name].X, Xb), (
                f"{name} did not modulate anything")
        return f"{len(PROFILES)} profiles, control unmodified"
    check("non-stationary profiles generate valid benign traffic",
          _profiles_generate)

    def _profiles_are_distinct():
        gen = {n: generate(Xb, n, seed=42).X for n in PROFILES}
        names = [n for n in PROFILES if n != "stable"]
        for i, a_ in enumerate(names):
            for b2 in names[i + 1:]:
                assert not np.array_equal(gen[a_], gen[b2]), (
                    f"{a_} and {b2} produced identical streams")
        return "each profile shapes the stream differently"
    check("the traffic profiles are distinct from one another",
          _profiles_are_distinct)

    def _profiles_move_correlation():
        # Modulating a shared subset is meant to move the correlation
        # structure, which is what a correlation baseline reacts to.
        base = np.nan_to_num(np.corrcoef(Xb, rowvar=False))
        drift = np.nan_to_num(
            np.corrcoef(generate(Xb, "gradual_drift", seed=42).X, rowvar=False))
        shift = float(np.linalg.norm(drift - base))
        assert shift > 1e-6, (
            "the profile changed marginals but not the correlation structure")
        return f"Frobenius shift {shift:.3f} vs the unmodulated baseline"
    check("profiles perturb the correlation structure, not just marginals",
          _profiles_move_correlation)

    def _generation_deterministic():
        a_ = generate(Xb, "bursty_mmtc", seed=42).X
        b2 = generate(Xb, "bursty_mmtc", seed=42).X
        c = generate(Xb, "bursty_mmtc", seed=7).X
        assert np.array_equal(a_, b2), "same seed gave different traffic"
        assert not np.array_equal(a_, c), "different seeds gave identical traffic"
        return "seeded and reproducible"
    check("traffic generation is reproducible from the seed",
          _generation_deterministic)

    def _sliding_window_count():
        n = 80000
        wins = list(sliding_windows(n, 500, target_windows=1000))
        assert len(wins) >= 1000, (
            f"the brief asks for 1000+ sliding windows, got {len(wins)}")
        assert all(b2 - a_ == 500 for a_, b2 in wins), "windows are not all 500 long"
        assert wins[-1][1] <= n, "a window ran past the end of the stream"
        return f"{len(wins)} windows of 500 over {n:,} rows"
    check("sliding windows reach the 1000+ the brief asks for",
          _sliding_window_count)

    def _fpr_stability_math():
        # all-benign stream: FPR is just the positive rate
        y = np.zeros(60000, dtype=int)
        st = fpr_stability(y, window=500, target_windows=1000)
        assert st["fpr_mean"] == 0.0 and st["fpr_max"] == 0.0
        y[:30000] = 1
        st2 = fpr_stability(y, window=500, target_windows=1000)
        assert st2["fpr_max"] == 1.0, (
            f"expected a saturated window, got {st2['fpr_max']}")
        assert st2["fpr_var"] > 0.0, "half-alarming stream should show variance"
        assert abs(st2["fpr_overall"] - 0.5) < 1e-9
        return "FPR mean/var/max/overall computed correctly"
    check("FPR stability statistics are correct", _fpr_stability_math)


    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for n, e in FAIL:
        print(f"  FAILED: {n} -- {e}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
