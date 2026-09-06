"""Synthetic non-stationary benign traffic — Member 3, faculty direction 5.

The brief asks us to "generate synthetic 'normal' traffic variations (e.g.
bursty mMTC, periodic URLLC) to challenge the online baseline updater", then
report FPR stability across 1000+ sliding windows.

Everything produced here is **benign**. That is the point: if the detector
raises an alarm on this traffic it is a false alarm by construction, so the
false-positive rate is measured directly with no labelling ambiguity.

The profiles modulate a *shared subset* of features by a common time-varying
factor. That is deliberate — scaling features independently would only move
their marginals, whereas moving them together changes the correlation structure,
which is exactly what the EWMA baseline tracks and therefore what a
correlation-based detector reacts to.

Features are already z-scored by `ncsrd_prep`, so a multiplicative factor scales
each row's deviation from the training mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Profile names. `stable` is the control: benign traffic left untouched, so any
#: FPR it shows is the detector's own baseline error rate.
PROFILES = ("stable", "bursty_mmtc", "periodic_urllc", "gradual_drift")


@dataclass
class TrafficProfile:
    """One synthetic benign stream plus the factor that generated it."""

    name: str
    X: np.ndarray
    #: Per-row modulation factor, for plotting/inspection.
    factor: np.ndarray
    #: Column indices that were modulated.
    columns: np.ndarray
    description: str


def _affected_columns(n_features: int, fraction: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Pick the feature subset that moves together."""
    k = max(2, int(round(n_features * fraction)))
    return np.sort(rng.choice(n_features, size=min(k, n_features), replace=False))


def generate(X_benign: np.ndarray,
             profile: str,
             seed: int = 42,
             strength: float = 0.6,
             affected_fraction: float = 0.4,
             burst_len: int = 300,
             burst_gap: int = 900,
             period: int = 1500) -> TrafficProfile:
    """Return a benign stream reshaped by one non-stationary profile.

    Parameters
    ----------
    X_benign
        Benign rows only, in time order.
    profile
        One of `PROFILES`.
    strength
        Peak relative change in the modulated columns. 0.6 means the affected
        features swing up to 60% above their normal deviation.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")

    X = np.asarray(X_benign, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X_benign must be a non-empty 2-D array")

    n, f = X.shape
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    if profile == "stable":
        return TrafficProfile(
            name=profile, X=X.copy(), factor=np.ones(n),
            columns=np.array([], dtype=int),
            description="unmodified benign traffic (control)")

    cols = _affected_columns(f, affected_fraction, rng)

    if profile == "bursty_mmtc":
        # Many short, high-amplitude bursts: a swarm of MTC devices waking,
        # transmitting briefly, and going quiet again.
        factor = np.ones(n)
        cycle = burst_len + burst_gap
        for start in range(0, n, cycle):
            end = min(start + burst_len, n)
            factor[start:end] = 1.0 + strength
        desc = f"bursts of {burst_len} rows every {cycle} rows"

    elif profile == "periodic_urllc":
        # Smooth periodic load, e.g. a control loop running at a fixed rate.
        factor = 1.0 + strength * np.sin(2.0 * np.pi * t / period)
        desc = f"sinusoidal modulation, period {period} rows"

    else:  # gradual_drift
        # Slow monotonic ramp: the "daily usage peak" the brief calls out as
        # the reason a static correlation baseline goes stale.
        factor = 1.0 + strength * (t / max(n - 1, 1))
        desc = "monotonic ramp across the whole stream"

    out = X.copy()
    out[:, cols] = out[:, cols] * factor[:, None]

    return TrafficProfile(name=profile, X=out, factor=factor, columns=cols,
                          description=f"{desc}; {len(cols)}/{f} features modulated")


def sliding_windows(n_rows: int, window: int, target_windows: int = 1000):
    """Yield `(start, end)` for >= `target_windows` overlapping windows.

    The brief asks for FPR stability across 1000+ sliding windows, so the stride
    is derived from the stream length rather than fixed.
    """
    if n_rows < window:
        return
    stride = max(1, (n_rows - window) // max(target_windows - 1, 1))
    for start in range(0, n_rows - window + 1, stride):
        yield start, start + window


def fpr_stability(y_pred: np.ndarray, window: int = 500,
                  target_windows: int = 1000) -> dict:
    """FPR mean/variance/max over sliding windows of an all-benign stream.

    Every positive is a false alarm here, so window FPR is just the positive
    rate. Returns the fields the brief asks for.
    """
    y_pred = np.asarray(y_pred).ravel().astype(int)
    fprs = [float(y_pred[a:b].mean()) for a, b in
            sliding_windows(len(y_pred), window, target_windows)]
    if not fprs:
        return {"n_windows": 0, "fpr_mean": 0.0, "fpr_var": 0.0,
                "fpr_max": 0.0, "fpr_overall": 0.0}
    arr = np.asarray(fprs)
    return {
        "n_windows": int(arr.size),
        "fpr_mean": round(float(arr.mean()), 6),
        "fpr_var": round(float(arr.var()), 8),
        "fpr_max": round(float(arr.max()), 6),
        "fpr_overall": round(float(y_pred.mean()), 6),
    }
