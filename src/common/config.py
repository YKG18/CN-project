"""Frozen, shared experiment settings.

Phase 0 of the implementation plan says the group must agree on ONE seed, ONE
split and ONE set of paths before anyone starts modelling. This module is that
agreement in code, so nobody has to hard-code `C:/Users/<name>/...` again.

Change anything here only after telling the group -- every method's numbers
depend on it.
"""

from __future__ import annotations

from pathlib import Path

# --- paths -------------------------------------------------------------
# Resolved from this file's location, so they work on every machine and from
# any working directory.
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
NCSRD_RAW = DATA / "ncsrd" / "raw"
NCSRD_PROCESSED = DATA / "ncsrd" / "processed"
DATA4CYBER_RAW = DATA / "data4cyber" / "raw"
DATA4CYBER_PROCESSED = DATA / "data4cyber" / "processed"

RESULTS = ROOT / "results"

# --- reproducibility ---------------------------------------------------
SEED = 42

TARGET = "attack_label"

# --- the two NCSRD split policies (PROJECT_DECISIONS.md, D2) -----------
# Every experiment uses one of these two. Nobody rolls their own.

#: REFERENCE REPRODUCTION -- compare against the published paper / Saurabh
#: numbers as faithfully as possible. Optimistic on purpose: a random split
#: over 5-second samples of the same 7 UEs puts near-duplicate rows on both
#: sides, and the scaler was fit on the whole file.
REFERENCE_SPLIT = dict(
    strategy="random", test_size=0.2, val_size=0.0,
    random_state=SEED, stratify=True, refit_scaler=False,
)

#: PROJECT STANDARD -- the headline Base vs Saurabh vs Proposed comparison.
#: Chronological, so no future information leaks backwards, and all scaling is
#: refit on training rows only. Freeze it once, share the index file, and have
#: everyone load_split() it.
PROJECT_STANDARD_SPLIT = dict(
    strategy="temporal", test_size=0.2, val_size=0.1,
    random_state=SEED, stratify=False, refit_scaler=True,
)

#: The frozen split index every method must load for the project-standard runs.
SPLIT_INDEX_FILE = DATA / "ncsrd" / "processed" / "split_project_standard.npz"

# --- windowing conventions (Saurabh's paper, Sections V-VII) -----------
THRESHOLD_WINDOW = 500   # samples per window for threshold adaptation
SHAP_WINDOW = 1000       # samples per window for SHAP drift detection
CORR_WINDOW = 500        # samples per window for the correlation graph
