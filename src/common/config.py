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

#: REFERENCE REPRODUCTION + VALIDATION -- the same family as REFERENCE_SPLIT,
#: with 10% held out for validation. Methods that need a validation set (early
#: stopping, threshold selection) use this instead: the base paper explicitly
#: divides into train/validation/test, while Saurabh's notebook does not.
#: Values are otherwise identical to REFERENCE_SPLIT, so reproduction fidelity
#: is unchanged.
BASE_REFERENCE_SPLIT = dict(
    strategy="random", test_size=0.2, val_size=0.1,
    random_state=SEED, stratify=True, refit_scaler=False,
)

#: Length of one contiguous time block for the project-standard split.
#: Chosen so a block comfortably exceeds the longest evaluation window any
#: method uses (SHAP_WINDOW = 1000 samples): at 20 minutes the median block
#: holds ~1,400 rows. Shorter blocks (10 min, ~700 rows) cannot hold a SHAP
#: window; longer ones (30 min) leave too few attack blocks to spread across
#: three splits. Do not change without re-checking both properties.
BLOCK_MINUTES = 20

#: PROJECT STANDARD -- the headline Base vs Saurabh vs Proposed comparison.
#:
#: Stratified contiguous time blocks. The capture is cut into BLOCK_MINUTES
#: blocks and WHOLE blocks are dealt to train/val/test, stratified by each
#: block's attack content. Rows never move individually, so near-duplicate
#: neighbours stay on the same side; and every split gets both classes, which
#: a plain chronological 70/10/20 slice does NOT -- NCSRD's five attack
#: windows are short and far apart, so the middle 10% band is entirely benign
#: and no threshold can be selected on it.
#:
#: All scaling is refit on training rows only. Freeze it once, share the index
#: file, and have everyone load_split() it.
PROJECT_STANDARD_SPLIT = dict(
    strategy="block", test_size=0.2, val_size=0.1,
    random_state=SEED, stratify=True, refit_scaler=True,
    block_minutes=BLOCK_MINUTES,
)

#: The frozen split index every method must load for the project-standard runs.
#: Regenerate it with `python src/common/data/freeze_split.py` if it is missing.
SPLIT_INDEX_FILE = DATA / "ncsrd" / "processed" / "split_project_standard.npz"

# --- windowing conventions (Saurabh's paper, Sections V-VII) -----------
# Under the project-standard split these windows must be built WITHIN one
# block (`bundle.test_block`), never across two -- blocks are scattered in
# time, so a window spanning a boundary would splice unrelated moments
# together. Skip blocks shorter than the window, as data4cyber_prep does.
THRESHOLD_WINDOW = 500   # samples per window for threshold adaptation
SHAP_WINDOW = 1000       # samples per window for SHAP drift detection
CORR_WINDOW = 500        # samples per window for the correlation graph
