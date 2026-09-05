"""Sanity checks for the shared data pipeline.

Run this after changing anything under `src/common/data/`:

    python tests/test_pipeline.py

It needs no real data. The NCSRD checks build a small synthetic raw CSV that
uses the **real 80 column names** of `amari_ue_data_classic_tabular.csv` (taken
from the outputs saved in `docs/Saurabh's Project Files/01_data_understanding.ipynb`),
so feature selection is verified against the true schema. The Data4Cyber checks
run only if `data/data4cyber/raw/` is populated, and are skipped otherwise.

Plain asserts and a `main()`; no pytest required, though pytest will collect it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "common" / "data"))

import ncsrd_prep                                    # noqa: E402
from ncsrd_adapter import NetworkDataAdapter         # noqa: E402

# --- ground truth ----------------------------------------------------------

#: The real raw schema (80 columns), from notebook 01's saved output.
RAW_COLUMNS = """_time imeisv 5g_tmsi amf_ue_id bearer_0_apn bearer_0_dl_total_bytes bearer_0_ip
bearer_0_ipv6 bearer_0_pdu_session_id bearer_0_qos_flow_id bearer_0_sst bearer_0_ul_total_bytes
bearer_1_apn bearer_1_dl_total_bytes bearer_1_ip bearer_1_pdu_session_id bearer_1_qos_flow_id
bearer_1_sst bearer_1_ul_total_bytes cell_1_cell_id cell_1_cqi cell_1_dl_bitrate cell_1_dl_err
cell_1_dl_mcs cell_1_dl_retx cell_1_dl_tx cell_1_epre cell_1_initial_ta cell_1_p_ue
cell_1_pusch_snr cell_1_ri cell_1_turbo_decoder_avg cell_1_turbo_decoder_max
cell_1_turbo_decoder_min cell_1_ul_bitrate cell_1_ul_err cell_1_ul_mcs cell_1_ul_n_layer
cell_1_ul_path_loss cell_1_ul_phr cell_1_ul_rank cell_1_ul_retx cell_1_ul_tx dl_bitrate ran_id
ran_plmn ran_ue_id registered rnti t3512 tac tac_plmn ue_aggregate_max_bitrate_dl
ue_aggregate_max_bitrate_ul ul_bitrate cell_3_cell_id cell_3_cqi cell_3_dl_bitrate cell_3_dl_err
cell_3_dl_mcs cell_3_dl_retx cell_3_dl_tx cell_3_epre cell_3_initial_ta cell_3_pusch_snr cell_3_ri
cell_3_turbo_decoder_avg cell_3_turbo_decoder_max cell_3_turbo_decoder_min cell_3_ul_bitrate
cell_3_ul_err cell_3_ul_rank cell_3_ul_retx cell_3_ul_tx bearer_1_ipv6 cell_3_p_ue cell_3_ul_mcs
cell_3_ul_n_layer cell_3_ul_path_loss cell_3_ul_phr""".split()

#: The 49 features notebook 04 actually trains on, from its saved output.
SAURABH49 = """bearer_0_dl_total_bytes bearer_0_pdu_session_id bearer_0_ul_total_bytes
bearer_1_dl_total_bytes bearer_1_pdu_session_id bearer_1_ul_total_bytes cell_1_cqi
cell_1_dl_bitrate cell_1_dl_err cell_1_dl_mcs cell_1_dl_tx cell_1_epre cell_1_initial_ta
cell_1_p_ue cell_1_pusch_snr cell_1_ri cell_1_turbo_decoder_avg cell_1_turbo_decoder_max
cell_1_turbo_decoder_min cell_1_ul_bitrate cell_1_ul_err cell_1_ul_mcs cell_1_ul_path_loss
cell_1_ul_phr cell_1_ul_tx dl_bitrate ran_ue_id ul_bitrate cell_3_cqi cell_3_dl_bitrate
cell_3_dl_err cell_3_dl_mcs cell_3_dl_tx cell_3_epre cell_3_initial_ta cell_3_pusch_snr cell_3_ri
cell_3_turbo_decoder_avg cell_3_turbo_decoder_max cell_3_turbo_decoder_min cell_3_ul_bitrate
cell_3_ul_err cell_3_ul_tx cell_3_p_ue cell_3_ul_mcs cell_3_ul_path_loss cell_3_ul_phr
ul_retx_max dl_retx_max""".split()

#: Columns that really are constant in the dataset (notebook 03 drops 13).
CONSTANT_RAW = [
    "bearer_0_qos_flow_id", "bearer_1_qos_flow_id", "bearer_0_sst", "bearer_1_sst",
    "cell_1_cell_id", "cell_3_cell_id", "cell_1_ul_n_layer", "cell_3_ul_n_layer",
    "cell_1_ul_rank", "cell_3_ul_rank", "t3512",
    "ue_aggregate_max_bitrate_dl", "ue_aggregate_max_bitrate_ul",
]

PASS, FAIL = [], []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        PASS.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}\n        {e}")
    except Exception as e:                                  # noqa: BLE001
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}\n        {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Synthetic raw NCSRD file with the true schema
# ---------------------------------------------------------------------------


def make_raw_csv(path: Path, n: int = 3000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    # 30s spacing over ~25 hours, so the fixture yields plenty of 20-minute
    # blocks. Timestamps straddle the real attack windows so both classes appear.
    t = pd.date_range("2024-08-18 06:30:00", periods=n, freq="30s", tz="UTC")
    data: dict[str, object] = {"_time": t.astype(str)}
    for c in RAW_COLUMNS:
        if c == "_time":
            continue
        if c.endswith("_apn"):
            data[c] = rng.choice(["internet", "ims"], n)
        elif c.endswith(("_ip", "_ipv6")):
            data[c] = [f"10.0.0.{i % 254}" for i in range(n)]
        elif c in CONSTANT_RAW:
            data[c] = np.full(n, 7.0)                     # genuinely constant
        elif c == "imeisv":
            data[c] = rng.integers(0, 7, n)               # 7 UEs, as in the capture
        else:
            data[c] = rng.normal(size=n) * rng.uniform(0.5, 5)
    df = pd.DataFrame(data)
    # Per-cell columns are sparse in reality: a UE attaches to one cell at a time.
    cell1 = [c for c in df.columns if c.startswith("cell_1_")]
    cell3 = [c for c in df.columns if c.startswith("cell_3_")]
    on1 = rng.random(n) < 0.5
    df.loc[~on1, cell1] = np.nan
    df.loc[on1, cell3] = np.nan
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# NCSRD checks
# ---------------------------------------------------------------------------


def ncsrd_checks(tmp: Path) -> None:
    raw = tmp / "amari_ue_data_classic_tabular.csv"
    make_raw_csv(raw)
    out = tmp / "processed"

    df_raw = ncsrd_prep.add_attack_labels(ncsrd_prep.load_raw(raw))
    n_rows = len(df_raw)
    b38 = ncsrd_prep.run(raw, out, feature_set="base38", df_raw=df_raw)
    s49 = ncsrd_prep.run(raw, out, feature_set="saurabh49", df_raw=df_raw)

    print("\n[NCSRD] feature sets")

    def _base38_exact():
        cols = [c for c in b38.columns if c != "attack_label"]
        assert cols == ncsrd_prep.BASE38_FEATURES, (
            f"base38 columns differ from Table II.\n"
            f"  missing: {set(ncsrd_prep.BASE38_FEATURES) - set(cols)}\n"
            f"  extra  : {set(cols) - set(ncsrd_prep.BASE38_FEATURES)}")
        assert len(cols) == 38, f"expected 38 features, got {len(cols)}"
        return "38 features, exact Table II names and order"
    check("base38 == the paper's Table II, exactly", _base38_exact)

    def _s49_exact():
        cols = [c for c in s49.columns if c != "attack_label"]
        assert set(cols) == set(SAURABH49), (
            f"saurabh49 differs from notebook 04's saved column list.\n"
            f"  missing: {sorted(set(SAURABH49) - set(cols))}\n"
            f"  extra  : {sorted(set(cols) - set(SAURABH49))}")
        return f"{len(cols)} features, matches notebook 04"
    check("saurabh49 == notebook 04's 49 columns", _s49_exact)

    def _disjointness():
        a, b = set(b38.columns) - {"attack_label"}, set(s49.columns) - {"attack_label"}
        assert not a.issubset(b) and not b.issubset(a), \
            "the two feature sets must be distinct lineages, not nested variants"
        return f"{len(a & b)} shared, {len(a - b)} base-only, {len(b - a)} saurabh-only"
    check("the two feature sets are genuinely different", _disjointness)

    def _constants_kept():
        meta = json.loads((out / "ncsrd_base38_meta.json").read_text())
        kept = meta["constant_features_kept"]
        assert kept, "base38 should retain the paper's constant features"
        assert "t3512" in kept, f"t3512 should be constant here; kept={kept}"
        return f"{len(kept)} constant features kept (paper keeps them)"
    check("base38 keeps constant features; saurabh49 drops them", _constants_kept)

    def _row_alignment():
        assert len(b38) == len(s49) == n_rows, \
            f"row counts differ: base38={len(b38)} saurabh49={len(s49)} raw={n_rows}"
        seq = pd.read_csv(out / "sequence_index.csv")
        assert len(seq) == n_rows, f"sequence_index has {len(seq)} rows, expected {n_rows}"
        assert np.array_equal(b38["attack_label"].to_numpy(),
                              s49["attack_label"].to_numpy()), \
            "labels diverge between feature sets - rows are not aligned"
        return f"{n_rows} rows, identical order in both sets"
    check("both feature sets share identical rows (one split fits both)", _row_alignment)

    def _no_nan():
        for name, d in (("base38", b38), ("saurabh49", s49)):
            n = int(d.isna().sum().sum())
            assert n == 0, f"{name} still has {n} NaNs after imputation"
        return "no NaNs in either set"
    check("imputation leaves no NaNs", _no_nan)

    def _no_identifiers_in_base38():
        banned = {"imeisv", "5g_tmsi", "amf_ue_id", "rnti", "ran_id", "ran_ue_id",
                  "_time", "bearer_0_ip", "bearer_1_ip"}
        leaked = banned & set(b38.columns)
        assert not leaked, f"identifier columns leaked into base38: {leaked}"
        return "no UE/session identifiers"
    check("base38 carries no identifier columns", _no_identifiers_in_base38)

    # --- adapter -----------------------------------------------------------
    print("\n[NCSRD] adapter")
    ad = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False)

    def _split_disjoint():
        ad.project_standard_split()
        sp = ad._split
        tr, va, te = sp["train"], sp["val"], sp["test"]
        assert not (set(tr) & set(te)), "train and test overlap"
        assert not (set(tr) & set(va)), "train and val overlap"
        assert not (set(va) & set(te)), "val and test overlap"
        assert len(tr) + len(va) + len(te) == len(ad.y), "split does not cover all rows"
        return f"train={len(tr)} val={len(va)} test={len(te)}, disjoint and complete"
    check("project-standard split is disjoint and complete", _split_disjoint)

    def _temporal_order():
        # strategy="temporal" is still offered (it is NOT the project standard
        # any more -- see D2 -- but the code path must stay correct).
        at = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False)
        at.split(strategy="temporal", test_size=0.2, val_size=0.1, random_state=42)
        sp, t = at._split, at._time
        assert t[sp["train"]].max() <= t[sp["val"]].min(), "train overlaps val in time"
        assert t[sp["val"]].max() <= t[sp["test"]].min(), "val overlaps test in time"
        return "train < val < test chronologically"
    check("the temporal strategy remains strictly chronological", _temporal_order)

    def _refit_scaler():
        tr = ad._split["train"]
        mu, sd = ad._X[tr].mean(0), ad._X[tr].std(0)
        assert np.allclose(mu, 0, atol=1e-4), f"train mean not 0 (max {abs(mu).max():.2e})"
        assert np.allclose(sd[sd > 0], 1, atol=1e-4), "train std not 1"
        return "train mean 0 / std 1 -> scaling fit on train only"
    check("refit_scaler removes scaling leakage", _refit_scaler)

    def _refit_equivalence():
        # Re-standardizing already-scaled data on train == scaling raw on train.
        raw_df = pd.read_csv(out / "ncsrd_saurabh49.csv").drop(columns=["attack_label"])
        Z = raw_df.to_numpy(np.float64)
        tr = ad._split["train"]
        m, s = Z[tr].mean(0), Z[tr].std(0)
        s[s == 0] = 1.0
        assert np.allclose((Z - m) / s, ad._X, atol=1e-4), "refit path is not affine-equivalent"
        return "affine equivalence holds"
    check("refit_scaler is equivalent to scaling raw data on train", _refit_equivalence)

    def _bundle_indices():
        b = ad.for_xgboost(balance="oversample")
        assert b.test_index is not None and len(b.test_index) == len(b.y_test)
        assert b.test_time is not None and len(b.test_time) == len(b.y_test)
        full_y = ad.y[b.test_index]
        assert np.array_equal(full_y, b.y_test), \
            "test_index does not recover y_test - window evaluation would be wrong"
        return "test_index/test_time round-trip to the source rows"
    check("DataBundle indices support window-based evaluation", _bundle_indices)

    def _test_untouched():
        b = ad.for_xgboost(balance="oversample")
        assert len(b.y_test) == len(ad._split["test"]), "test set was resampled"
        assert len(b.y_train) > len(ad._split["train"]), "oversampling did not apply to train"
        return "test never resampled; train was"
    check("balancing touches only the training split", _test_untouched)

    def _split_roundtrip():
        p = ad.save_split(tmp / "split.npz")
        ad2 = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False).load_split(p)
        assert np.array_equal(ad2._split["test"], ad._split["test"])
        assert ad2._split["refit_scaler"] == ad._split["refit_scaler"]
        assert np.allclose(ad2._X, ad._X), "reloaded split restores different scaling"
        return "same rows AND same scaling after reload"
    check("frozen split round-trips exactly", _split_roundtrip)

    def _reference_split():
        ad3 = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False)
        ad3.reference_split()
        assert ad3._split["strategy"] == "random"
        assert ad3._split["refit_scaler"] is False
        assert abs(len(ad3._split["test"]) / n_rows - 0.2) < 0.01
        return "random stratified 80/20, notebook-04 parity"
    check("reference split reproduces notebook 04's policy", _reference_split)

    def _base38_adapter():
        adb = NetworkDataAdapter.for_feature_set("base38", processed_dir=out, verbose=False)
        assert len(adb.feature_names) == 38
        adb.project_standard_split()
        b = adb.for_xgboost(balance="none")
        assert b.X_train.shape[1] == 38
        return f"base38 bundle X_train {b.X_train.shape}"
    check("adapter loads base38 by name", _base38_adapter)


# ---------------------------------------------------------------------------
# Data4Cyber checks (skipped when the raw data is absent)
# ---------------------------------------------------------------------------


    # --- project-standard block split (D2) ---------------------------------
    print("\n[NCSRD] project-standard block split")
    from common import config as cfg

    adb = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False)
    adb.split(**cfg.PROJECT_STANDARD_SPLIT)

    def _block_non_degenerate():
        sp = adb._split
        for name in ("train", "val", "test"):
            y = adb.y[sp[name]]
            assert len(y) > 0, f"{name} split is empty"
            assert len(np.unique(y)) == 2, (
                f"{name} split has only class {np.unique(y).tolist()} - a "
                f"threshold cannot be selected on a single-class validation set")
        rates = {k: round(float(adb.y[sp[k]].mean()), 4)
                 for k in ("train", "val", "test")}
        return f"all splits have both classes; attack rates {rates}"
    check("block split is non-degenerate (both classes everywhere)",
          _block_non_degenerate)

    def _blocks_whole():
        sp = adb._split
        seen = {}
        for name in ("train", "val", "test"):
            for blk in np.unique(adb._blocks[sp[name]]):
                assert blk not in seen, (
                    f"block {blk} appears in both {seen[blk]} and {name} - "
                    f"a contiguous block must stay entirely on one side")
                seen[blk] = name
        return f"{len(seen)} blocks, each entirely within one split"
    check("no contiguous block spans two splits", _blocks_whole)

    def _no_row_shuffling():
        # Every row of a block must land in the same split as its block.
        sp = adb._split
        side = np.empty(len(adb.y), dtype=object)
        for name in ("train", "val", "test"):
            side[sp[name]] = name
        for blk in np.unique(adb._blocks):
            sides = set(side[adb._blocks == blk])
            assert len(sides) == 1, (
                f"block {blk} was split across {sides} - rows must move as "
                f"whole blocks, never individually")
        return "rows move only as whole blocks"
    check("no row-level shuffling across splits", _no_row_shuffling)

    def _disjoint_complete():
        sp = adb._split
        tr, va, te = (set(sp[k].tolist()) for k in ("train", "val", "test"))
        assert not (tr & va) and not (tr & te) and not (va & te), "splits overlap"
        assert len(tr | va | te) == len(adb.y), "split does not cover every row"
        return f"train={len(tr)} val={len(va)} test={len(te)}"
    check("block split is disjoint and covers every row", _disjoint_complete)

    def _deterministic():
        a = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False)
        a.split(**cfg.PROJECT_STANDARD_SPLIT)
        b_ = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False)
        b_.split(**cfg.PROJECT_STANDARD_SPLIT)
        for k in ("train", "val", "test"):
            assert np.array_equal(a._split[k], b_._split[k]), (
                f"{k} differs between two identical runs - split is not "
                f"deterministic")
        c = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False)
        c.split(**{**cfg.PROJECT_STANDARD_SPLIT, "random_state": 7})
        assert not np.array_equal(a._split["test"], c._split["test"]), (
            "a different seed produced an identical split")
        return "same config -> identical indices; different seed -> different"
    check("block split is deterministic from the seed", _deterministic)

    def _both_feature_sets_same_split():
        a = NetworkDataAdapter(out / "ncsrd_base38.csv", verbose=False)
        a.split(**cfg.PROJECT_STANDARD_SPLIT)
        for k in ("train", "val", "test"):
            assert np.array_equal(a._split[k], adb._split[k]), (
                f"base38 and saurabh49 disagree on the {k} rows - one frozen "
                f"index file must cover both feature sets")
        return "base38 and saurabh49 get identical row indices"
    check("one split covers both feature sets", _both_feature_sets_same_split)

    def _block_roundtrip():
        pth = adb.save_split(tmp / "block_split.npz")
        re = NetworkDataAdapter(out / "ncsrd_saurabh49.csv", verbose=False).load_split(pth)
        assert re._split["strategy"] == "block"
        assert re._split["block_minutes"] == cfg.BLOCK_MINUTES
        assert np.array_equal(re._split["test"], adb._split["test"])
        assert re._blocks is not None, "block ids were not restored on load"
        assert np.array_equal(re._blocks, adb._blocks)
        return "strategy, block_minutes, indices and block ids all restored"
    check("frozen block split round-trips exactly", _block_roundtrip)

    def _bundle_carries_blocks():
        bb = adb.for_xgboost(balance="none")
        assert bb.test_block is not None, (
            "DataBundle.test_block is required so methods can window inside a "
            "block instead of across boundaries")
        assert len(bb.test_block) == len(bb.y_test)
        assert np.array_equal(bb.test_block, adb._blocks[bb.test_index])
        # ordered by (block, time) -> each block is one contiguous run
        runs = int((np.diff(bb.test_block) != 0).sum()) + 1
        assert runs == len(np.unique(bb.test_block)), (
            "test rows are not grouped by block; windowing would interleave "
            "blocks")
        return f"{len(np.unique(bb.test_block))} test blocks, contiguous runs"
    check("DataBundle exposes block ids for window construction",
          _bundle_carries_blocks)

    def _blocks_are_contiguous_in_time():
        # Within a block, timestamps must form one contiguous stretch.
        span = cfg.BLOCK_MINUTES * 60 * 1_000_000_000
        for blk in np.unique(adb._blocks)[:20]:
            ts = adb._time[adb._blocks == blk]
            assert ts.max() - ts.min() < span, (
                f"block {blk} spans more than {cfg.BLOCK_MINUTES} minutes")
        return f"each block covers at most {cfg.BLOCK_MINUTES} minutes"
    check("blocks are contiguous in time", _blocks_are_contiguous_in_time)

    def _timestamp_units():
        # int64 must be NANOseconds; a us/ns mix-up silently maps 2024 -> 1970.
        yr = pd.to_datetime(pd.Series(adb._time), unit="ns", utc=True).dt.year
        assert set(yr.unique()) == {2024}, (
            f"timestamps decode to years {sorted(set(yr.unique()))}, expected "
            f"2024 - _time must be nanoseconds since epoch")
        return "test_time / _time are nanoseconds since epoch"
    check("timestamps are nanoseconds, not microseconds", _timestamp_units)

def data4cyber_checks() -> None:
    proc = ROOT / "data" / "data4cyber" / "processed"
    if not (proc / "block" / "manifest.json").exists():
        print("\n[Data4Cyber] SKIPPED - run src/common/data/data4cyber_prep.py first")
        return
    print("\n[Data4Cyber] block split (primary)")
    man = json.loads((proc / "block" / "manifest.json").read_text())

    def _no_clock():
        bad = [f for f in man["features"] if f.endswith((".realtime", ".timestamp"))]
        assert not bad, f"clock features survived into the feature set: {bad}"
        return f"{len(man['features'])} features, no wall-clocks"
    check("no absolute-clock features (scenario leakage)", _no_clock)

    def _families_everywhere():
        tr = set(man["train"]["scenarios"])
        for s in ("validation", "test"):
            assert set(man[s]["scenarios"]) == tr, \
                f"{s} scenarios {man[s]['scenarios']} != train {sorted(tr)}"
        return f"all {len(tr)} scenarios present in train/val/test"
    check("every attack family appears in every split", _families_everywhere)

    def _balanced():
        rates = {s: man[s]["positive_row_rate"] for s in ("train", "validation", "test")}
        spread = max(rates.values()) - min(rates.values())
        assert spread < 0.20, f"attack-rate spread {spread:.2f} across splits: {rates}"
        return f"attack rates {rates}, spread {spread:.3f}"
    check("class balance is comparable across splits", _balanced)

    def _blocks_disjoint():
        seen: dict[str, str] = {}
        for s in ("train", "validation", "test"):
            z = np.load(proc / "block" / f"{s}_rows.npz", allow_pickle=False)
            for b in np.unique(z["block"]):
                assert b not in seen, \
                    f"block {b} is in both {seen[b]} and {s} - boundary leak"
                seen[b] = s
        return f"{len(seen)} blocks, each entirely on one side"
    check("no contiguous block spans two splits", _blocks_disjoint)

    def _row_shapes():
        nf = len(man["features"])
        for s in ("train", "validation", "test"):
            z = np.load(proc / "block" / f"{s}_rows.npz", allow_pickle=False)
            assert z["X"].shape == (man[s]["rows"], nf), \
                f"{s} rows X is {z['X'].shape}, expected {(man[s]['rows'], nf)}"
            assert len(z["y"]) == len(z["scenario"]) == len(z["timestamp"]) == len(z["X"])
            w = np.load(proc / "block" / f"{s}_windows.npz", allow_pickle=False)
            assert w["X"].shape[1:] == (man["window_size_seconds"], nf), \
                f"{s} window shape {w['X'].shape}"
        return "row and window tensors have consistent shapes and metadata"
    check("row-level and window-level outputs are consistent", _row_shapes)

    def _secondary_exists():
        m2 = json.loads((proc / "scenario" / "manifest.json").read_text())
        assert "S6_mqtt_supply_chain_compromise" in m2["test"]["scenarios"]
        assert "S6_mqtt_supply_chain_compromise" not in m2["train"]["scenarios"]
        return "S6 held out of training, as the robustness experiment intends"
    check("secondary scenario split still holds out an unseen family", _secondary_exists)


def main() -> int:
    print("=" * 70)
    print("Shared data-pipeline checks")
    print("=" * 70)
    with tempfile.TemporaryDirectory() as td:
        ncsrd_checks(Path(td))
    data4cyber_checks()
    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for name, err in FAIL:
        print(f"  FAILED: {name} -- {err}")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
