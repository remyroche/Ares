#!/usr/bin/env python3
"""Matched S/R and 15-minute-E2 input ablation for the retained dual MC1 maps.

This is deliberately an offline, *map-input* test.  It retains the separate
current-v5 and BCF score families, their absolute-policy-EV MC1_d2 target,
the 21-day prior-resolved score-band shift, dual +50-bps admission, BCF-EV
auction priority, source-aligned rich-policy outcomes and the controlled
global portfolio replay.

The frozen August-17 MC1 receipts are the production reference.  Their exact
BCF pre-February score warm-up ledger was pruned during the later research
cleanup, so a byte-identical refit of that historical map is not possible.
Accordingly this runner reports both:

* the untouched frozen receipt control; and
* a matched post-February refit control used as the only delta baseline for
  S/R/E2 input variants.

No result here changes a live, canonical, policy or model artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS
from extreme_price_movements.causal_profile_geometry import PROFILE_FEATURES
from extreme_price_movements.portfolio_policy_replay import replay_candidates
from scripts import run_causal_sr_mc1_residual_ablation as sr
from scripts.ablate_strict_r3_bcf_current_v5_agreement_blend import (
    POLICY_COLUMNS,
    _to_candidates,
    _valid,
)
from scripts.report_strict_r3_mc1_d2_controlled_portfolio import (
    CAUSAL_AUCTION_CURVE,
    _metrics,
    _params,
)
from scripts.run_strict_r3_mc1_d2_controlled_ablation import (
    CORE,
    SEED,
    _causal_shifts,
    _day_balanced,
    _fit_hgb,
    _score_bands,
)
from scripts.run_strict_r3_p8u_v58_e2_50_exact_matched import (
    _materialize_target_free_features,
)


CURRENT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
)
BCF = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
)
SR_ROOT = ROOT / "data_perp/artifacts/causal_sr_heads_oof_20260830_v3_entrypivotfix"
DEFAULT_OUT = ROOT / "data_perp/artifacts/canonical_sr_e2_mc1_input_ablation_20260831_v1"

# The E2 feature can first be trained causally from the retained target-free
# family score population in February 2026.  April/May predictions warm the
# June MC1 fit; June and July are the predeclared comparable scored months.
FEATURE_START = pd.Timestamp("2026-02-01T00:00:00Z")
EVAL_START = pd.Timestamp("2026-06-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-08-01T00:00:00Z")
E2_TRAIN_MONTHS = 4
E2_MIN_ROWS = 200
E2_OUTPUT = "e2_15m_prequential_raw_policy_bps"
E2_AVAILABLE = "e2_15m_prequential_available"
SR_AVAILABLE = "sr_snapshot_available"
OI_POSITIONING_FEATURES = (
    "oi_long_build_support_probability",
    "oi_short_build_resistance_probability",
    "oi_failure_probability_long_build",
    "oi_failure_probability_short_build",
    "oi_trap_probability_long_build",
    "oi_trap_probability_short_build",
    "oi_unwind_probability_long_build",
    "oi_unwind_probability_short_build",
)
OI_POSITIONING_AVAILABLE = "oi_positioning_snapshot_available"
OI_POSITIONING_FILE = "entry_oi_positioning_oof_features.parquet"
PROFILE_GEOMETRY_HEADS = (
    "profile_conditional_utility",
    "profile_favorable_magnitude_q50",
    "profile_adverse_break_probability",
)
PROFILE_GEOMETRY_AVAILABLE = "profile_snapshot_available"
PROFILE_GEOMETRY_FILE = "entry_profile_geometry_oof_features.parquet"
ANCHOR_ENTRY_HEADS = (
    "anchor_bullish_expected_utility_atr",
    "anchor_bearish_expected_utility_atr",
    "anchor_bullish_rejection_probability",
    "anchor_bearish_rejection_probability",
    "anchor_bullish_continuation_probability",
    "anchor_bearish_continuation_probability",
    "anchor_bullish_accepted_cross_probability",
    "anchor_bearish_accepted_cross_probability",
    "anchor_long_structure_utility",
    "anchor_long_structure_continuation_balance",
)
ANCHOR_AVAILABLE = "anchor_snapshot_available"
ADMISSION_BPS = 50.0

# C1 is deliberately decomposed into two semantic blocks.  The risk block can
# only describe an adverse break/rejection context, while the support block
# describes the supporting structure and its maturity.  Both retain the same
# explicit availability indicator: an absent causal snapshot is never a
# candidate filter.
SR_DEMOTION_FEATURES = (
    "sr_long_resistance_break_probability",
    "sr_long_downside_break_probability",
    "sr_long_resistance_rejection_strength",
    "sr_long_resistance_distance_atr",
    "sr_resistance_prior_strength",
    "sr_resistance_reaction_magnitude_q50",
)
SR_SUPPORT_FEATURES = (
    "sr_long_support_hold_strength",
    "sr_long_structure_balance",
    "sr_long_support_distance_atr",
    "sr_support_prior_strength",
    "sr_support_reaction_magnitude_q50",
)

POLICY_FORBIDDEN = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    "policy_outcome_source",
})


@dataclass(frozen=True)
class FamilySource:
    name: str
    scores: pd.DataFrame
    labels: pd.DataFrame
    frozen_map: pd.Series


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    # Every caller treats ``end`` as exclusive.  Subtracting one nanosecond
    # before normalising avoids accidentally scoring the first day/month of a
    # held-out successor period.
    last = (end - pd.Timedelta(nanoseconds=1)).normalize()
    return pd.date_range(start.normalize(), last, freq="MS", tz="UTC")


def _required_columns() -> tuple[str, ...]:
    return (
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", *CORE,
        "score_band", "mc1_expected_bps", *POLICY_COLUMNS,
    )


def _load_family(path: Path, name: str) -> FamilySource:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(_required_columns()).difference(available))
    if missing:
        raise ValueError(f"{name}: source prediction lacks fields: {missing}")
    frame = pd.read_parquet(path, columns=list(_required_columns())).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["policy_label_available_ts"] = pd.to_datetime(
        frame["policy_label_available_ts"], utc=True, errors="raise"
    )
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{name}: duplicate retained candidate identities")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{name}: retained source is not long-only")
    finite = frame.loc[:, list(CORE)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    if not finite.all():
        raise AssertionError(f"{name}: retained source includes incomplete MC1-core rows")
    rebuilt_band = _score_bands(frame)
    stored_band = pd.to_numeric(frame["score_band"], errors="raise").to_numpy(np.int8)
    if not np.array_equal(rebuilt_band, stored_band):
        raise AssertionError(f"{name}: retained score-band contract no longer reproduces")
    scores = frame.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *CORE, "score_band"]].copy()
    labels = frame.loc[:, ["candidate_id", *POLICY_COLUMNS]].copy()
    frozen = pd.to_numeric(frame["mc1_expected_bps"], errors="raise").copy()
    return FamilySource(name=name, scores=scores, labels=labels, frozen_map=frozen)


def _assert_policy_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    overlap = left.merge(right, on="candidate_id", suffixes=("_left", "_right"), validate="one_to_one")
    if overlap.empty:
        raise AssertionError("family MC1 sources have no common candidate identities")
    for field in POLICY_COLUMNS:
        lhs, rhs = overlap[f"{field}_left"], overlap[f"{field}_right"]
        if pd.api.types.is_numeric_dtype(lhs):
            equal = np.isclose(
                pd.to_numeric(lhs, errors="coerce").to_numpy(float),
                pd.to_numeric(rhs, errors="coerce").to_numpy(float),
                equal_nan=True, rtol=0.0, atol=0.0,
            ).all()
        else:
            equal = lhs.fillna("__null__").astype(str).equals(rhs.fillna("__null__").astype(str))
        if not equal:
            raise AssertionError(f"retained policy labels differ across score families: {field}")


def _score_union(current: FamilySource, bcf: FamilySource) -> pd.DataFrame:
    """Return one target-free source row per retained family candidate ID."""
    bcf_scores = bcf.scores.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score"]].copy()
    bcf_scores = bcf_scores.rename(columns={"final_score": "bcf_final_score"})
    current_scores = current.scores.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score"]].copy()
    current_scores = current_scores.rename(columns={"final_score": "current_final_score"})
    combined = bcf_scores.merge(
        current_scores, on=["candidate_id", "__decision_ts__", "__symbol__", "side_name"],
        how="outer", validate="one_to_one",
    )
    combined["score_for_feature_rank"] = combined["bcf_final_score"].combine_first(combined["current_final_score"])
    if combined["score_for_feature_rank"].isna().any():
        raise AssertionError("target-free score union has no family score")
    combined = combined.sort_values(
        ["__decision_ts__", "score_for_feature_rank", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    )
    combined["base_timestamp_rank"] = combined.groupby("__decision_ts__", sort=False).cumcount() + 1
    combined["symbol"] = combined["__symbol__"].astype(str)
    combined["timestamp"] = combined["__decision_ts__"]
    # The reusable feature materialiser carries these fields only as target-free
    # provenance; no current or BCF MC1 output influences the feature values.
    combined["bcf_mc1_expected_bps"] = 0.0
    combined["current_mc1_expected_bps"] = 0.0
    combined["dual_mc1_min_bps"] = 0.0
    if combined["candidate_id"].duplicated().any():
        raise AssertionError("target-free family union duplicated identity")
    return combined


def _load_or_materialize_features(route: pd.DataFrame, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise causal 15m values from target-free candidate identities only."""
    if cache.exists():
        feature = pd.read_parquet(cache)
        if feature["candidate_id"].astype(str).duplicated().any():
            raise AssertionError("cached target-free E2 features duplicate candidate identity")
        route_ids = pd.Index(route["candidate_id"].astype(str))
        if not route_ids.equals(pd.Index(feature["candidate_id"].astype(str))):
            raise AssertionError("cached E2 feature identity does not match the declared target-free route")
        return feature, pd.DataFrame([{"status": "reused", "rows": int(len(feature))}])
    feature = _materialize_target_free_features(route)
    feature["candidate_id"] = feature["candidate_id"].astype(str)
    ordered = route.loc[:, ["candidate_id"]].merge(feature, on="candidate_id", how="left", validate="one_to_one")
    if len(ordered) != len(route) or ordered["candidate_id"].duplicated().any():
        raise AssertionError("causal E2 feature materialisation changed target-free identity")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_parquet(cache, index=False, compression="zstd")
    coverage = ordered.groupby("feature_source_status", dropna=False, sort=True).size().rename("rows").reset_index()
    return ordered, coverage


def _candidate_labels(current: FamilySource, bcf: FamilySource) -> pd.DataFrame:
    """Use the immutable identical policy row, never its field in a score view."""
    _assert_policy_equal(current.labels, bcf.labels)
    merged = bcf.labels.merge(current.labels, on="candidate_id", how="outer", suffixes=("_bcf", "_current"))
    result = pd.DataFrame({"candidate_id": merged["candidate_id"].astype(str)})
    for field in POLICY_COLUMNS:
        result[field] = merged[f"{field}_bcf"].combine_first(merged[f"{field}_current"])
    if result["candidate_id"].duplicated().any():
        raise AssertionError("combined policy label source duplicates candidate identity")
    result["policy_label_available_ts"] = pd.to_datetime(
        result["policy_label_available_ts"], utc=True, errors="raise"
    )
    return result


def _fit_e2(train: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=350, learning_rate=.03,
        max_depth=4, num_leaves=15,
        min_child_samples=max(8, int(math.ceil(len(train) * .02))),
        subsample=.80, colsample_bytree=.80, reg_lambda=4.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    model.fit(train.loc[:, list(FIFTEEN_MINUTE_FEATURE_KEYS)], train["policy_net_bps"].to_numpy(float))
    return model


def _prequential_e2(
    route: pd.DataFrame, labels: pd.DataFrame, features: pd.DataFrame,
    *, start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce direct-E2 scores month-by-month from prior-resolved labels only."""
    target_free = route.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    feature_columns = ["candidate_id", "feature_source_status", "finite_feature_count", *FIFTEEN_MINUTE_FEATURE_KEYS]
    missing = sorted(set(feature_columns).difference(features.columns))
    if missing:
        raise AssertionError(f"E2 feature cache lacks fields: {missing}")
    prepared = target_free.merge(features.loc[:, feature_columns], on="candidate_id", how="left", validate="one_to_one")
    if len(prepared) != len(target_free):
        raise AssertionError("E2 feature merge changed target-free identity")
    prepared = prepared.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    status = prepared["feature_source_status"].fillna("missing_feature_row").astype(str).eq("ok")
    prepared["e2_feature_available"] = status & pd.to_numeric(
        prepared["finite_feature_count"], errors="coerce"
    ).ge(50)
    pieces: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for month in _month_range(start, end):
        held_end = month + pd.offsets.MonthBegin(1)
        held = prepared.loc[
            prepared["__decision_ts__"].ge(month) & prepared["__decision_ts__"].lt(held_end)
        ].copy()
        train_start = month - pd.DateOffset(months=E2_TRAIN_MONTHS)
        fit = prepared.loc[
            prepared["__decision_ts__"].ge(train_start)
            & prepared["__decision_ts__"].lt(month)
            & prepared["policy_path_valid"].fillna(False).astype(bool)
            & prepared["policy_label_available_ts"].lt(month)
            & prepared["e2_feature_available"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(prepared["policy_net_bps"], errors="coerce"))
        ].copy()
        observed_months = set(fit["__decision_ts__"].dt.strftime("%Y-%m"))
        enough = len(fit) >= E2_MIN_ROWS and len(observed_months) >= 2
        held[E2_OUTPUT] = np.nan
        if enough:
            model = _fit_e2(fit)
            eligible = held["e2_feature_available"].fillna(False).astype(bool)
            if eligible.any():
                held.loc[eligible, E2_OUTPUT] = model.predict(held.loc[eligible, list(FIFTEEN_MINUTE_FEATURE_KEYS)])
        held[E2_AVAILABLE] = held[E2_OUTPUT].notna().astype(np.int8)
        if held.loc[held[E2_AVAILABLE].eq(1), E2_OUTPUT].isna().any():
            raise AssertionError("finite E2 availability without a held direct prediction")
        pieces.append(held.loc[:, ["candidate_id", E2_OUTPUT, E2_AVAILABLE]])
        audits.append({
            "month": f"{month:%Y-%m}", "train_start": train_start.isoformat(),
            "train_rows": int(len(fit)), "train_feature_months": sorted(observed_months),
            "held_rows": int(len(held)), "held_e2_available": int(held[E2_AVAILABLE].sum()),
            "status": "scored" if enough else "insufficient_prior_feature_support",
        })
    result = pd.concat(pieces, ignore_index=True)
    if result["candidate_id"].duplicated().any() or len(result) != len(target_free):
        raise AssertionError("prequential E2 output does not cover target-free route exactly once")
    return result, pd.DataFrame(audits)


def _augment_family(
    source: FamilySource, sr_root: Path, e2: pd.DataFrame, positioning_root: Path | None = None,
    profile_root: Path | None = None, anchor_root: Path | None = None, anchor_variant: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = source.scores.copy()
    scores, _ = sr._merge_causal_sr(scores, sr_root)
    scores = _merge_causal_oi_positioning(scores, positioning_root)
    scores = _merge_causal_profile_geometry(scores, profile_root)
    scores = _merge_causal_anchor(scores, anchor_root, anchor_variant)
    scores = scores.merge(e2, on="candidate_id", how="left", validate="one_to_one")
    scores[E2_AVAILABLE] = scores[E2_AVAILABLE].fillna(0).astype(np.int8)
    labels = source.labels.copy()
    labels["policy_label_available_ts"] = pd.to_datetime(labels["policy_label_available_ts"], utc=True, errors="raise")
    full = scores.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    if len(full) != len(source.scores):
        raise AssertionError(f"{source.name}: input augmentation changed family identity")
    return scores, full


def _assert_causal_oi_positioning_root(root: Path) -> Path:
    resolved = root.resolve()
    manifest_path = resolved / "run_manifest.json"
    snapshot_path = resolved / OI_POSITIONING_FILE
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("causal OI-positioning root lacks its manifest or entry snapshot output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    causality = " ".join((str(manifest.get("schema", "")), str(manifest.get("causality", "")))).upper()
    if "CAUSAL-OI-POSITIONING-HEADS-OOF" not in causality:
        raise AssertionError("OI-positioning source does not declare an OOF causal contract")
    return snapshot_path


def _merge_causal_oi_positioning(panel: pd.DataFrame, root: Path | None) -> pd.DataFrame:
    """Candidate-time causal OI merge; absent optional source remains model missingness."""
    work = panel.copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    if root is None:
        for field in OI_POSITIONING_FEATURES:
            work[field] = np.nan
        work[OI_POSITIONING_AVAILABLE] = np.int8(0)
        return work
    snapshot_path = _assert_causal_oi_positioning_root(root)
    columns = ["candidate_id", "snapshot_ts", *OI_POSITIONING_FEATURES]
    available = set(pq.ParquetFile(snapshot_path).schema_arrow.names)
    missing = sorted(set(columns).difference(available))
    if missing:
        raise AssertionError(f"OI-positioning entry output lacks fields: {missing}")
    snapshots = pd.read_parquet(snapshot_path, columns=columns)
    snapshots["candidate_id"] = snapshots["candidate_id"].astype(str)
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots["snapshot_ts"], utc=True, errors="raise")
    if snapshots.duplicated(["candidate_id", "snapshot_ts"]).any():
        raise AssertionError("causal OI-positioning output duplicates candidate-time identity")
    work["snapshot_ts"] = work["__decision_ts__"]
    merged = work.merge(snapshots, on=["candidate_id", "snapshot_ts"], how="left", validate="one_to_one")
    if len(merged) != len(work) or not np.array_equal(merged["candidate_id"].to_numpy(str), work["candidate_id"].to_numpy(str)):
        raise AssertionError("OI-positioning merge changed target-free candidate identity or order")
    merged[OI_POSITIONING_AVAILABLE] = merged.loc[:, list(OI_POSITIONING_FEATURES)].notna().any(axis=1).astype(np.int8)
    return merged


def _assert_causal_profile_geometry_root(root: Path) -> Path:
    resolved = root.resolve()
    manifest_path = resolved / "run_manifest.json"
    snapshot_path = resolved / PROFILE_GEOMETRY_FILE
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise FileNotFoundError("causal profile-geometry root lacks its manifest or entry snapshot output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = " ".join((str(manifest.get("schema", "")), str(manifest.get("causality", "")), str(manifest.get("training", "")))).upper()
    if "CAUSAL-PROFILE-GEOMETRY-HEADS-2025TRAIN-2026CONFIRM" not in contract:
        raise AssertionError("profile-geometry source does not declare the required 2025-train causal contract")
    return snapshot_path


def _merge_causal_profile_geometry(panel: pd.DataFrame, root: Path | None) -> pd.DataFrame:
    """Merge optional, target-free causal profile-head outputs by candidate time."""
    work = panel.copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    fields = (*PROFILE_GEOMETRY_HEADS, PROFILE_GEOMETRY_AVAILABLE)
    if root is None:
        for field in PROFILE_GEOMETRY_HEADS:
            work[field] = np.nan
        work[PROFILE_GEOMETRY_AVAILABLE] = np.int8(0)
        return work
    snapshot_path = _assert_causal_profile_geometry_root(root)
    columns = ["candidate_id", "snapshot_ts", *PROFILE_GEOMETRY_HEADS, PROFILE_GEOMETRY_AVAILABLE]
    available = set(pq.ParquetFile(snapshot_path).schema_arrow.names)
    missing = sorted(set(columns).difference(available))
    if missing:
        raise AssertionError(f"profile-geometry entry output lacks fields: {missing}")
    snapshots = pd.read_parquet(snapshot_path, columns=columns)
    snapshots["candidate_id"] = snapshots["candidate_id"].astype(str)
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots["snapshot_ts"], utc=True, errors="raise")
    if snapshots.duplicated(["candidate_id", "snapshot_ts"]).any():
        raise AssertionError("profile-geometry output duplicates candidate-time identity")
    work["snapshot_ts"] = work["__decision_ts__"]
    merged = work.merge(snapshots, on=["candidate_id", "snapshot_ts"], how="left", validate="one_to_one")
    if len(merged) != len(work) or not np.array_equal(merged["candidate_id"].to_numpy(str), work["candidate_id"].to_numpy(str)):
        raise AssertionError("profile-geometry merge changed target-free candidate identity or order")
    availability = pd.to_numeric(merged[PROFILE_GEOMETRY_AVAILABLE], errors="coerce").fillna(0).astype("int8")
    finite = merged.loc[:, list(PROFILE_GEOMETRY_HEADS)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    merged[PROFILE_GEOMETRY_AVAILABLE] = (availability.eq(1) & finite).astype("int8")
    return merged


def _assert_causal_anchor_root(root: Path, variant: str | None) -> tuple[Path, str]:
    """Resolve the *2025-selected* anchor source variant and its entry file."""
    resolved = root.resolve()
    manifest_path = resolved / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("causal anchor root lacks run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = " ".join((str(manifest.get("schema", "")), str(manifest.get("causality", "")), str(manifest.get("selection", "")))).upper()
    if "CAUSAL-ANCHOR-HEADS-2025SELECT-2026CONFIRM" not in contract:
        raise AssertionError("anchor source does not declare the required 2025-select causal contract")
    selected = str(variant or manifest.get("downstream_contract", {}).get("selected_variant", ""))
    if not selected:
        raise AssertionError("anchor root lacks a selected 2025 source variant")
    if variant is not None and selected != str(manifest.get("downstream_contract", {}).get("selected_variant", "")):
        raise AssertionError("MC1 may consume only the anchor variant selected in 2025; variants cannot be selected on 2026")
    path = resolved / f"entry_anchor_{selected}_oof_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"anchor entry snapshot unavailable for selected variant {selected}")
    return path, selected


def _merge_causal_anchor(panel: pd.DataFrame, root: Path | None, variant: str | None) -> pd.DataFrame:
    """Attach optional, causal, 2025-selected anchor-head outputs by identity."""
    work = panel.copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    if root is None:
        for field in ANCHOR_ENTRY_HEADS:
            work[field] = np.nan
        work[ANCHOR_AVAILABLE] = np.int8(0)
        return work
    snapshot_path, _ = _assert_causal_anchor_root(root, variant)
    columns = ["candidate_id", "snapshot_ts", *ANCHOR_ENTRY_HEADS, ANCHOR_AVAILABLE]
    available = set(pq.ParquetFile(snapshot_path).schema_arrow.names)
    missing = sorted(set(columns).difference(available))
    if missing:
        raise AssertionError(f"anchor entry output lacks fields: {missing}")
    snapshots = pd.read_parquet(snapshot_path, columns=columns)
    snapshots["candidate_id"] = snapshots["candidate_id"].astype(str)
    snapshots["snapshot_ts"] = pd.to_datetime(snapshots["snapshot_ts"], utc=True, errors="raise")
    if snapshots.duplicated(["candidate_id", "snapshot_ts"]).any():
        raise AssertionError("anchor output duplicates candidate-time identity")
    work["snapshot_ts"] = work["__decision_ts__"]
    merged = work.merge(snapshots, on=["candidate_id", "snapshot_ts"], how="left", validate="one_to_one")
    if len(merged) != len(work) or not np.array_equal(merged["candidate_id"].to_numpy(str), work["candidate_id"].to_numpy(str)):
        raise AssertionError("anchor merge changed target-free candidate identity or order")
    availability = pd.to_numeric(merged[ANCHOR_AVAILABLE], errors="coerce").fillna(0).astype("int8")
    finite = merged.loc[:, list(ANCHOR_ENTRY_HEADS)].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1).ge(4)
    merged[ANCHOR_AVAILABLE] = (availability.eq(1) & finite).astype("int8")
    return merged


def _target_free_prediction_view(frame: pd.DataFrame, mapped: pd.Series, family: str) -> pd.DataFrame:
    fields = ["candidate_id", "__decision_ts__", "__symbol__", "side_name", "final_score", *sr.SR_FEATURES, SR_AVAILABLE, *OI_POSITIONING_FEATURES, OI_POSITIONING_AVAILABLE, *PROFILE_GEOMETRY_HEADS, PROFILE_GEOMETRY_AVAILABLE, *ANCHOR_ENTRY_HEADS, ANCHOR_AVAILABLE, E2_OUTPUT, E2_AVAILABLE]
    available = [field for field in fields if field in frame.columns]
    result = frame.loc[:, available].copy()
    result["mc1_family"] = family
    result["mc1_expected_bps"] = pd.to_numeric(mapped, errors="raise").to_numpy(float)
    leaked = sorted(POLICY_FORBIDDEN.intersection(result.columns))
    if leaked:
        raise AssertionError(f"target-free prediction view leaked policy fields: {leaked}")
    return result


def _refit_family(
    full: pd.DataFrame, *, family: str, extras: Sequence[str], start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the absolute MC1 map on the retained score surface only."""
    full = full.copy()
    full["day"] = full["__decision_ts__"].dt.normalize()
    fields = tuple((*CORE, *extras))
    missing = sorted(set(fields).difference(full.columns))
    if missing:
        raise AssertionError(f"{family}: MC1 input augmentation lacks fields: {missing}")
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for month in _month_range(start, end):
        held_end = month + pd.offsets.MonthBegin(1)
        fit = full.loc[
            full["__decision_ts__"].ge(FEATURE_START)
            & full["__decision_ts__"].lt(month)
            & full["policy_path_valid"].fillna(False).astype(bool)
            & full["policy_label_available_ts"].lt(month)
            & np.isfinite(pd.to_numeric(full["policy_net_bps"], errors="coerce"))
        ].copy()
        held = full.loc[
            full["__decision_ts__"].ge(month) & full["__decision_ts__"].lt(held_end)
        ].copy()
        if len(fit) < 5_000 or held.empty:
            raise RuntimeError(f"{family}/{month:%Y-%m}: insufficient prequential MC1 support")
        substrate = _day_balanced(fit)
        model, medians, curve, clip = _fit_hgb(substrate, fields)
        matrix = held.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").fillna(medians)
        held["static_expected_bps"] = model.predict(matrix)
        day = held["__decision_ts__"].dt.floor("1d")
        shifts = _causal_shifts(full, curve, pd.DatetimeIndex(day.unique()), "1d")
        held["recent_shift_bps"] = day.map(shifts).fillna(0.0).to_numpy(float)
        held["mc1_expected_bps"] = held["static_expected_bps"] + held["recent_shift_bps"]
        held["mc1_family"] = family
        held["fold_start"] = month
        held["mc1_target_clip_low_bps"], held["mc1_target_clip_high_bps"] = clip
        if not np.isfinite(held["mc1_expected_bps"].to_numpy(float)).all():
            raise AssertionError(f"{family}/{month:%Y-%m}: non-finite MC1 output")
        rows.append(held)
        audits.append({
            "family": family, "held_month": f"{month:%Y-%m}", "train_rows": int(len(fit)),
            "day_balanced_train_rows": int(len(substrate)), "held_rows": int(len(held)),
            "input_features": list(fields), "clip_low_bps": float(clip[0]), "clip_high_bps": float(clip[1]),
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def _combine_predictions(current: pd.DataFrame, bcf: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_tf = _target_free_prediction_view(current, current["mc1_expected_bps"], "current_v5")
    bcf_tf = _target_free_prediction_view(bcf, bcf["mc1_expected_bps"], "bcf")
    left = bcf_tf.rename(columns={
        "final_score": "bcf_final_score", "mc1_expected_bps": "bcf_mc1_expected_bps",
    })
    right = current_tf.loc[:, ["candidate_id", "__decision_ts__", "mc1_expected_bps"]].rename(
        columns={"mc1_expected_bps": "current_mc1_expected_bps"}
    )
    target_free = left.merge(right, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
    target_free["dual_admitted"] = (
        target_free["bcf_mc1_expected_bps"].ge(ADMISSION_BPS)
        & target_free["current_mc1_expected_bps"].ge(ADMISSION_BPS)
    )
    target_free["auction_priority_bps"] = target_free["bcf_mc1_expected_bps"]
    leaked = sorted(POLICY_FORBIDDEN.intersection(target_free.columns))
    if leaked:
        raise AssertionError(f"combined target-free admission panel leaked outcomes: {leaked}")
    outcome = target_free.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    if len(outcome) != len(target_free):
        raise AssertionError("outcome attachment changed target-free MC1 identity")
    return target_free, outcome


def _replay(target_free: pd.DataFrame, outcome: pd.DataFrame, arm: str, out: Path) -> tuple[dict[str, object], pd.DataFrame]:
    admit = target_free["dual_admitted"].astype(bool)
    candidates = _to_candidates(outcome, admission=admit, priority=target_free["auction_priority_bps"])
    decisions, equity, _ = replay_candidates(
        candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
        market_mode="perps", initial_wallet=1000.0,
    )
    if not decisions.empty:
        provenance = candidates.loc[:, ["candidate_id"]].reset_index(drop=True)
        provenance.index.name = "candidate_index"
        decisions = decisions.merge(provenance, on="candidate_index", how="left", validate="many_to_one")
        if decisions["candidate_id"].isna().any():
            raise AssertionError("portfolio decision lacks target-free candidate provenance")
        # ``_to_candidates`` has already excluded every invalid/unresolved
        # label before capacity is allocated.  The generic replay drops this
        # field, so restore the provenance explicitly for its metric adapter.
        decisions["policy_outcome_available"] = True
    decisions.to_parquet(out / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
    metric = _metrics(decisions, equity, arm, "2026_jun_jul")
    metric["dual_admitted_rows"] = int(admit.sum())
    metric["portfolio_candidate_rows"] = int(len(candidates))
    metric["admission_threshold_bps"] = ADMISSION_BPS
    metric["auction_priority"] = "bcf_mc1_expected_bps"
    accepted = decisions.loc[decisions.get("accepted", pd.Series(index=decisions.index, dtype=bool)).fillna(False).astype(bool)].copy()
    if accepted.empty:
        monthly = pd.DataFrame(columns=["arm", "month", "portfolio_accepted_trades", "net_ev_bps_per_trade", "net_sum_bps"])
    else:
        accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="raise")
        accepted["month"] = accepted["timestamp"].dt.strftime("%Y-%m")
        accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
        monthly = accepted.groupby("month", sort=True).agg(
            portfolio_accepted_trades=("net_bps", "size"),
            net_ev_bps_per_trade=("net_bps", "mean"), net_sum_bps=("net_bps", "sum"),
        ).reset_index()
        monthly.insert(0, "arm", arm)
    return metric, monthly


def _frozen_control(current: FamilySource, bcf: FamilySource, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_frame = current.scores.copy()
    current_frame["mc1_expected_bps"] = current.frozen_map.to_numpy(float)
    bcf_frame = bcf.scores.copy()
    bcf_frame["mc1_expected_bps"] = bcf.frozen_map.to_numpy(float)
    return _combine_predictions(current_frame, bcf_frame, labels)


def _append_delta(summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    control = summary.loc[summary["arm"].eq(baseline)]
    if len(control) != 1:
        raise AssertionError("matched refit control is missing from summary")
    row = control.iloc[0]
    for field in (
        "accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised",
        "worst_month_bps", "worst_week_bps", "max_drawdown", "sortino",
    ):
        if field in summary:
            summary[f"delta_vs_{baseline}_{field}"] = summary[field] - row[field]
    return summary


def _monthly_from_decisions(decisions: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Return the common constrained monthly metrics from an accepted ledger."""
    accepted = decisions.loc[
        decisions.get("accepted", pd.Series(index=decisions.index, dtype=bool)).fillna(False).astype(bool)
    ].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["arm", "month", "portfolio_accepted_trades", "net_ev_bps_per_trade", "net_sum_bps"])
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="raise")
    accepted["month"] = accepted["timestamp"].dt.strftime("%Y-%m")
    accepted["net_bps"] = pd.to_numeric(accepted["position_net_return"], errors="coerce") * 10_000.0
    monthly = accepted.groupby("month", sort=True).agg(
        portfolio_accepted_trades=("net_bps", "size"),
        net_ev_bps_per_trade=("net_bps", "mean"), net_sum_bps=("net_bps", "sum"),
    ).reset_index()
    monthly.insert(0, "arm", arm)
    return monthly


def _finalize_existing(args: argparse.Namespace, out: Path) -> Path:
    """Receipt a completed ledger when a prior run failed only after replay.

    This deliberately never refits, rescored, or rewrites candidate decisions.  It
    reads the immutable replay ledgers, hashes them, recreates only their missing
    aggregate summaries, and records that post-hoc scope in a separate receipt.
    """
    if not args.only_arms:
        raise ValueError("--finalize-existing requires the exact --only-arms selection used by the prior run")
    requested = tuple(name.strip() for name in args.only_arms.split(",") if name.strip())
    if not requested:
        raise ValueError("--finalize-existing received no arms")
    arms = ("C0_frozen_retained", *requested)
    metric_rows: list[dict[str, object]] = []
    monthly_rows: list[pd.DataFrame] = []
    input_hashes: dict[str, str] = {}
    for arm in arms:
        decision_path = out / f"{arm}_portfolio_decisions.parquet"
        equity_path = out / f"{arm}_portfolio_equity.parquet"
        if not decision_path.exists() or not equity_path.exists():
            raise FileNotFoundError(f"{arm}: completed decision/equity ledgers are required for finalization")
        decisions = pd.read_parquet(decision_path)
        equity = pd.read_parquet(equity_path)
        input_hashes[decision_path.name] = _sha256(decision_path)
        input_hashes[equity_path.name] = _sha256(equity_path)
        metric = _metrics(decisions, equity, arm, "2026_jun_jul")
        target_free_path = out / f"{arm}_target_free_admission.parquet"
        if target_free_path.exists():
            target_free = pd.read_parquet(target_free_path)
            input_hashes[target_free_path.name] = _sha256(target_free_path)
            if "dual_admitted" in target_free:
                metric["dual_admitted_rows"] = int(target_free["dual_admitted"].fillna(False).astype(bool).sum())
        metric_rows.append(metric)
        monthly_rows.append(_monthly_from_decisions(decisions, arm))
    summary = pd.DataFrame(metric_rows)
    refit_baseline = "C0_refit_core_postfeb"
    if refit_baseline in set(summary["arm"]):
        summary = _append_delta(summary, refit_baseline)
        delta_baseline: str | None = refit_baseline
    else:
        delta_baseline = None
    summary.to_parquet(out / "portfolio_summary.parquet", index=False, compression="zstd")
    summary.to_csv(out / "portfolio_summary.csv", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    receipt = {
        "schema": "canonical_sr_e2_mc1_input_ablation_finalization_v1",
        "scope": "post-hoc aggregate finalization from completed immutable no-order ledgers; no refit, score, admission, auction, policy, or execution mutation",
        "only_arms": list(requested),
        "input_ledger_sha256": input_hashes,
        "delta_baseline": delta_baseline,
        "frozen_control_role": "context-only when no C0_refit_core_postfeb ledger was materialized",
        "exchange_calls": 0,
    }
    (out / "finalization_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return out


def run(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if out.exists():
        if not args.finalize_existing:
            raise FileExistsError(f"immutable output exists: {out}")
        return _finalize_existing(args, out)
    if _utc(args.start) != EVAL_START or _utc(args.end) != EVAL_END:
        raise ValueError("this matched rerun fixes the held window to June--July 2026")
    out.mkdir(parents=True)
    current = _load_family(args.current, "current_v5")
    bcf = _load_family(args.bcf, "bcf")
    labels = _candidate_labels(current, bcf)
    union = _score_union(current, bcf)
    feature_route = union.loc[
        union["__decision_ts__"].ge(FEATURE_START) & union["__decision_ts__"].lt(EVAL_END)
    ].copy().reset_index(drop=True)
    feature_cache = args.feature_cache.resolve() if args.feature_cache else out / "target_free_15m_features.parquet"
    e2_features, feature_coverage = _load_or_materialize_features(feature_route, feature_cache)
    # The E2 source contains only the period beginning at FEATURE_START.  Add
    # it back to the full family universe by identity; earlier rows remain an
    # explicitly unavailable optional mapper input, never a candidate filter.
    e2, e2_audit = _prequential_e2(union.loc[union["__decision_ts__"].ge(FEATURE_START)].copy(), labels, e2_features, start=FEATURE_START, end=EVAL_END)
    sr_probe, sr_coverage = sr._merge_causal_sr(current.scores.copy(), args.sr_root)
    # Validate the causal source once before each family-local one-to-one merge.
    del sr_probe
    current_scores, current_full = _augment_family(current, args.sr_root, e2, args.positioning_root, args.profile_root, args.anchor_root, args.anchor_variant)
    bcf_scores, bcf_full = _augment_family(bcf, args.sr_root, e2, args.positioning_root, args.profile_root, args.anchor_root, args.anchor_variant)
    del current_scores, bcf_scores

    arms: dict[str, tuple[str, ...]] = {
        "C0_refit_core_postfeb": (),
        "C1_refit_core_plus_causal_sr": (*sr.SR_FEATURES, SR_AVAILABLE),
        "C2_refit_core_plus_15m_e2": (E2_OUTPUT, E2_AVAILABLE),
        "C3_refit_core_plus_causal_sr_15m_e2": (*sr.SR_FEATURES, SR_AVAILABLE, E2_OUTPUT, E2_AVAILABLE),
    }
    if args.component_ablation:
        arms.update({
            "C1a_refit_core_plus_sr_demotion": (*SR_DEMOTION_FEATURES, SR_AVAILABLE),
            "C1b_refit_core_plus_sr_support": (*SR_SUPPORT_FEATURES, SR_AVAILABLE),
        })
    if args.positioning_root is not None:
        arms["C4_refit_core_plus_causal_sr_oi_positioning"] = (*sr.SR_FEATURES, SR_AVAILABLE, *OI_POSITIONING_FEATURES, OI_POSITIONING_AVAILABLE)
    if args.profile_root is not None:
        arms["C5_refit_core_plus_causal_sr_profile_geometry"] = (*sr.SR_FEATURES, SR_AVAILABLE, *PROFILE_GEOMETRY_HEADS, PROFILE_GEOMETRY_AVAILABLE)
    if args.anchor_root is not None:
        arms["C6_refit_core_plus_causal_anchor"] = (*ANCHOR_ENTRY_HEADS, ANCHOR_AVAILABLE)
        arms["C7_refit_core_plus_causal_sr_anchor"] = (*sr.SR_FEATURES, SR_AVAILABLE, *ANCHOR_ENTRY_HEADS, ANCHOR_AVAILABLE)
    if args.only_arms:
        requested = tuple(name.strip() for name in args.only_arms.split(",") if name.strip())
        unknown = set(requested).difference(arms)
        if unknown:
            raise ValueError(f"unknown --only-arms entries: {sorted(unknown)}")
        arms = {name: arms[name] for name in requested}
    metric_rows: list[dict[str, object]] = []
    monthly_rows: list[pd.DataFrame] = []
    fold_rows: list[pd.DataFrame] = []

    frozen_tf, frozen_outcome = _frozen_control(current, bcf, labels)
    frozen_tf = frozen_tf.loc[frozen_tf["__decision_ts__"].ge(EVAL_START) & frozen_tf["__decision_ts__"].lt(EVAL_END)].copy()
    frozen_outcome = frozen_outcome.loc[frozen_outcome["__decision_ts__"].ge(EVAL_START) & frozen_outcome["__decision_ts__"].lt(EVAL_END)].copy()
    frozen_tf.to_parquet(out / "C0_frozen_retained_target_free_admission.parquet", index=False, compression="zstd")
    metric, monthly = _replay(frozen_tf, frozen_outcome, "C0_frozen_retained", out)
    metric_rows.append(metric)
    monthly_rows.append(monthly)

    for arm, extras in arms.items():
        current_pred, current_audit = _refit_family(current_full, family="current_v5", extras=extras, start=EVAL_START, end=EVAL_END)
        bcf_pred, bcf_audit = _refit_family(bcf_full, family="bcf", extras=extras, start=EVAL_START, end=EVAL_END)
        current_pred.to_parquet(out / f"{arm}_current_mc1_predictions.parquet", index=False, compression="zstd")
        bcf_pred.to_parquet(out / f"{arm}_bcf_mc1_predictions.parquet", index=False, compression="zstd")
        target_free, outcome = _combine_predictions(current_pred, bcf_pred, labels)
        target_free.to_parquet(out / f"{arm}_target_free_admission.parquet", index=False, compression="zstd")
        metric, monthly = _replay(target_free, outcome, arm, out)
        metric_rows.append(metric)
        monthly_rows.append(monthly)
        audit = pd.concat([current_audit, bcf_audit], ignore_index=True)
        audit.insert(0, "arm", arm)
        fold_rows.append(audit)

    summary = pd.DataFrame(metric_rows)
    refit_baseline = "C0_refit_core_postfeb"
    if refit_baseline in set(summary["arm"]):
        summary = _append_delta(summary, refit_baseline)
    summary.to_parquet(out / "portfolio_summary.parquet", index=False, compression="zstd")
    summary.to_csv(out / "portfolio_summary.csv", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_parquet(out / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat(fold_rows, ignore_index=True).to_parquet(out / "mc1_fold_audit.parquet", index=False, compression="zstd")
    feature_coverage.to_parquet(out / "target_free_15m_feature_coverage.parquet", index=False, compression="zstd")
    e2_audit.to_parquet(out / "e2_prequential_audit.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "canonical_sr_e2_mc1_input_ablation_v1",
        "scope": "offline matched MC1-input challenger; no live/canonical/policy/execution mutation",
        "held_period": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "frozen_reference": "August-17 separately materialised current-v5 and BCF MC1_d2 receipts at dual +50 bps",
        "delta_baseline": (
            "C0_refit_core_postfeb only; it shares the exact retained score/policy surface and post-February score history with all retrained variants"
            if refit_baseline in set(summary["arm"])
            else "not materialized; C0_frozen_retained is context-only for a narrowed --only-arms run"
        ),
        "warmup_limitation": "The exact pre-February BCF score ledger used by the frozen reference was pruned. Frozen C0 is reported as context; no refit arm is claimed bit-identical to it.",
        "families": {
            "current": {"path": str(args.current), "sha256": _sha256(args.current)},
            "bcf": {"path": str(args.bcf), "sha256": _sha256(args.bcf)},
        },
        "sr": {
            "path": str(args.sr_root), "manifest_sha256": _sha256(args.sr_root / "run_manifest.json"),
            "fields": list(sr.SR_FEATURES), "contract": "causal OOF outputs only; missingness plus availability flag is a mapper feature, never eligibility",
        },
        "oi_positioning": None if args.positioning_root is None else {
            "path": str(args.positioning_root), "manifest_sha256": _sha256(args.positioning_root / "run_manifest.json"),
            "fields": list(OI_POSITIONING_FEATURES),
            "contract": "strictly prior OI observations and OOF directional positioning-head outputs; optional missingness is a mapper feature",
        },
        "profile_geometry": None if args.profile_root is None else {
            "path": str(args.profile_root), "manifest_sha256": _sha256(args.profile_root / "run_manifest.json"),
            "fields": list(PROFILE_GEOMETRY_HEADS),
            "contract": "2025-trained causal profile/channel source heads; optional missingness plus availability is a mapper feature, never eligibility",
        },
        "anchor_discovery": None if args.anchor_root is None else {
            "path": str(args.anchor_root), "manifest_sha256": _sha256(args.anchor_root / "run_manifest.json"),
            "variant": _assert_causal_anchor_root(args.anchor_root, args.anchor_variant)[1],
            "fields": list(ANCHOR_ENTRY_HEADS),
            "contract": "2025-selected, causal event-anchor forward-path heads; target-free 2026 snapshots plus availability are mapper inputs only and never eligibility",
        },
        "e2": {
            "target_free_feature_cache": {"path": str(feature_cache), "sha256": _sha256(feature_cache)},
            "features": list(FIFTEEN_MINUTE_FEATURE_KEYS), "target": "source-aligned rich policy_net_bps with its embedded 100-bps cost exactly once",
            "model": "LightGBM L1, depth=4, leaves=15, 350 trees, lr=.03, L2=4, seed=1729",
            "fit": "four-month monthly direct model; labels resolve before held month; E2 values for an MC1 train row are prior-month predictions only",
            "availability": "target-free causal 15m source status plus finite-feature gate; unavailable outputs stay missing and do not change candidates",
        },
        "mc1": {
            "target": "absolute rich policy_net_bps, clipped p02/p98 in each family/fold",
            "model": "retained HGB d2/80/.04/L2=20/min_leaf=100/seed=1729",
            "fit": "family-specific, all retained post-February rows with labels resolved before held month",
            "shift": "21-day 10%-trimmed score-band residual from labels available before daily bucket",
        },
        "admission": "BCF MC1 >= +50 AND current-v5 MC1 >= +50; priority is BCF MC1 EV",
        "portfolio": "existing global controlled 7x/10%-slot, 2-new, 8-concurrent, 80%-wallet auction; invalid outcomes excluded before capacity",
        "arms": {arm: list(extras) for arm, extras in arms.items()},
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=CURRENT)
    parser.add_argument("--bcf", type=Path, default=BCF)
    parser.add_argument("--sr-root", type=Path, default=SR_ROOT)
    parser.add_argument("--positioning-root", type=Path, help="optional causal OI-positioning OOF head root")
    parser.add_argument("--profile-root", type=Path, help="optional 2025-trained causal profile/channel OOF head root")
    parser.add_argument("--anchor-root", type=Path, help="optional 2025-selected causal Anchor Discovery OOF head root")
    parser.add_argument("--anchor-variant", help="must equal the source's 2025-selected variant; prevents 2026 selection leakage")
    parser.add_argument(
        "--component-ablation", action="store_true",
        help="add causal S/R demotion-only and support-only C1 decomposition arms",
    )
    parser.add_argument(
        "--only-arms",
        help="comma-separated subset of non-frozen arms; useful for isolated immutable follow-up runs",
    )
    parser.add_argument(
        "--finalize-existing", action="store_true",
        help="write missing aggregate receipts from completed immutable replay ledgers; never refit or rescore",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--feature-cache", type=Path, default=None,
        help="verified target-free causal 15m cache; identity must exactly match the declared route",
    )
    parser.add_argument("--start", default=EVAL_START.isoformat())
    parser.add_argument("--end", default=EVAL_END.isoformat())
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
