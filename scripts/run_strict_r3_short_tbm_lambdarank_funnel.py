#!/usr/bin/env python3
"""Sequential strict-OOS short base ablation for triple-barrier ordinals.

This experiment isolates *base* supervision and LambdaRank query geometry.  It
uses the frozen P0/F90 causal feature contract and never exposes H12 path
labels to the feature matrix.  Candidate rows remain target-free until after
the feature identity join; incomplete paths are excluded from supervised
fitting, never recoded as grade-zero failures.

Funnel
------
1. On Apr--Jun 2024 only, screen the predeclared first-touch ordinal targets
   under exact timestamp-by-short-side queries.
2. Sweep five coarser, inference-valid time query groupings for the three best
   triple-barrier arms from step 1.
3. Confirm those finalists on Jul--Sep and freeze a winner by pooled
   development economics, then evaluate that single frozen winner on Oct--Dec.

The final Oct--Dec period is not used for target, gains, or query selection.
This is research-only and has no admission, portfolio, or live authority.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.query_candidate_definitions import (  # noqa: E402
    QueryDefinition,
    assign_query_ids,
    query_definitions_by_name,
)
from extreme_price_movements.tail_base_targets import grade_first_touch_tbm  # noqa: E402


SIDE = "short"
SEED = 17
MAX_STALE_TRIALS = 20
TAILS = (0.01, 0.02, 0.05, 0.10)
RECALL_FRACTIONS = (0.30, 0.40)

CANDIDATES = ROOT / "data_perp/artifacts/strict_r3_short_base_3m_train_3m_oos_2024_20260820_v1/short_target_free_candidate_population.parquet"
FEATURES = ROOT / "data_perp/artifacts/strict_r3_short_features_full2024_20260820_v1/canonical120_features.parquet"
TBM_LABELS = ROOT / "data_perp/artifacts/strict_r3_short_tbm_labels_full2024_20260821_v2"
POLICY_LABELS = ROOT / "data_perp/artifacts/strict_r3_short_policy_conversion_labels_12m_20260820_v1"
P0_CONFIG = ROOT / "config/strict_r3_short_p0_f90_base_v1.json"

FOLDS: tuple[tuple[str, str, str, str], ...] = (
    ("D1", "2024-01-01", "2024-04-01", "2024-07-01"),
    ("D2", "2024-04-01", "2024-07-01", "2024-10-01"),
    ("F3", "2024-07-01", "2024-10-01", "2025-01-01"),
)


@dataclass(frozen=True)
class TargetContract:
    name: str
    description: str
    family: Literal["p0_policy", "tbm_native", "tbm_economic", "tbm_economic_atr"]
    favourable_atr: int | None = None
    adverse_atr: int | None = None
    clear_floor_bps: float | None = None
    robust_floor_bps: float | None = None
    # Keep this before new ATR-only fields so the established positional
    # contracts below retain their exact semantics.
    label_gain_name: Literal["p0", "linear", "economic_step", "moderate_tail", "compressed_clear"] = "economic_step"
    clear_floor_atr: float | None = None
    robust_floor_atr: float | None = None
    grade_scheme: Literal["five", "collapsed_clear", "clear_terciles"] = "five"

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


TARGETS: tuple[TargetContract, ...] = (
    TargetContract(
        "P0_policy_ordinal_control",
        "Frozen P0/F90 policy-net ordinal LambdaRank control; exact one-minute parent policy labels.",
        "p0_policy", label_gain_name="p0",
    ),
    TargetContract(
        "TBM_nested_4_6_linear",
        "Nested first-touch ordinal: -6/-4/timeout/+4/+6 ATR; ties adverse.",
        "tbm_native", label_gain_name="linear",
    ),
    TargetContract(
        "TBM_nested_4_6_economic",
        "Same nested TP4/TP6/SL4/SL6 first-touch ordinal with economic-tail gains.",
        "tbm_native", label_gain_name="economic_step",
    ),
    TargetContract(
        "TBM_tp4_sl4_c50_r100",
        "TP4/SL4 first-touch ordinal; positive classes require >50 then >100 bps exact H12 net.",
        "tbm_economic", 4, 4, 50.0, 100.0, "economic_step",
    ),
    TargetContract(
        "TBM_tp6_sl4_c50_r100",
        "TP6/SL4 first-touch ordinal; positive classes require >50 then >100 bps exact H12 net.",
        "tbm_economic", 6, 4, 50.0, 100.0, "economic_step",
    ),
    TargetContract(
        "TBM_tp6_sl6_c50_r100",
        "TP6/SL6 first-touch ordinal; positive classes require >50 then >100 bps exact H12 net.",
        "tbm_economic", 6, 6, 50.0, 100.0, "economic_step",
    ),
    TargetContract(
        "TBM_tp6_sl4_c75_r150",
        "TP6/SL4 first-touch ordinal; stricter >75 then >150 bps economic clear thresholds.",
        "tbm_economic", 6, 4, 75.0, 150.0, "moderate_tail",
    ),
    TargetContract(
        "TBM_tp6_sl4_c100_r200",
        "TP6/SL4 first-touch ordinal; strict >100 then >200 bps economic clear thresholds.",
        "tbm_economic", 6, 4, 100.0, 200.0, "moderate_tail",
    ),
)

# The first-touch barriers above are already expressed in decision-time ATR.
# These contracts additionally express the *economic* robust-clear grades in
# net-ATR units.  They therefore test whether the cost-clearing grade has a
# consistent meaning between low- and high-volatility candidates.  Net bps
# remains the label value, so the fixed 100-bps policy cost is applied once.
ATR_NORMALIZED_TARGETS: tuple[TargetContract, ...] = (
    TargetContract(
        "TBM_tp4_sl4_atr_c050_r100",
        "TP4/SL4 first-touch ordinal; net economic grades at 0.50/1.00 decision ATR.",
        "tbm_economic_atr", 4, 4, clear_floor_atr=0.50, robust_floor_atr=1.00,
        label_gain_name="economic_step",
    ),
    TargetContract(
        "TBM_tp6_sl4_atr_c050_r100",
        "TP6/SL4 first-touch ordinal; net economic grades at 0.50/1.00 decision ATR.",
        "tbm_economic_atr", 6, 4, clear_floor_atr=0.50, robust_floor_atr=1.00,
        label_gain_name="economic_step",
    ),
    TargetContract(
        "TBM_tp6_sl4_atr_c075_r150",
        "TP6/SL4 first-touch ordinal; net economic grades at 0.75/1.50 decision ATR.",
        "tbm_economic_atr", 6, 4, clear_floor_atr=0.75, robust_floor_atr=1.50,
        label_gain_name="moderate_tail",
    ),
    TargetContract(
        "TBM_tp6_sl4_atr_c100_r200",
        "TP6/SL4 first-touch ordinal; net economic grades at 1.00/2.00 decision ATR.",
        "tbm_economic_atr", 6, 4, clear_floor_atr=1.00, robust_floor_atr=2.00,
        label_gain_name="moderate_tail",
    ),
    TargetContract(
        "TBM_tp6_sl6_atr_c050_r100",
        "TP6/SL6 first-touch ordinal; net economic grades at 0.50/1.00 decision ATR.",
        "tbm_economic_atr", 6, 6, clear_floor_atr=0.50, robust_floor_atr=1.00,
        label_gain_name="economic_step",
    ),
)

ALL_TARGETS: tuple[TargetContract, ...] = TARGETS + ATR_NORMALIZED_TARGETS


@dataclass(frozen=True)
class WeightSpec:
    """Training-only class balancing; never an inference feature."""

    name: str
    description: str
    family: Literal["uniform", "sqrt_inverse", "effective_number"]
    cap: float = 4.0
    beta: float | None = None

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


WEIGHT_SPECS: tuple[WeightSpec, ...] = (
    WeightSpec("W0_uniform", "Natural class prevalence.", "uniform"),
    WeightSpec(
        "W1_sqrt_inverse_cap4",
        "Square-root inverse-frequency class weights, normalized to mean one and capped at four.",
        "sqrt_inverse",
    ),
    WeightSpec(
        "W2_effective_number_b9999_cap4",
        "Effective-number class weights (beta=0.9999), normalized to mean one and capped at four.",
        "effective_number", beta=0.9999,
    ),
)

# These contracts remove the grade-2/3 bottleneck observed in the first ATR
# screen.  They retain adverse-first as grade zero and make all non-clearing
# paths one dense weak class.  The tercile variants use training-only quantiles
# of positive clear margins, so positive-tail labels have usable support.
ATR_COMPRESSION_TARGETS: tuple[TargetContract, ...] = (
    TargetContract(
        "TBM_tp4_sl4_atr_c050_collapsed",
        "TP4/SL4: adverse / weak-or-marginal / clear above 0.50 ATR.",
        "tbm_economic_atr", 4, 4, clear_floor_atr=0.50,
        grade_scheme="collapsed_clear", label_gain_name="compressed_clear",
    ),
    TargetContract(
        "TBM_tp6_sl4_atr_c025_collapsed",
        "TP6/SL4: adverse / weak-or-marginal / clear above 0.25 ATR.",
        "tbm_economic_atr", 6, 4, clear_floor_atr=0.25,
        grade_scheme="collapsed_clear", label_gain_name="compressed_clear",
    ),
    TargetContract(
        "TBM_tp6_sl4_atr_c050_collapsed",
        "TP6/SL4: adverse / weak-or-marginal / clear above 0.50 ATR.",
        "tbm_economic_atr", 6, 4, clear_floor_atr=0.50,
        grade_scheme="collapsed_clear", label_gain_name="compressed_clear",
    ),
    TargetContract(
        "TBM_tp6_sl4_atr_c075_collapsed",
        "TP6/SL4: adverse / weak-or-marginal / clear above 0.75 ATR.",
        "tbm_economic_atr", 6, 4, clear_floor_atr=0.75,
        grade_scheme="collapsed_clear", label_gain_name="compressed_clear",
    ),
    TargetContract(
        "TBM_tp6_sl4_atr_c025_clear_terciles",
        "TP6/SL4: adverse / weak / three training-only terciles above a 0.25-ATR clear floor.",
        "tbm_economic_atr", 6, 4, clear_floor_atr=0.25,
        grade_scheme="clear_terciles", label_gain_name="economic_step",
    ),
    TargetContract(
        "TBM_tp6_sl4_atr_c050_clear_terciles",
        "TP6/SL4: adverse / weak / three training-only terciles above a 0.50-ATR clear floor.",
        "tbm_economic_atr", 6, 4, clear_floor_atr=0.50,
        grade_scheme="clear_terciles", label_gain_name="economic_step",
    ),
)

ALL_TARGETS = ALL_TARGETS + ATR_COMPRESSION_TARGETS

EXACT_QUERY = "q0_exact_timestamp_side"
QUERY_SWEEP = (
    "q1_cycle_2h_side",
    "q1_cycle_4h_side",
    "q1_cycle_6h_side",
    "q1_cycle_8h_side",
    "q1_cycle_12h_side",
)

LABEL_GAINS: dict[str, list[float]] = {
    "p0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "linear": [0.0, 1.0, 2.0, 3.0, 4.0],
    "economic_step": [0.0, 0.25, 1.0, 3.0, 7.0],
    "moderate_tail": [0.0, 0.25, 1.0, 4.0, 10.0],
    "compressed_clear": [0.0, 1.0, 5.0],
}

P0_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "n_estimators": 140,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 350,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.80,
    "reg_lambda": 8.0,
    "random_state": SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "lambdarank_norm": True,
    "lambdarank_truncation_level": 32,
}


@dataclass
class FoldData:
    """Fold-local target-free feature population, kept compact for fitting."""

    train_population: pd.DataFrame
    test_population: pd.DataFrame
    coverage: pd.Series


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp | pd.Series:
    return pd.to_datetime(value, utc=True, errors="raise")


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start.normalize().replace(day=1), end, freq="MS", inclusive="left"))


def _f90_fields() -> tuple[list[str], dict[str, Any]]:
    config = json.loads(P0_CONFIG.read_text())
    selection = ROOT / str(config["feature_contract"]["selection_artifact"])
    payload = json.loads(selection.read_text())
    fields = [str(item) for item in payload["feature_sets"]["90"]]
    if len(fields) != 90 or len(set(fields)) != 90:
        raise ValueError("frozen P0/F90 feature selection must contain 90 unique fields")
    return fields, {
        "config": str(P0_CONFIG), "config_sha256": _sha256(P0_CONFIG),
        "selection": str(selection), "selection_sha256": _sha256(selection),
        "field_count": len(fields),
    }


def _load_candidates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "entry_executable", "eligibility_reason"]
    frame = pd.read_parquet(CANDIDATES, columns=columns, filters=[("__ts__", ">=", start), ("__ts__", "<", end)])
    frame["__ts__"] = _utc(frame["__ts__"])
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("target-free short candidate identities are invalid")
    if frame.entry_executable.isna().any():
        raise ValueError("target-free entry-executable state is null")
    return frame


def _load_features(start: pd.Timestamp, end: pd.Timestamp, fields: list[str]) -> pd.DataFrame:
    # Reading a parquet schema must not materialise the complete full-year
    # panel; doing so defeats the fold-bounded memory design below.
    schema = set(pq.ParquetFile(FEATURES).schema_arrow.names)
    missing = sorted(set(fields).difference(schema))
    if missing:
        raise ValueError(f"F90 fields are missing from full-2024 feature panel: {missing[:8]}")
    columns = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", *fields]
    frame = pd.read_parquet(FEATURES, columns=columns, filters=[("__ts__", ">=", start), ("__ts__", "<", end)])
    frame["__ts__"] = _utc(frame["__ts__"])
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    if frame.candidate_id.duplicated().any():
        raise ValueError("feature identity is not unique")
    return frame


def _load_h12_labels(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "__label_available_at__",
        "label_valid", "target_invalid", "invalid_reason", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
        "atr_bps", "first_tp4_minute", "first_tp6_minute", "first_sl4_minute", "first_sl6_minute",
    ]
    for month in _months(start, end):
        part = TBM_LABELS / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        if not part.exists():
            raise FileNotFoundError(part)
        parts.append(pd.read_parquet(part, columns=columns))
    frame = pd.concat(parts, ignore_index=True)
    for col in ("__ts__", "__decision_ts__", "__label_available_at__"):
        frame[col] = _utc(frame[col])
    if frame.candidate_id.duplicated().any():
        raise ValueError("H12 label identity is not unique")
    return frame


def _load_policy_labels(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", "policy_path_valid",
        "policy_label_available_at", "p0_canonical_net_bps",
    ]
    for month in _months(start, end):
        part = POLICY_LABELS / "parts" / f"month={month:%Y-%m}" / "side=short.parquet"
        if not part.exists():
            raise FileNotFoundError(part)
        parts.append(pd.read_parquet(part, columns=columns))
    frame = pd.concat(parts, ignore_index=True)
    for col in ("__ts__", "__decision_ts__", "policy_label_available_at"):
        frame[col] = _utc(frame[col])
    if frame.candidate_id.duplicated().any():
        raise ValueError("policy label identity is not unique")
    return frame


def _load_fold_ledger(start: pd.Timestamp, end: pd.Timestamp, fields: list[str]) -> pd.DataFrame:
    """Load one six-month train/OOS block only.

    The full 2024 F90 panel is intentionally not held in memory: it is about
    1.5m rows, while every experiment only needs a 3m train + 3m OOS block.
    This keeps the label boundary and candidate identity contract identical
    while making the funnel practical on the research workstation.
    """
    candidates = _load_candidates(start, end)
    features = _load_features(start, end, fields)
    h12 = _load_h12_labels(start, end)
    policy = _load_policy_labels(start, end)
    keys = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]
    ledger = candidates.merge(features, on=keys, how="left", validate="one_to_one")
    ledger = ledger.merge(h12, on=keys, how="left", validate="one_to_one")
    ledger = ledger.merge(policy, on=keys, how="left", validate="one_to_one")
    if len(ledger) != len(candidates) or ledger.candidate_id.duplicated().any():
        raise AssertionError("target-free identity cardinality changed during label joins")
    if ledger.loc[ledger.entry_executable.astype(bool), fields].isna().all(axis=1).any():
        raise ValueError("entry-executable candidate lacks its target-free feature row")
    return ledger


def _prepare_fold_data(fold: tuple[str, str, str, str], fields: list[str]) -> FoldData:
    """Project an I/O ledger into compact train/test populations then release it."""
    _, train_start_s, oos_start_s, oos_end_s = fold
    train_start, oos_start, oos_end = (pd.Timestamp(value, tz="UTC") for value in (train_start_s, oos_start_s, oos_end_s))
    ledger = _load_fold_ledger(train_start, oos_end, fields)
    coverage = _coverage_fields(ledger.loc[ledger["__ts__"].ge(train_start) & ledger["__ts__"].lt(oos_start)], fields)
    label_columns = [
        "label_valid", "target_invalid", "__label_available_at__", "t4_tp6_sl4_gross_bps",
        "t4_tp6_sl4_net_bps", "atr_bps", "first_tp4_minute", "first_tp6_minute", "first_sl4_minute",
        "first_sl6_minute", "policy_path_valid", "policy_label_available_at", "p0_canonical_net_bps",
    ]
    train_columns = ["candidate_id", "__ts__", "side_name", *label_columns, *fields]
    test_columns = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name", *label_columns, *fields]
    train_population = ledger.loc[
        ledger["__ts__"].ge(train_start) & ledger["__ts__"].lt(oos_start) & ledger.entry_executable.astype(bool), train_columns
    ].copy()
    test_population = ledger.loc[
        ledger["__ts__"].ge(oos_start) & ledger["__ts__"].lt(oos_end) & ledger.entry_executable.astype(bool), test_columns
    ].copy()
    del ledger
    gc.collect()
    if train_population.empty or test_population.empty:
        raise ValueError("fold projection has no target-free executable train or OOS rows")
    # Features are causal numeric inputs.  Store them compactly once per fold;
    # conversion/imputation below remains fit on training rows only.
    for population in (train_population, test_population):
        values = population.loc[:, fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        population.loc[:, fields] = values.astype(np.float32)
    return FoldData(train_population=train_population, test_population=test_population, coverage=coverage)


def _valid_h12(frame: pd.DataFrame) -> pd.Series:
    return frame.label_valid.fillna(False).astype(bool) & ~frame.target_invalid.fillna(True).astype(bool)


def _valid_policy(frame: pd.DataFrame) -> pd.Series:
    return frame.policy_path_valid.fillna(False).astype(bool) & pd.to_numeric(frame.p0_canonical_net_bps, errors="coerce").notna()


def _economic_tbm_grade(frame: pd.DataFrame, contract: TargetContract) -> pd.Series:
    """Vectorised five-grade directional first-touch target.

    Grade 0 is adverse-first (same-bar ties adverse); grade 1 is timeout or a
    favourable touch that does not clear cost; grades 2--4 are increasingly
    robust cost-clearing favourable paths.  Exact H12 TP6/SL4 net is used only
    as a *label* margin, never as a model field.
    """
    if contract.favourable_atr not in {4, 6} or contract.adverse_atr not in {4, 6}:
        raise ValueError("economic triple-barrier contract must use materialised 4 or 6 ATR touches")
    valid = _valid_h12(frame).to_numpy(bool)
    tp = pd.to_numeric(frame[f"first_tp{contract.favourable_atr}_minute"], errors="coerce").to_numpy(float)
    sl = pd.to_numeric(frame[f"first_sl{contract.adverse_atr}_minute"], errors="coerce").to_numpy(float)
    net = pd.to_numeric(frame.t4_tp6_sl4_net_bps, errors="coerce").to_numpy(float)
    margin = net.copy()
    if contract.family == "tbm_economic_atr":
        atr_bps = pd.to_numeric(frame.atr_bps, errors="coerce").to_numpy(float)
        if np.any(valid & (~np.isfinite(atr_bps) | (atr_bps <= 0.0))):
            raise ValueError("valid H12 row is missing a positive decision-time ATR bps")
        margin = net / atr_bps
        if contract.clear_floor_atr is None:
            raise ValueError("ATR-normalized economic target needs a clear floor")
        clear_floor = float(contract.clear_floor_atr)
        robust_floor = float(contract.robust_floor_atr) if contract.robust_floor_atr is not None else clear_floor
    else:
        clear_floor = float(contract.clear_floor_bps)
        robust_floor = float(contract.robust_floor_bps)
    if np.any(valid & (~np.isfinite(tp) | ~np.isfinite(sl) | ~np.isfinite(net))):
        raise ValueError("valid H12 row is missing first-touch or economic-label data")
    if np.any(valid & ((tp < -1) | (sl < -1) | (tp >= 720) | (sl >= 720))):
        raise ValueError("materialised first-touch minute lies outside H12")
    adverse = valid & (sl >= 0.0) & ((tp < 0.0) | (sl <= tp))
    favourable = valid & ~adverse & (tp >= 0.0) & ((sl < 0.0) | (tp < sl))
    grade = np.full(len(frame), -1, dtype=np.int8)
    grade[valid] = 1
    grade[adverse] = 0
    if contract.grade_scheme == "collapsed_clear":
        grade[favourable & (margin > clear_floor)] = 2
        return pd.Series(grade, index=frame.index, dtype="int8")
    if contract.grade_scheme == "clear_terciles":
        clear = favourable & (margin > clear_floor)
        if int(clear.sum()) < 30:
            raise ValueError("insufficient clear rows for stable training-only ATR-margin terciles")
        lower, upper = np.quantile(margin[clear], [1.0 / 3.0, 2.0 / 3.0])
        grade[clear & (margin <= lower)] = 2
        grade[clear & (margin > lower) & (margin <= upper)] = 3
        grade[clear & (margin > upper)] = 4
        return pd.Series(grade, index=frame.index, dtype="int8")
    if contract.grade_scheme != "five":
        raise ValueError(f"unknown economic target grade scheme: {contract.grade_scheme}")
    # Positive barrier-first paths that fail to clear the fixed cost remain a
    # weak class.  This is intentional: barrier event and economic outcome are
    # not silently treated as the same thing.
    grade[favourable & (margin > 0.0) & (margin <= clear_floor)] = 2
    grade[favourable & (margin > clear_floor) & (margin <= robust_floor)] = 3
    grade[favourable & (margin > robust_floor)] = 4
    return pd.Series(grade, index=frame.index, dtype="int8")


def _target(frame: pd.DataFrame, contract: TargetContract) -> tuple[pd.Series, pd.Series]:
    if contract.family == "p0_policy":
        valid = _valid_policy(frame)
        net = pd.to_numeric(frame.p0_canonical_net_bps, errors="coerce")
        grade = pd.Series(-1, index=frame.index, dtype="int8")
        grade.loc[valid] = np.digitize(net.loc[valid], [-400., -200., 0., 100., 200., 400.], right=True).astype(np.int8)
        return grade, valid
    valid = _valid_h12(frame)
    if contract.family == "tbm_native":
        grade = pd.Series(
            grade_first_touch_tbm(
                frame.first_tp4_minute.fillna(-1).to_numpy(), frame.first_tp6_minute.fillna(-1).to_numpy(),
                frame.first_sl4_minute.fillna(-1).to_numpy(), frame.first_sl6_minute.fillna(-1).to_numpy(), valid.to_numpy(bool),
            ), index=frame.index, dtype="int8",
        )
        return grade, valid
    return _economic_tbm_grade(frame, contract), valid


def _matrix(frame: pd.DataFrame, fields: list[str], medians: pd.Series) -> np.ndarray:
    values = frame.loc[:, fields].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    missing = np.isnan(values)
    if missing.any():
        values[missing] = np.take(medians.to_numpy(dtype=np.float32), np.where(missing)[1])
    if not np.isfinite(values).all():
        raise ValueError("training-only F90 imputation left missing model inputs")
    return values


def _coverage_fields(frame: pd.DataFrame, fields: list[str]) -> pd.Series:
    values = frame.loc[frame.entry_executable.astype(bool), fields].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.notna().mean()


def _rankable_train(frame: pd.DataFrame, grade: pd.Series, query: QueryDefinition, fields: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    x = frame.loc[:, ["candidate_id", "__ts__", "side_name", *fields]].copy()
    x["_grade"] = grade.astype(np.int8)
    x["_query"] = assign_query_ids(x, query)
    stats = x.groupby("_query", observed=True)["_grade"].agg(["size", "nunique"])
    keep = stats.index[(stats["size"] >= 2) & (stats["nunique"] >= 2)]
    x = x.loc[x._query.isin(keep)].sort_values(["_query", "candidate_id"], kind="stable").reset_index(drop=True)
    groups = x.groupby("_query", observed=True, sort=False).size().to_numpy(np.int32)
    if x.empty or int(groups.sum()) != len(x) or (groups < 2).any():
        raise ValueError("rank query construction produced no rankable training rows")
    return x, groups


def _weight_by_name(name: str) -> WeightSpec:
    for spec in WEIGHT_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown predeclared class-weight contract: {name}")


def _class_sample_weights(labels: pd.Series, spec: WeightSpec) -> tuple[np.ndarray, dict[str, Any]]:
    """Return capped, mean-one train-only class weights for LambdaRank."""
    classes = labels.astype(np.int16)
    counts = classes.value_counts().sort_index()
    if counts.empty or (counts <= 0).any():
        raise ValueError("class-weight computation needs non-empty observed classes")
    if spec.family == "uniform":
        per_class = pd.Series(1.0, index=counts.index, dtype=float)
    elif spec.family == "sqrt_inverse":
        per_class = np.sqrt(float(len(classes)) / (float(len(counts)) * counts.astype(float)))
    elif spec.family == "effective_number":
        if spec.beta is None or not (0.0 < spec.beta < 1.0):
            raise ValueError("effective-number weighting needs beta strictly between zero and one")
        beta = float(spec.beta)
        per_class = (1.0 - beta) / (-np.expm1(np.log(beta) * counts.astype(float)))
    else:
        raise ValueError(f"unsupported class weighting family: {spec.family}")
    raw_sample = classes.map(per_class).to_numpy(dtype=np.float64)
    # A second normalisation *after* clipping can push rare-class weights
    # above the declared cap.  Solve for a single scale factor inside the
    # bounded transform instead.  The resulting weights have mean one and
    # remain in the documented [1/cap, cap] interval.
    lower = 1.0 / float(spec.cap)
    upper = float(spec.cap)

    def _mean_at(scale: float) -> float:
        return float(np.mean(np.clip(raw_sample * scale, lower, upper)))

    lo_scale, hi_scale = 0.0, 1.0
    while _mean_at(hi_scale) < 1.0:
        hi_scale *= 2.0
        if hi_scale > 1e12:
            raise RuntimeError("could not normalise bounded class weights")
    for _ in range(80):
        mid_scale = 0.5 * (lo_scale + hi_scale)
        if _mean_at(mid_scale) < 1.0:
            lo_scale = mid_scale
        else:
            hi_scale = mid_scale
    sample = np.clip(raw_sample * hi_scale, lower, upper)
    if not np.isclose(float(np.mean(sample)), 1.0, atol=1e-8):
        raise AssertionError("bounded class weights failed mean-one normalisation")
    audit = {
        "weight_spec": spec.manifest(),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "class_weights": {str(key): float(value) for key, value in per_class.items()},
        "sample_weight_mean": float(np.mean(sample)),
        "sample_weight_min": float(np.min(sample)),
        "sample_weight_max": float(np.max(sample)),
    }
    return sample.astype(np.float32), audit


def _fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, fields: list[str], contract: TargetContract,
    query: QueryDefinition, weight_spec: WeightSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    grade, valid = _target(train, contract)
    if not valid.all() or (grade < 0).any():
        raise AssertionError("training population must be target-valid before LambdaRank fit")
    ordered, groups = _rankable_train(train, grade, query, fields)
    medians = ordered.loc[:, fields].replace([np.inf, -np.inf], np.nan).median()
    if medians.isna().any():
        raise ValueError("a selected F90 field has no training-only median")
    gains = LABEL_GAINS[contract.label_gain_name]
    model = lgb.LGBMRanker(**dict(P0_PARAMS, label_gain=gains))
    x_train = _matrix(ordered, fields, medians)
    x_test = _matrix(test, fields, medians)
    sample_weight, weight_audit = _class_sample_weights(ordered._grade, weight_spec)
    model.fit(x_train, ordered._grade.to_numpy(np.int8), group=groups, sample_weight=sample_weight)
    score = np.asarray(model.predict(x_test), dtype=np.float32)
    audit = {
        "train_target_rows": int(len(train)), "rankable_train_rows": int(len(ordered)),
        "rankable_train_fraction": float(len(ordered) / len(train)), "rank_query_count": int(len(groups)),
        "query_size_median": float(np.median(groups)), "query_size_p90": float(np.quantile(groups, .90)),
        "target_grade_counts": {str(k): int(v) for k, v in ordered._grade.value_counts().sort_index().items()},
        "class_weighting": weight_audit,
        "training_medians": {key: float(value) for key, value in medians.items()},
    }
    # sklearn's wrapper can retain its native Dataset past a normal Python
    # scope exit.  This funnel fits many isolated arms sequentially, so free
    # it explicitly before the next target/query trial rather than allowing a
    # native-memory accumulation to determine which hypotheses are reached.
    model.booster_.free_dataset()
    del model, ordered, x_train, x_test, sample_weight
    gc.collect()
    return score, audit


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    return float(left.loc[valid].corr(right.loc[valid], method="spearman")) if int(valid.sum()) >= 3 else float("nan")


def _query_evaluation(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    y = pd.to_numeric(frame[outcome], errors="coerce")
    for _, local in frame.assign(_outcome=y).groupby("__ts__", sort=False):
        local = local.loc[local._outcome.notna()]
        if len(local) < 2:
            continue
        record: dict[str, float] = {"n": float(len(local)), "ic": _spearman(local.score, local._outcome)}
        pred = local.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
        truth = local.sort_values(["_outcome", "candidate_id"], ascending=[False, True], kind="stable")
        for fraction in RECALL_FRACTIONS:
            k = max(1, int(math.ceil(len(local) * fraction)))
            record[f"recall_{int(fraction * 100)}"] = float(len(set(pred.head(k).candidate_id).intersection(truth.head(k).candidate_id)) / k)
        rows.append(record)
    query = pd.DataFrame(rows)
    if query.empty:
        return {"query_count": 0.0, "ic": float("nan"), "positive_ic_fraction": float("nan"), **{f"recall_{int(f * 100)}": float("nan") for f in RECALL_FRACTIONS}}
    result = {
        "query_count": float(len(query)),
        "ic": float(np.average(query.ic.fillna(0.0), weights=query.n)),
        "positive_ic_fraction": float((query.ic > 0.0).mean()),
    }
    result.update({f"recall_{int(f * 100)}": float(np.average(query[f"recall_{int(f * 100)}"], weights=query.n)) for f in RECALL_FRACTIONS})
    return result


def _scope_metrics(prediction: pd.DataFrame, *, trial_id: str, phase: str, fold: str, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h12 = prediction.loc[_valid_h12(prediction)].copy()
    policy = prediction.loc[_valid_policy(prediction)].copy()
    record: dict[str, Any] = {
        "trial_id": trial_id, "phase": phase, "fold": fold, "scope": scope,
        "scored_rows": int(len(prediction)), "h12_resolved_rows": int(len(h12)), "policy_resolved_rows": int(len(policy)),
        "h12_resolved_fraction": float(len(h12) / max(1, len(prediction))),
        "policy_resolved_fraction": float(len(policy) / max(1, len(prediction))),
    }
    for prefix, data, outcome in (("h12", h12, "t4_tp6_sl4_net_bps"), ("policy", policy, "p0_canonical_net_bps")):
        q = _query_evaluation(data, outcome)
        record.update({f"{prefix}_{key}": value for key, value in q.items()})
        net = pd.to_numeric(data[outcome], errors="coerce")
        for fraction in TAILS:
            selected = data.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable").head(max(1, int(math.ceil(len(prediction) * fraction))))
            value = pd.to_numeric(selected[outcome], errors="coerce")
            record[f"{prefix}_top{int(fraction * 100)}_net_bps"] = float(value.mean()) if len(value) else float("nan")
            record[f"{prefix}_top{int(fraction * 100)}_rows"] = int(len(value))
    rows.append(record)
    return rows


def _development_score(metrics: pd.DataFrame) -> pd.Series:
    """Predeclared portable selection score using only development folds."""
    x = metrics.loc[metrics.scope.eq("all")].copy()
    # Exact policy economic quality is primary.  Query IC and the worst
    # development month are explicit robustness tiebreakers rather than an
    # unbounded multi-metric HPO objective.
    monthly = metrics.loc[metrics.scope.str.match(r"^2024-", na=False)].groupby("trial_id", observed=True)["policy_top5_net_bps"].min().rename("worst_month")
    x = x.join(monthly, on="trial_id")
    x["selection_score"] = (
        x.policy_top5_net_bps
        + 50.0 * x.policy_ic.fillna(-1.0)
        - np.maximum(0.0, -x.worst_month.fillna(-1e6))
    )
    return x.set_index("trial_id")["selection_score"]


def _run_trial(
    data: FoldData, fields: list[str], *, contract: TargetContract, query_name: str,
    fold: tuple[str, str, str, str], phase: str, out: Path, persist_prediction: bool,
    weight_spec: WeightSpec,
) -> tuple[str, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    fold_name, train_start_s, oos_start_s, oos_end_s = fold
    train_start, oos_start, oos_end = (pd.Timestamp(v, tz="UTC") for v in (train_start_s, oos_start_s, oos_end_s))
    query = query_definitions_by_name([query_name])[0]
    grade, valid = _target(data.train_population, contract)
    label_available = data.train_population.policy_label_available_at if contract.family == "p0_policy" else data.train_population["__label_available_at__"]
    train = data.train_population.loc[
        valid & pd.to_datetime(label_available, utc=True, errors="coerce").lt(oos_start)
    ].copy()
    train["_grade"] = grade.loc[train.index].to_numpy(np.int8)
    test = data.test_population
    if train.empty or test.empty:
        raise ValueError(f"{fold_name} has no strict train or target-free OOS population")
    coverage = data.coverage
    failed = [field for field in fields if float(coverage[field]) < .90]
    if failed:
        raise ValueError(f"{fold_name} F90 coverage failure: {failed[:8]}")
    score, audit = _fit_predict(train, test, fields, contract, query, weight_spec)
    trial_id = f"{phase}__{fold_name}__{contract.name}__{query_name}__{weight_spec.name}"
    prediction = test.loc[:, [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        "label_valid", "target_invalid", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
        "policy_path_valid", "p0_canonical_net_bps",
    ]].copy()
    prediction["score"] = score
    metrics = _scope_metrics(prediction, trial_id=trial_id, phase=phase, fold=fold_name, scope="all")
    for month, local in prediction.groupby(prediction["__ts__"].dt.strftime("%Y-%m"), sort=True):
        metrics.extend(_scope_metrics(local, trial_id=trial_id, phase=phase, fold=fold_name, scope=str(month)))
    summary = {
        "trial_id": trial_id, "phase": phase, "fold": fold_name, "target": contract.name, "query": query_name,
        "weighting": weight_spec.manifest(),
        "train_start": train_start.isoformat(), "oos_start": oos_start.isoformat(), "oos_end": oos_end.isoformat(),
        "training_rows": int(len(train)), "oos_rows": int(len(test)), "coverage_min": float(coverage.min()),
        "target": contract.manifest(), "ranker": {**P0_PARAMS, "label_gain": LABEL_GAINS[contract.label_gain_name]}, "audit": audit,
    }
    if persist_prediction:
        prediction.to_parquet(out / f"oos_predictions_{trial_id}.parquet", index=False, compression="zstd")
    return trial_id, prediction, summary, metrics


def _contract_by_name(name: str) -> TargetContract:
    for contract in ALL_TARGETS:
        if contract.name == name:
            return contract
    raise KeyError(f"unknown predeclared target contract: {name}")


def _fold_by_name(name: str) -> tuple[str, str, str, str]:
    for fold in FOLDS:
        if fold[0] == name:
            return fold
    raise KeyError(f"unknown predeclared fold: {name}")


def _run_worker(
    *, out: Path, target_name: str, query_name: str, fold_name: str,
    phase: str, persist_prediction: bool, weight_name: str = "W0_uniform",
) -> Path:
    """Run one arm in a fresh process, releasing LightGBM native memory at exit."""
    if out.exists():
        raise FileExistsError(f"worker output already exists: {out}")
    out.mkdir(parents=True)
    fields, _ = _f90_fields()
    contract = _contract_by_name(target_name)
    weight_spec = _weight_by_name(weight_name)
    fold = _fold_by_name(fold_name)
    data = _prepare_fold_data(fold, fields)
    trial_id, prediction, summary, metrics = _run_trial(
        data, fields, contract=contract, query_name=query_name, fold=fold,
        phase=phase, out=out, persist_prediction=persist_prediction, weight_spec=weight_spec,
    )
    pd.DataFrame(metrics).to_parquet(out / "metrics.parquet", index=False, compression="zstd")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out / "worker_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_tbm_lambdarank_worker_v1", "trial_id": trial_id,
        "target": contract.manifest(), "query": query_name, "fold": fold_name,
        "phase": phase, "weighting": weight_spec.manifest(), "persist_prediction": persist_prediction,
    }, indent=2) + "\n")
    del prediction, data
    gc.collect()
    return out


def _invoke_worker(
    *, root: Path, contract: TargetContract, query_name: str, fold: tuple[str, str, str, str],
    phase: str, persist_prediction: bool, weight_spec: WeightSpec = WEIGHT_SPECS[0],
) -> tuple[str, dict[str, Any], list[dict[str, Any]], pd.DataFrame | None]:
    """Call a one-trial worker with process isolation for LightGBM datasets."""
    trial_id = f"{phase}__{fold[0]}__{contract.name}__{query_name}__{weight_spec.name}"
    worker_out = root / "workers" / trial_id
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker-out", str(worker_out),
        "--target", contract.name, "--query", query_name, "--fold", fold[0], "--phase", phase,
        "--weight", weight_spec.name,
    ]
    if persist_prediction:
        command.append("--persist-prediction")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode:
        tail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"isolated worker failed for {trial_id}:\n{tail}")
    summary = json.loads((worker_out / "summary.json").read_text())
    metrics = pd.read_parquet(worker_out / "metrics.parquet").to_dict("records")
    prediction_path = worker_out / f"oos_predictions_{trial_id}.parquet"
    prediction = pd.read_parquet(prediction_path) if persist_prediction else None
    return trial_id, summary, metrics, prediction


def run(*, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True)
    fields, feature_audit = _f90_fields()
    trial_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    stale = 0
    best_score = -np.inf

    # Round A: target and label-gain screen on the first development block.
    round_a: list[str] = []
    for contract in TARGETS:
        trial_id, summary, metrics, _ = _invoke_worker(
            root=out, contract=contract, query_name=EXACT_QUERY, fold=FOLDS[0], phase="A_target", persist_prediction=False,
        )
        trial_rows.append(summary); metric_rows.extend(metrics); round_a.append(trial_id)
        value = float(_development_score(pd.DataFrame(metric_rows)).get(trial_id, -np.inf))
        if value > best_score + 1e-12:
            best_score, stale = value, 0
        else:
            stale += 1
        if stale >= MAX_STALE_TRIALS:
            break
        gc.collect()

    all_metrics = pd.DataFrame(metric_rows)
    score_a = _development_score(all_metrics)
    target_lookup = {contract.name: contract for contract in TARGETS}
    ranked_a = pd.DataFrame([row for row in trial_rows if row["phase"] == "A_target"])
    ranked_a["selection_score"] = ranked_a.trial_id.map(score_a)
    # The policy control is a control, not a candidate replacement.  Keep the
    # three strongest triple-barrier arms even if the control dominates it.
    tbm_a = ranked_a.loc[ranked_a.target.apply(lambda item: item["family"] != "p0_policy")]
    top_targets = tbm_a.sort_values(["selection_score", "trial_id"], ascending=[False, True], kind="stable").head(3)
    selected_contracts = [target_lookup[row.target["name"]] for _, row in top_targets.iterrows()]

    # Round B: grouping sweep.  The exact-time result above remains part of
    # the candidate set, so each selected target receives five new groupings.
    round_b_ids: list[str] = []
    for contract in selected_contracts:
        for query_name in QUERY_SWEEP:
            trial_id, summary, metrics, _ = _invoke_worker(
                root=out, contract=contract, query_name=query_name, fold=FOLDS[0], phase="B_query", persist_prediction=False,
            )
            trial_rows.append(summary); metric_rows.extend(metrics); round_b_ids.append(trial_id)
            value = float(_development_score(pd.DataFrame(metric_rows)).get(trial_id, -np.inf))
            if value > best_score + 1e-12:
                best_score, stale = value, 0
            else:
                stale += 1
            if stale >= MAX_STALE_TRIALS:
                break
            gc.collect()
        if stale >= MAX_STALE_TRIALS:
            break

    all_metrics = pd.DataFrame(metric_rows)
    scores_dev1 = _development_score(all_metrics)
    candidate_ids = [row["trial_id"] for row in trial_rows if row["phase"] in {"A_target", "B_query"} and row["target"]["family"] != "p0_policy"]
    top_trial_ids = sorted(candidate_ids, key=lambda value: (-float(scores_dev1.get(value, -np.inf)), value))[:3]
    by_trial = {row["trial_id"]: row for row in trial_rows}

    # Round C1: a separate three-month confirmation; it has no authority to
    # reconsider the target/query grid, only to select among the frozen three.
    confirmation_ids: list[str] = []
    for source_id in top_trial_ids:
        row = by_trial[source_id]
        target = target_lookup[row["target"]["name"]]
        trial_id, summary, metrics, _ = _invoke_worker(
            root=out, contract=target, query_name=row["query"], fold=FOLDS[1], phase="C_confirm", persist_prediction=False,
        )
        trial_rows.append(summary); metric_rows.extend(metrics); confirmation_ids.append(trial_id)
        gc.collect()
    confirm_metrics = pd.DataFrame(metric_rows)
    dev_metrics = confirm_metrics.loc[confirm_metrics.fold.isin(["D1", "D2"])].copy()
    # Pair D1 selection result with matched D2 confirmation and use a simple
    # mean-minus-worst rule.  Both blocks are development; F3 remains final.
    combined: list[tuple[float, str, dict[str, Any]]] = []
    for source_id, confirm_id in zip(top_trial_ids, confirmation_ids, strict=True):
        a = float(_development_score(confirm_metrics.loc[confirm_metrics.trial_id.eq(source_id)]).get(source_id, -np.inf))
        b = float(_development_score(confirm_metrics.loc[confirm_metrics.trial_id.eq(confirm_id)]).get(confirm_id, -np.inf))
        combined.append(((a + b) / 2.0 - max(0.0, -min(a, b)), source_id, by_trial[source_id]))
    _, winner_source_id, winner = sorted(combined, key=lambda item: (-item[0], item[1]))[0]
    winner_target = target_lookup[winner["target"]["name"]]
    winner_query = str(winner["query"])

    # F3: exactly one final untouched target/query configuration.
    final_id, final_summary, final_metrics, final_prediction = _invoke_worker(
        root=out, contract=winner_target, query_name=winner_query, fold=FOLDS[2], phase="F_final", persist_prediction=True,
    )
    if final_prediction is None:
        raise AssertionError("final isolated worker did not return its OOS prediction")
    trial_rows.append(final_summary); metric_rows.extend(final_metrics)

    pd.DataFrame(trial_rows).to_parquet(out / "trial_audit.parquet", index=False, compression="zstd")
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_parquet(out / "metrics_by_trial_scope.parquet", index=False, compression="zstd")
    final_prediction.to_parquet(out / "final_oos_predictions.parquet", index=False, compression="zstd")
    decision = {
        "schema": "strict_r3_short_tbm_lambdarank_funnel_v1",
        "status": "complete",
        "side": SIDE,
        "feature_contract": feature_audit,
        "candidate_source": str(CANDIDATES), "candidate_source_sha256": _sha256(CANDIDATES),
        "feature_source": str(FEATURES), "feature_source_sha256": _sha256(FEATURES),
        "triple_barrier_label_source": str(TBM_LABELS), "triple_barrier_label_manifest_sha256": _sha256(TBM_LABELS / "run_manifest.json"),
        "policy_label_source": str(POLICY_LABELS), "policy_label_manifest_sha256": _sha256(POLICY_LABELS / "run_manifest.json"),
        "folds": [dict(name=name, train_start=train, oos_start=oos, oos_end=end) for name, train, oos, end in FOLDS],
        "targets": [contract.manifest() for contract in TARGETS], "query_sweep": [EXACT_QUERY, *QUERY_SWEEP],
        "selection": {
            "development_blocks": ["D1", "D2"], "final_untouched_block": "F3",
            "max_stale_trials": MAX_STALE_TRIALS, "stale_trials_at_stop": stale,
            "round_a_top_tbm": [item["target"]["name"] for item in top_targets.to_dict("records")],
            "round_b_finalists": top_trial_ids, "development_winner_source_trial": winner_source_id,
            "winner_target": winner_target.manifest(), "winner_query": winner_query, "final_trial": final_id,
        },
        "strictness": {
            "candidates": "target-free; entry-executable only for supervised fitting and scoring",
            "labels": "H12/policy labels are joined after features; invalid paths are excluded from fit",
            "training_gate": "label available strictly before each OOS fold",
            "evaluation": "OOS rows ranked before future H12/policy outcomes are used for metrics",
            "feature_contract": "frozen P0/F90; no target-driven feature selection in this funnel",
            "query_membership": "side-local decision-time timestamp/cycle buckets only",
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(decision, indent=2, default=str) + "\n")
    (out / "target_query_decision.json").write_text(json.dumps(decision["selection"], indent=2, default=str) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="immutable parent-funnel output")
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--query", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fold", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--weight", default="W0_uniform", help=argparse.SUPPRESS)
    parser.add_argument("--persist-prediction", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_out is not None:
        required = {"--target": args.target, "--query": args.query, "--fold": args.fold, "--phase": args.phase}
        missing = [flag for flag, value in required.items() if not value]
        if missing:
            parser.error("worker mode requires " + ", ".join(missing))
        print(_run_worker(
            out=args.worker_out.resolve(), target_name=str(args.target), query_name=str(args.query),
            fold_name=str(args.fold), phase=str(args.phase), persist_prediction=bool(args.persist_prediction),
            weight_name=str(args.weight),
        ))
        return
    if args.out is None:
        parser.error("--out is required outside worker mode")
    print(run(out=args.out.resolve()))


if __name__ == "__main__":
    main()
