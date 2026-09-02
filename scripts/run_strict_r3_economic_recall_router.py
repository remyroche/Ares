#!/usr/bin/env python3
"""Strict-OOF economic-recall router with a replacement A--E auxiliary suite.

This is deliberately a *base-router* experiment, not an amendment to the
live stack.  Its model inputs are exactly the sealed 120 causal base features.
Every policy or path quantity is read only after the target-free candidate
panel is fixed, and only on training rows whose 12-hour labels were resolved
before the reserve preceding the held month.

The complete auxiliary suite replaces earlier B0/efficiency/timing side
targets for this experiment:

* A: favourable reach before adverse movement;
* B: speed to favourable milestones, in train-only 20% quantiles;
* C: bounded positive path/policy potential;
* D: MFE efficiency ratios; and
* E: adverse-first probabilities (their score authority is inverted).

No member of that suite is persisted as a score-time feature.  Models use the
same causal 120-field matrix; their train-distribution rank references are
combined into an economic-recall routing score.  This keeps the architecture
usable at inference after a future model-selection/freeze step.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, early_stopping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "strict_r3_economic_recall_router_ae_v1"
SEED = 1729
DEFAULT_TRAIN_MONTHS = 6
DEFAULT_RESERVE_DAYS = 28
DEFAULT_N_JOBS = min(4, os.cpu_count() or 1)
DEFAULT_ROUTE_FRACTIONS = (0.30, 0.40, 0.50)
MAX_LABEL_HORIZON_HOURS = 12
FEATURE_HASH = "b2c2725813d30c02ee298f82292d848d0e1133eb01be3f1398003163523ec2a1"
DEFAULT_BUNDLE = ROOT / "data_perp/artifacts/strict_r3_lockstep_successor28_long_aug1_7_current_spread_20260812_v1/bundles/cutoff=20260801/upstream/monthly_upstream_bundle.joblib"
DEFAULT_FEATURE_ROOT = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_threeway_targetfree_20260824_v1/target_free_monthly"
DEFAULT_AUX_ROOT = ROOT / "data_perp/artifacts/strict_r3_o3v2_recall_router_aux_labels_20240201_20260731_20260825_v3"
DEFAULT_POLICY = ROOT / "data_perp/artifacts/strict_r3_enhanced_base_rich_policy_labels_reconciled_20260823_v1/canonical_reconciled_policy_labels.parquet"
DEFAULT_MONTHS = tuple(pd.date_range("2025-11-01", "2026-07-01", freq="MS", tz="UTC"))
ALL_GROUPS = ("A", "B", "C", "D", "E")

# Frozen router-selection utility from the research specification.  These
# constants belong to evaluation only: no target-free score ever receives a
# policy outcome or this realised-utility quantity as an input.
PRIMARY_UTILITY_FLOOR_BPS = 50.0
# Frozen Router50 selection utility for the active optimisation contract.
# Alternative p/c geometries remain training-label candidates; these constants
# define the outcome-joined evaluation receipt used to select among them.
PRIMARY_UTILITY_CAP_BPS = 225.0
PRIMARY_UTILITY_POWER = 0.75
PRIMARY_UTILITY_TIMESTAMP_GAMMA = 0.5
PRIMARY_UTILITY_TIMESTAMP_CAP = 2.0

# Direct policy-net grade controls retained only as the router's *main* target.
# A--E replaces all former auxiliary targets, not the economically meaningful
# main routing objective itself.
# Threshold-based policy relevance controls.  The remaining named primary
# targets below are built by ``_primary_target`` because their bins are either
# train-quantile-derived or combine a declared path condition with policy
# value.  All of them remain training labels only.
PRIMARY_TARGETS: dict[str, tuple[float, ...]] = {
    "P1_net_0_50_100_150_300": (0.0, 50.0, 100.0, 150.0, 300.0),
    "P2_net_25_50_100_200_400": (25.0, 50.0, 100.0, 200.0, 400.0),
    "P3_net_50_75_125_200_350": (50.0, 75.0, 125.0, 200.0, 350.0),
    "P4_net_0_100_200_300_500": (0.0, 100.0, 200.0, 300.0, 500.0),
    # The explicit +100-floor alternative from the shared specification:
    # <=10 is zero relevance; 10--150 begins the first positive grade.
    "P5_plus100_floor_10_150_200_250_400": (10.0, 150.0, 200.0, 250.0, 400.0),
    "P7_downside_200_0_50_100_200": (-200.0, 0.0, 50.0, 100.0, 200.0),
}

PRIMARY_SPECIAL_TARGETS = {
    "P8_path_policy",
    "P8u_floor0_cap250", "P8u_floor0_cap400", "P8u_floor0_cap600",
    "P8u_floor50_cap250", "P8u_floor50_cap400", "P8u_floor50_cap600",
    "P8u_floor100_cap250", "P8u_floor100_cap400", "P8u_floor100_cap600",
    "P9_positive_quantiles",
    "T1_tbm_clearance_a_1_2_3_4_6",
    "T1_tbm_clearance_b_2_3_4_6_8",
    "T1_tbm_clearance_c_1_2_4_6_10",
    "T2_tbm_path_quality",
    # Direct binary recall heads.  They deliberately answer only whether a
    # fully-resolved policy outcome clears the declared economic hurdle.  The
    # surrounding LambdaRank query contract is unchanged, which makes these
    # heads a clean test of threshold-specific Router recall rather than a
    # different candidate or evaluation universe.
    "R50_policy_net_gt50",
    "R100_policy_net_gt100",
    "R200_policy_net_gt200",
}
UTILITY_GEOMETRY_TARGETS = tuple(
    f"U50_p{power:03d}_c{cap}"
    for power in (50, 75, 100)
    for cap in (150, 225, 300)
)
ALL_PRIMARY_TARGETS = tuple(sorted((*PRIMARY_TARGETS, *PRIMARY_SPECIAL_TARGETS, *UTILITY_GEOMETRY_TARGETS)))



@dataclass(frozen=True)
class ScoreReference:
    """Immutable empirical CDF fitted only on a model's training predictions."""

    sorted_values: np.ndarray
    source: str

    @classmethod
    def fit(cls, values: Sequence[float], source: str) -> "ScoreReference":
        work = np.asarray(values, dtype=float)
        work = np.sort(work[np.isfinite(work)], kind="stable")
        if len(work) < 100:
            raise ValueError(f"{source}: insufficient finite rank-reference support")
        return cls(work.astype(np.float32), source)

    def cdf(self, values: Sequence[float]) -> np.ndarray:
        value = np.asarray(values, dtype=float)
        result = np.full(len(value), np.nan, dtype=np.float32)
        valid = np.isfinite(value)
        left = np.searchsorted(self.sorted_values, value[valid], side="left")
        right = np.searchsorted(self.sorted_values, value[valid], side="right")
        result[valid] = ((left + right) * .5 + .5) / float(len(self.sorted_values))
        return np.clip(result, 0.0, 1.0).astype(np.float32)


@dataclass
class FittedTarget:
    name: str
    group: str
    inverse: bool
    fields: tuple[str, ...]
    medians: np.ndarray
    model: LGBMRanker
    reference: ScoreReference
    classes: int
    rows: int
    queries: int
    weight_summary: dict[str, float]

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw = self.model.predict(_matrix(frame, self.fields, self.medians)).astype(np.float32)
        rank = self.reference.cdf(raw)
        if self.inverse:
            rank = 1.0 - rank
        return raw, rank.astype(np.float32)


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _validate_base_fields(fields: Sequence[str]) -> tuple[str, ...]:
    fields = tuple(str(field) for field in fields)
    digest = hashlib.sha256("\n".join(fields).encode()).hexdigest()
    if len(fields) != 120 or len(set(fields)) != 120 or digest != FEATURE_HASH:
        raise AssertionError(f"sealed 120-field contract mismatch: {digest}")
    return fields


def _base_fields(bundle: Path, sealed_feature_contract: Path | None = None) -> tuple[str, ...]:
    """Read the sealed base contract from its declared immutable source.

    Offline router research never consumes the upstream booster itself: it
    requires only the exact feature order.  A completed router receipt can
    therefore serve as an explicit fallback if an old joblib is no longer
    deserialisable in the current Python environment.  The fallback is opt-in
    and must still match the hard-coded 120-field SHA; no name-based feature
    discovery is permitted.
    """
    if sealed_feature_contract is not None:
        payload = json.loads(sealed_feature_contract.read_text())
        fields = payload.get("feature_contract") if isinstance(payload, dict) else None
        # The canonical schema-v2 contract owns the same ordered long-side
        # 120-field parent list under ``base_fields_by_side``.  Accept that
        # sealed representation explicitly so a historical, zero-byte joblib
        # carrier is never needed merely to recover feature order.
        if fields is None and isinstance(payload, dict):
            base_by_side = payload.get("base_fields_by_side")
            if isinstance(base_by_side, dict):
                fields = base_by_side.get("long")
        if not isinstance(fields, list):
            raise ValueError(
                "sealed feature contract must contain feature_contract or "
                "base_fields_by_side.long"
            )
        return _validate_base_fields(fields)
    fitted = joblib.load(bundle)
    return _validate_base_fields(fitted.base_fields)


def _feature_hash(fields: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(field) for field in fields).encode()).hexdigest()


def _read_feature_contract(path: Path) -> tuple[str, ...]:
    """Read a frozen external causal feature contract without name discovery.

    The recall-router feature study may deliberately search a broader causal
    universe than the deployed 120-field base contract.  Its winner still has
    to be an explicit, ordered and immutable list: this helper never accepts
    a glob, a prefix, or a runtime feature-discovery rule.
    """
    payload = json.loads(path.read_text())
    requested = payload.get("feature_contract", payload) if isinstance(payload, dict) else payload
    if not isinstance(requested, list) or not requested:
        raise ValueError("full feature contract must be a non-empty JSON list or feature_contract object")
    fields = tuple(str(field) for field in requested)
    if len(fields) != len(set(fields)):
        raise ValueError("full feature contract contains duplicate fields")
    return fields


def _selected_fields(
    bundle: Path, feature_list: Path | None, sealed_feature_contract: Path | None = None,
    full_feature_contract: Path | None = None,
) -> tuple[str, ...]:
    """Return an ordered, sealed subset of the canonical 120 causal fields.

    Feature selection may remove fields but cannot introduce a new field,
    change input ordering, or quietly substitute a name-based contract.  A
    standalone JSON list or an object containing ``feature_contract`` is
    accepted so the MDA receipt itself can be passed directly to this runner.
    """
    # The default path remains intentionally locked to the deployed 120-field
    # base source.  Full-universe selection is opt-in and must name a frozen
    # contract generated by the strict causal screening receipt.
    if full_feature_contract is not None:
        if feature_list is not None or sealed_feature_contract is not None:
            raise ValueError(
                "full-feature-contract is mutually exclusive with sealed 120-field "
                "feature selection inputs"
            )
        return _read_feature_contract(full_feature_contract)
    base = _base_fields(bundle, sealed_feature_contract)
    if feature_list is None:
        return base
    payload = json.loads(feature_list.read_text())
    requested = payload.get("feature_contract", payload) if isinstance(payload, dict) else payload
    if not isinstance(requested, list) or not requested:
        raise ValueError("feature-list must be a non-empty JSON list or feature_contract object")
    requested_names = [str(field) for field in requested]
    if len(requested_names) != len(set(requested_names)):
        raise ValueError("feature-list contains duplicate fields")
    unexpected = sorted(set(requested_names) - set(base))
    if unexpected:
        raise ValueError(f"feature-list contains fields outside the sealed 120-field base contract: {unexpected}")
    result = tuple(field for field in base if field in set(requested_names))
    if len(result) != len(requested_names):
        raise AssertionError("selected field count differs after canonical ordering")
    return result


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[str]:
    periods = pd.period_range(start.to_period("M"), (end - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M")
    for period in periods:
        yield period.strftime("%Y-%m")


def _normalise_feature_roots(root: Path | Sequence[Path]) -> tuple[Path, ...]:
    """Return one or more explicit, non-duplicated causal feature stores.

    A historical Router replay can join retained, immutable *monthly* stores
    only when their date ranges are disjoint.  Accepting an ordered sequence
    makes that provenance explicit and prevents an accidental fallback to a
    different same-month panel.  It is not a feature union: every selected
    field must come from the one unique source that owns that month.
    """
    roots = (root,) if isinstance(root, Path) else tuple(Path(value) for value in root)
    if not roots:
        raise ValueError("at least one feature root is required")
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate feature root")
    return roots


def _window_features(root: Path | Sequence[Path], start: pd.Timestamp, end: pd.Timestamp, columns: Sequence[str]) -> pd.DataFrame:
    roots = _normalise_feature_roots(root)
    chunks: list[pd.DataFrame] = []
    for token in _months_between(start, end):
        available: list[Path] = []
        for candidate_root in roots:
            month_root = candidate_root / f"month={token}"
            # ``scores_features`` is the legacy 120-field receipt.  The new
            # full-universe materializer deliberately emits a differently named
            # target-free panel, which is equally valid for this offline router
            # only after its fields have been frozen by the selector.
            source = month_root / "scores_features.parquet"
            if not source.exists():
                source = month_root / "causal_feature_universe.parquet"
            if source.exists():
                available.append(source)
        if not available:
            continue
        if len(available) != 1:
            raise AssertionError(
                f"{token}: feature month is present in multiple explicit roots; "
                "a Router fold must use one unambiguous source panel"
            )
        source = available[0]
        part = pd.read_parquet(source, columns=list(columns))
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        chunks.append(part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)].copy())
    if not chunks:
        return pd.DataFrame(columns=list(columns))
    result = pd.concat(chunks, ignore_index=True)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("target-free feature window has duplicate candidate identity")
    return result


def _validate_aux_reuse_source(
    source: Path, fields: Sequence[str], months: Sequence[pd.Timestamp], args: argparse.Namespace,
) -> None:
    """Allow only exact-contract, target-free A--E score reuse.

    A--E labels do not depend on the main direct-policy grade.  Reusing their
    already OOF-scored ranks makes a main-target sweep computationally cheaper
    without leaking a policy outcome or changing an auxiliary prediction.  The
    source must otherwise be the same strict training/feature contract.
    """
    contract_path = source / "run_contract.json"
    manifest_path = source / "run_manifest.json"
    if not contract_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("auxiliary reuse source must be a completed router artifact")
    source_contract = json.loads(contract_path.read_text())
    source_manifest = json.loads(manifest_path.read_text())
    if source_contract.get("schema") != SCHEMA or source_manifest.get("status") != "complete":
        raise AssertionError("auxiliary reuse source is not a completed strict-R3 router artifact")
    expected_months = [month.strftime("%Y-%m") for month in months]
    expected = {
        "feature_contract": list(fields), "feature_contract_sha256": _feature_hash(fields),
        "months": expected_months, "train_months": args.train_months,
        "reserve_days": args.reserve_days, "train_cap": args.train_cap,
        "n_jobs": args.n_jobs, "aux_groups": list(args.aux_groups),
    }
    mismatches = [key for key, value in expected.items() if source_contract.get(key) != value]
    if mismatches:
        raise AssertionError(f"auxiliary reuse contract mismatch: {mismatches}")


def _reused_aux_scores(
    source: Path, month: pd.Timestamp, held: pd.DataFrame, groups: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    """Load only immutable auxiliary ranks and prove exact held identity parity."""
    score_path = source / "target_free_scores" / f"month={month:%Y-%m}.parquet"
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    source_score = pd.read_parquet(score_path)
    identity = ["candidate_id", "__decision_ts__", "side_name"]
    required = {f"router_group_{group}_rank" for group in groups}
    raw_aux = [
        column for column in source_score
        if column.startswith("router__") or column.startswith("router_group_")
    ]
    if required - set(raw_aux):
        raise AssertionError(f"{score_path}: missing group ranks {sorted(required - set(raw_aux))}")
    reuse = source_score.loc[:, [*identity, *raw_aux]].copy()
    reuse["__decision_ts__"] = pd.to_datetime(reuse["__decision_ts__"], utc=True, errors="raise")
    if reuse["candidate_id"].duplicated().any():
        raise AssertionError(f"{score_path}: duplicate candidate identity")
    held_keys = held.loc[:, identity].copy()
    merged = held_keys.merge(reuse, on=identity, how="left", validate="one_to_one", indicator=True)
    if len(merged) != len(held_keys) or not merged["_merge"].eq("both").all():
        raise AssertionError(f"{score_path}: reused auxiliary score identity differs from held target-free panel")
    return merged.drop(columns=["_merge"]), _sha256_file(score_path)


def _window_aux(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    wanted = [
        "candidate_id", "__decision_ts__", "side_name", "aux_label_available_ts", "aux_path_valid",
        "aux_reached_2atr", "aux_reached_3atr", "aux_reached_4atr", "aux_reached_6atr",
        "aux_reached_8atr", "aux_reached_10atr",
        "aux_reached_1atr", "aux_time_to_1atr_h", "aux_time_to_2atr_h", "aux_time_to_3atr_h",
        "aux_time_to_4atr_h", "aux_time_to_6atr_h", "aux_time_to_8atr_h", "aux_time_to_10atr_h",
        "aux_time_to_50bps_h", "aux_time_to_100bps_h",
        "aux_peak_mfe_atr_h12", "aux_peak_mfe_bps_h12", "aux_mae_atr_12h", "aux_path_length_bps_h12",
        "aux_mfe_over_path_length_h12", "aux_mfe_over_abs_mae_h12",
        "aux_reached_adverse_3atr", "aux_time_to_adverse_3atr_h",
        "aux_reached_adverse_4atr", "aux_time_to_adverse_4atr_h",
        "aux_reached_stop_loss", "aux_first_policy_hit_interval_h",
    ]
    for token in _months_between(start, end):
        source = root / "parts" / f"month={token}" / "auxiliary_path_labels.parquet"
        if not source.exists():
            continue
        part = pd.read_parquet(source, columns=wanted)
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        part["aux_label_available_ts"] = pd.to_datetime(part["aux_label_available_ts"], utc=True, errors="coerce")
        chunks.append(part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)].copy())
    if not chunks:
        return pd.DataFrame(columns=wanted)
    result = pd.concat(chunks, ignore_index=True)
    if result["candidate_id"].duplicated().any():
        raise AssertionError("auxiliary label window has duplicate candidate identity")
    return result


def _policy_window(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cols = [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_exit_reason", "policy_label_available_ts",
    ]
    frame = pd.read_parquet(path, columns=cols)
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
    # The policy ledger is identity keyed, while source date selection is only
    # possible after joining a target-free timestamp.  Keep it narrow here.
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy ledger contains duplicate candidate identities")
    return frame


def _matrix(frame: pd.DataFrame, fields: Sequence[str], medians: Sequence[float] | None = None) -> np.ndarray:
    values = frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = values.median().fillna(0.0).to_numpy(np.float32) if medians is None else np.asarray(medians, dtype=np.float32)
    if len(fill) != len(fields):
        raise AssertionError("model median and frozen feature contracts differ")
    return values.fillna(pd.Series(fill, index=fields)).fillna(0.0).to_numpy(np.float32)


def _medians(frame: pd.DataFrame, fields: Sequence[str]) -> np.ndarray:
    return frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0).to_numpy(np.float32)


_ROW_WEIGHT_SCHEMES = (
    # W0: retained incumbent objective.
    "uniform", "clipped_excess", "sqrt_excess",
    # W1--W3: predeclared timestamp-balanced alternatives.
    "timestamp_uniform",
    "economic_025", "economic_050", "economic_100",
    "positive_125", "positive_150", "positive_200",
    # W4 is deliberately narrow and must only be launched after an
    # individually helpful W2/W3 arm.  It has bounded maximum authority.
    "combined_e025_p125", "combined_e050_p125",
)


def _query_weights(
    frame: pd.DataFrame, *, scheme: str = "uniform", floor_bps: float = 100.0,
    cap_bps: float = 250.0, primary_utility: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, float]]:
    """Sort full timestamp queries and assign each timestamp equal total loss weight.

    The two economic schemes change only *within-query* relative weight.  They
    are normalised again per timestamp, so an active hour with many candidates
    cannot dominate the objective merely because it has more rows.
    """
    if scheme not in _ROW_WEIGHT_SCHEMES:
        raise ValueError(f"unknown row-weight scheme: {scheme}")
    if cap_bps <= 0:
        raise ValueError("row-weight cap must be positive")
    columns = ["candidate_id", "__decision_ts__"]
    if scheme not in {"uniform", "timestamp_uniform"}:
        columns.append("policy_net_bps")
    work = frame.loc[:, columns].copy()
    if primary_utility is not None:
        utility = np.asarray(primary_utility, dtype=np.float32)
        if len(utility) != len(work):
            raise AssertionError("primary utility and query-weight rows differ")
        work["__primary_utility__"] = utility
    elif scheme.startswith("economic_") or scheme.startswith("combined_"):
        raise ValueError(f"{scheme} requires a declared primary utility")
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    sizes = work.groupby("__decision_ts__", sort=False).size()
    work["__size__"] = work["__decision_ts__"].map(sizes).astype(np.int32)
    work = work.loc[work["__size__"].ge(2)].copy()
    groups = work.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)
    if scheme in {"uniform", "timestamp_uniform"}:
        relative = np.ones(len(work), dtype=float)
    elif scheme in {"clipped_excess", "sqrt_excess"}:
        net = pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float)
        excess = np.clip(np.maximum(net - float(floor_bps), 0.0), 0.0, float(cap_bps)) / float(cap_bps)
        # Keep all resolved rows in the loss while giving economically larger
        # outcomes up to 2x relative authority.  A zero weight would turn the
        # requested robustness ablation into a different positive-only target.
        relative = 1.0 + (np.sqrt(excess) if scheme == "sqrt_excess" else excess)
    elif scheme.startswith("economic_"):
        alpha = float(scheme.rsplit("_", 1)[1]) / 100.0
        utility = pd.to_numeric(work["__primary_utility__"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(float)
        relative = 1.0 + alpha * utility
    elif scheme.startswith("positive_"):
        multiplier = float(scheme.rsplit("_", 1)[1]) / 100.0
        net = pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float)
        relative = np.where(net > 50.0, multiplier, 1.0)
    elif scheme.startswith("combined_"):
        token = scheme.removeprefix("combined_")
        alpha_token, positive_token = token.split("_")
        alpha = float(alpha_token.removeprefix("e")) / 100.0
        multiplier = float(positive_token.removeprefix("p")) / 100.0
        utility = pd.to_numeric(work["__primary_utility__"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(float)
        net = pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float)
        relative = (1.0 + alpha * utility) * np.where(net > 50.0, multiplier, 1.0)
    else:
        raise AssertionError(scheme)
    work["__relative__"] = relative
    normaliser = work.groupby("__decision_ts__", sort=False)["__relative__"].transform("sum").to_numpy(float)
    raw_weights = np.divide(relative, normaliser, out=np.zeros(len(relative), dtype=float), where=normaliser > 0)
    # LightGBM's regularisation is sensitive to a global re-scaling of every
    # row weight.  Preserve the incumbent's mean weight of one while retaining
    # *exactly equal timestamp authority*: all timestamp totals have the same
    # multiplier.  The audit below also stores the unscaled, specification
    # level weights whose within-timestamp total is exactly one.
    effective_scale = len(raw_weights) / max(float(work["__decision_ts__"].nunique()), 1.0)
    weights = raw_weights * effective_scale
    def _quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_q50": float(np.quantile(values, .50)),
            f"{prefix}_q90": float(np.quantile(values, .90)),
            f"{prefix}_q95": float(np.quantile(values, .95)),
            f"{prefix}_q99": float(np.quantile(values, .99)),
            f"{prefix}_max": float(np.max(values)),
        }
    raw_totals = pd.Series(raw_weights).groupby(work["__decision_ts__"], sort=False).sum().to_numpy(float)
    effective_totals = pd.Series(weights).groupby(work["__decision_ts__"], sort=False).sum().to_numpy(float)
    summary = {
        "raw_timestamp_total_min": float(np.min(raw_totals)),
        "raw_timestamp_total_max": float(np.max(raw_totals)),
        "effective_timestamp_total_min": float(np.min(effective_totals)),
        "effective_timestamp_total_max": float(np.max(effective_totals)),
        "effective_scale": float(effective_scale),
        **_quantiles(raw_weights, "raw_weight"),
        **_quantiles(weights, "effective_weight"),
    }
    return work, groups, weights.astype(np.float32), summary


def _deterministic_query_cap(frame: pd.DataFrame, *, cap: int) -> pd.DataFrame:
    """Keep whole timestamp queries without target-aware selection."""
    if len(frame) <= cap:
        return frame
    timestamps = pd.Series(frame["__decision_ts__"].drop_duplicates().sort_values().to_numpy())
    token = timestamps.astype(str).map(lambda value: int(hashlib.sha256((str(SEED) + value).encode()).hexdigest()[:16], 16))
    ordered = pd.DataFrame({"__decision_ts__": timestamps, "__hash__": token}).sort_values("__hash__", kind="stable")
    keep: list[pd.Timestamp] = []
    running = 0
    sizes = frame.groupby("__decision_ts__", sort=False).size()
    for stamp in ordered["__decision_ts__"]:
        n = int(sizes.loc[stamp])
        if running and running + n > cap:
            continue
        keep.append(stamp)
        running += n
        if running >= cap:
            break
    result = frame.loc[frame["__decision_ts__"].isin(keep)].copy()
    if len(result) < 5_000:
        raise AssertionError("query-level cap left inadequate training support")
    return result


def _speed_grade(values: pd.Series) -> np.ndarray:
    """Train-only five 20%-quantiles for reached paths; unreached is grade 0."""
    raw = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    finite = np.isfinite(raw)
    grade = np.zeros(len(raw), dtype=np.int32)
    if finite.sum() < 500:
        raise ValueError("speed target has insufficient reached-path support")
    # Rank avoids unstable duplicate qcut edges and gives exactly train-local
    # 20% partitions up to ties.  Faster time receives a higher ordered grade.
    rank = pd.Series(raw[finite]).rank(method="average", pct=True).to_numpy(float)
    grade[finite] = 5 - np.minimum(4, np.floor(rank * 5.0 - 1e-12).astype(np.int32))
    return grade


def _quintile_grade(values: pd.Series) -> np.ndarray:
    raw = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    finite = np.isfinite(raw)
    if finite.sum() < 500:
        raise ValueError("continuous auxiliary target has insufficient support")
    result = np.zeros(len(raw), dtype=np.int32)
    rank = pd.Series(raw[finite]).rank(method="average", pct=True).to_numpy(float)
    result[finite] = np.minimum(4, np.floor(rank * 5.0 - 1e-12).astype(np.int32)) + 1
    return result


def _first_before(
    frame: pd.DataFrame, adverse_reached: str, adverse_time: str, favourable_reached: str, favourable_time: str,
) -> np.ndarray:
    adverse = pd.to_numeric(frame[adverse_reached], errors="coerce").fillna(0).to_numpy(float) > 0.5
    favourable = pd.to_numeric(frame[favourable_reached], errors="coerce").fillna(0).to_numpy(float) > 0.5
    at = pd.to_numeric(frame[adverse_time], errors="coerce").to_numpy(float)
    ft = pd.to_numeric(frame[favourable_time], errors="coerce").to_numpy(float)
    return (adverse & (~favourable | (np.isfinite(at) & (~np.isfinite(ft) | (at < ft))))).astype(np.int32)


def _positive_quantile_grade(values: np.ndarray, *, floor: float = 50.0) -> np.ndarray:
    """Return a train-local five-quantile positive-excess label.

    There is deliberately no held-period quantile fit: the quantiles are
    recomputed inside each training fold before its booster is fitted.
    """
    raw = np.asarray(values, dtype=float)
    positive = np.isfinite(raw) & (raw > float(floor))
    if int(positive.sum()) < 500:
        raise ValueError("positive-quantile target has insufficient positive support")
    result = np.zeros(len(raw), dtype=np.int32)
    rank = pd.Series(raw[positive]).rank(method="average", pct=True).to_numpy(float)
    result[positive] = np.minimum(4, np.floor(rank * 5.0 - 1e-12).astype(np.int32)) + 1
    return result


def _clipped_utility_grade(policy_net: np.ndarray, *, floor: float, cap: float) -> np.ndarray:
    """Six ordered labels for capped positive policy utility.

    The four positive cut points are proportional to the declared cap, so the
    400-bps arm reproduces 50/100/175/275 bps after a zero-utility class.
    """
    utility = np.clip(np.maximum(np.asarray(policy_net, dtype=float) - float(floor), 0.0), 0.0, float(cap))
    result = np.zeros(len(utility), dtype=np.int32)
    positive = np.isfinite(utility) & (utility > 0.0)
    cuts = float(cap) * np.asarray((0.125, 0.25, 0.4375, 0.6875), dtype=float)
    result[positive] = 1 + np.digitize(utility[positive], cuts, right=True).astype(np.int32)
    return result


def _utility_geometry_grade(policy_net: np.ndarray, *, power: float, cap: float) -> np.ndarray:
    """Map the declared excess-over-50 p/c utility to the incumbent six grades.

    The Router specification varies exactly the positive economic geometry in
    Phase 1 while retaining the incumbent ordinal target cardinality.  Fixed
    cuts on the transformed [0, 1] utility preserve the normal six-level
    LightGBM relevance contract without changing features, label gains, query
    construction, or ranker complexity.
    """
    excess = np.maximum(np.asarray(policy_net, dtype=float) - 50.0, 0.0)
    utility = np.power(np.minimum(excess, float(cap)) / float(cap), float(power))
    result = np.zeros(len(utility), dtype=np.int32)
    positive = np.isfinite(utility) & (utility > 0.0)
    cuts = np.asarray((0.125, 0.25, 0.4375, 0.6875), dtype=float)
    result[positive] = 1 + np.digitize(utility[positive], cuts, right=True).astype(np.int32)
    return result


def _tbm_clearance(frame: pd.DataFrame, thresholds: Sequence[int]) -> np.ndarray:
    """Adverse-first versus progressive favourable ATR clearance.

    This is a seven-class ordinal target: 0 adverse first; 1 no one-ATR
    clearance; then one class for each cumulative ATR milestone.  Keeping the
    highest positive class avoids silently collapsing the requested largest
    clearance state into the preceding threshold.
    """
    adverse_first = _first_before(
        frame, "aux_reached_adverse_3atr", "aux_time_to_adverse_3atr_h",
        "aux_reached_1atr", "aux_time_to_1atr_h",
    ).astype(bool)
    result = np.ones(len(frame), dtype=np.int32)
    result[adverse_first] = 0
    for grade, threshold in enumerate(thresholds, start=2):
        reached = pd.to_numeric(frame[f"aux_reached_{threshold}atr"], errors="coerce").fillna(0).to_numpy(float) > 0.5
        if threshold == 1:
            first = ~adverse_first & reached
        else:
            t = pd.to_numeric(frame[f"aux_time_to_{threshold}atr_h"], errors="coerce").to_numpy(float)
            adverse_t = pd.to_numeric(frame["aux_time_to_adverse_3atr_h"], errors="coerce").to_numpy(float)
            first = ~adverse_first & reached & (~np.isfinite(adverse_t) | (np.isfinite(t) & (t < adverse_t)))
        result[first] = grade
    return result


def _tbm_clearance_a(frame: pd.DataFrame) -> np.ndarray:
    return _tbm_clearance(frame, (1, 2, 3, 4, 6))


def _tbm_clearance_b(frame: pd.DataFrame) -> np.ndarray:
    return _tbm_clearance(frame, (2, 3, 4, 6, 8))


def _tbm_clearance_c(frame: pd.DataFrame) -> np.ndarray:
    return _tbm_clearance(frame, (1, 2, 4, 6, 10))


def _tbm_path_quality(frame: pd.DataFrame, policy_net: np.ndarray) -> np.ndarray:
    """Simplified policy-aligned clean-path grade, derived only on train rows."""
    adverse = _first_before(
        frame, "aux_reached_adverse_3atr", "aux_time_to_adverse_3atr_h",
        "aux_reached_2atr", "aux_time_to_2atr_h",
    ).astype(bool)
    mfe = pd.to_numeric(frame["aux_peak_mfe_atr_h12"], errors="coerce").to_numpy(float)
    mae = np.maximum(pd.to_numeric(frame["aux_mae_atr_12h"], errors="coerce").to_numpy(float), 0.0)
    result = np.ones(len(frame), dtype=np.int32)  # timeout/noisy unresolved
    result[adverse] = 0
    active = ~adverse & np.isfinite(mfe)
    # Each later grade must retain the preceding positive-path condition;
    # assignments are deliberately cumulative from weakest to strongest.
    result[active & (mfe >= 2.0)] = 2
    result[active & (mfe >= 3.0) & (mae <= 3.0)] = 3
    result[active & (mfe >= 4.0) & (mae <= 2.0)] = 4
    result[active & (mfe >= 6.0) & (mae <= 1.0)] = 5
    # A clean positive policy conversion should never be downgraded merely
    # because the 15-minute path proxy rounded a boundary equality.
    result[(~adverse) & np.isfinite(policy_net) & (policy_net > 200.0)] = np.maximum(
        result[(~adverse) & np.isfinite(policy_net) & (policy_net > 200.0)], 4,
    )
    return result


def _primary_target(frame: pd.DataFrame, primary: str) -> np.ndarray:
    """Construct one declared router main label from strict-resolved rows."""
    if primary not in ALL_PRIMARY_TARGETS:
        raise ValueError(f"unknown primary target {primary}; expected one of {list(ALL_PRIMARY_TARGETS)}")
    policy_net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    if primary in PRIMARY_TARGETS:
        return np.digitize(policy_net, PRIMARY_TARGETS[primary]).astype(np.int32)
    if primary == "P8_path_policy":
        severe = _first_before(
            frame, "aux_reached_adverse_3atr", "aux_time_to_adverse_3atr_h",
            "aux_reached_2atr", "aux_time_to_2atr_h",
        ).astype(bool) | (policy_net <= -200.0)
        result = 1 + np.digitize(policy_net, (0.0, 50.0, 100.0, 200.0)).astype(np.int32)
        result[severe] = 0
        return result
    if primary.startswith("P8u_floor"):
        floor_token, cap_token = primary.split("_")[-2:]
        return _clipped_utility_grade(policy_net, floor=float(floor_token.removeprefix("floor")), cap=float(cap_token.removeprefix("cap")))
    if primary in {"R50_policy_net_gt50", "R100_policy_net_gt100", "R200_policy_net_gt200"}:
        hurdle = float(primary.split("_gt", 1)[1])
        # Strictly greater matches the post-score Recall@{50,100,200}
        # evaluation denominators.  Invalid labels have already been excluded
        # in _prepare_train and can never become implicit zero examples here.
        return (policy_net > hurdle).astype(np.int32)
    if primary.startswith("U50_p"):
        _, power_token, cap_token = primary.split("_")
        return _utility_geometry_grade(
            policy_net,
            power=float(power_token.removeprefix("p")) / 100.0,
            cap=float(cap_token.removeprefix("c")),
        )
    if primary == "P9_positive_quantiles":
        return _positive_quantile_grade(policy_net)
    if primary == "T1_tbm_clearance_a_1_2_3_4_6":
        return _tbm_clearance_a(frame)
    if primary == "T1_tbm_clearance_b_2_3_4_6_8":
        return _tbm_clearance_b(frame)
    if primary == "T1_tbm_clearance_c_1_2_4_6_10":
        return _tbm_clearance_c(frame)
    if primary == "T2_tbm_path_quality":
        return _tbm_path_quality(frame, policy_net)
    raise AssertionError(primary)


def _primary_weight_utility(
    frame: pd.DataFrame, primary: str, relevance: np.ndarray,
) -> np.ndarray:
    """Return the declared main-target utility in ``[0, 1]`` for W2/W4.

    This loss-only quantity is calculated from the same resolved policy
    outcome that forms the main label.  It is never persisted in a held score
    receipt or exposed at inference.  Utility-labelled targets retain their
    exact predeclared p/c geometry; legacy ordinal labels fall back to their
    fitted ordered relevance rather than acquiring a new hidden transform.
    """
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    if primary.startswith("U50_p"):
        _, power_token, cap_token = primary.split("_")
        power = float(power_token.removeprefix("p")) / 100.0
        cap = float(cap_token.removeprefix("c"))
        value = np.power(np.minimum(np.maximum(net - 50.0, 0.0), cap) / cap, power)
    elif primary.startswith("P8u_floor"):
        floor_token, cap_token = primary.split("_")[-2:]
        floor = float(floor_token.removeprefix("floor"))
        cap = float(cap_token.removeprefix("cap"))
        value = np.minimum(np.maximum(net - floor, 0.0), cap) / cap
    else:
        value = np.asarray(relevance, dtype=float)
        ceiling = float(np.nanmax(value)) if np.isfinite(value).any() else 0.0
        value = value / ceiling if ceiling > 0.0 else np.zeros(len(value), dtype=float)
    return np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0).astype(np.float32)


def _targets(
    frame: pd.DataFrame, primary: str, *, include_auxiliary: bool = True,
) -> dict[str, tuple[object, ...]]:
    """Build only the label families required by this fit.

    The direct policy-net primary labels are self-contained.  A primary-only
    run must not materialise A--E path targets as a side effect, otherwise an
    unrelated auxiliary schema can silently become part of a supposed
    policy-only experiment.  Path-defined primary targets still require their
    declared auxiliary fields and fail clearly if requested without them.
    """
    policy_net = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float)
    if not include_auxiliary:
        if primary in {"P8_path_policy", "T1_tbm_clearance_a_1_2_3_4_6", "T1_tbm_clearance_b_2_3_4_6", "T1_tbm_clearance_c_1_2_4_6", "T2_tbm_path_quality"}:
            raise ValueError(f"{primary} requires auxiliary path labels; primary-only policy mode is invalid")
        return {"main": (primary, False, _primary_target(frame, primary))}
    policy_gross = pd.to_numeric(frame["policy_gross_bps"], errors="coerce").to_numpy(float)
    result: dict[str, tuple[object, ...]] = {
        "main": (primary, False, _primary_target(frame, primary)),
        # A: favourable milestone reached before the policy's meaningful stop.
        "A_reach_2atr": ("A", False, pd.to_numeric(frame["aux_reached_2atr"], errors="coerce").fillna(0).astype(np.int32).to_numpy()),
        "A_reach_3atr": ("A", False, pd.to_numeric(frame["aux_reached_3atr"], errors="coerce").fillna(0).astype(np.int32).to_numpy()),
        "A_reach_4atr": ("A", False, pd.to_numeric(frame["aux_reached_4atr"], errors="coerce").fillna(0).astype(np.int32).to_numpy()),
        "A_reach_6atr": ("A", False, pd.to_numeric(frame["aux_reached_6atr"], errors="coerce").fillna(0).astype(np.int32).to_numpy()),
        # B: exactly five train-local speed quintiles among paths that reached.
        "B_time_2atr": ("B", False, _speed_grade(frame["aux_time_to_2atr_h"])),
        "B_time_3atr": ("B", False, _speed_grade(frame["aux_time_to_3atr_h"])),
        "B_time_50bps": ("B", False, _speed_grade(frame["aux_time_to_50bps_h"])),
        "B_time_100bps": ("B", False, _speed_grade(frame["aux_time_to_100bps_h"])),
        # C: bounded economics.  ATR MFE is clipped at the largest declared
        # reach level; bps quantities use the requested 400-bps gross ceiling.
        "C_clipped_mfe_atr": ("C", False, _quintile_grade(np.clip(pd.to_numeric(frame["aux_peak_mfe_atr_h12"], errors="coerce"), 0.0, 6.0))),
        "C_clipped_mfe_bps": ("C", False, _quintile_grade(np.clip(pd.to_numeric(frame["aux_peak_mfe_bps_h12"], errors="coerce"), 0.0, 400.0))),
        "C_positive_policy_potential": ("C", False, _quintile_grade(np.clip(np.maximum(policy_gross, 0.0), 0.0, 400.0))),
        # D: compact path efficiency.  Bound the long tail prior to train-only
        # quintiles; this avoids a few near-zero ATR denominators dominating.
        "D_mfe_over_path_length": ("D", False, _quintile_grade(np.clip(pd.to_numeric(frame["aux_mfe_over_path_length_h12"], errors="coerce"), 0.0, 10.0))),
        "D_mfe_over_abs_mae": ("D", False, _quintile_grade(np.clip(pd.to_numeric(frame["aux_mfe_over_abs_mae_h12"], errors="coerce"), 0.0, 20.0))),
    }
    result["E_adverse3_before_2"] = (
        "E", True, _first_before(frame, "aux_reached_adverse_3atr", "aux_time_to_adverse_3atr_h", "aux_reached_2atr", "aux_time_to_2atr_h"),
    )
    result["E_adverse4_before_3"] = (
        "E", True, _first_before(frame, "aux_reached_adverse_4atr", "aux_time_to_adverse_4atr_h", "aux_reached_3atr", "aux_time_to_3atr_h"),
    )
    severe = pd.to_numeric(frame["policy_net_bps"], errors="coerce").to_numpy(float) <= -200.0
    hit = pd.to_numeric(frame["aux_first_policy_hit_interval_h"], errors="coerce").to_numpy(float)
    mfe2 = pd.to_numeric(frame["aux_time_to_2atr_h"], errors="coerce").to_numpy(float)
    result["E_severe_loss_before_meaningful_mfe"] = (
        "E", True, (severe & (~np.isfinite(mfe2) | (np.isfinite(hit) & (hit < mfe2)))).astype(np.int32),
    )
    return result


def _ranker_spec(args: argparse.Namespace) -> dict[str, object]:
    gains = (
        tuple(float(value.strip()) for value in args.label_gains.split(",") if value.strip())
        if args.label_gains else None
    )
    return {
        "objective": args.objective,
        "n_estimators": args.n_estimators, "learning_rate": args.learning_rate,
        "max_depth": args.max_depth, "num_leaves": args.num_leaves,
        "min_child_fraction": args.min_child_fraction, "min_child_floor": args.min_child_floor,
        "min_split_gain": args.min_split_gain, "subsample": args.subsample,
        "feature_fraction": args.feature_fraction, "l1": args.l1, "l2": args.l2,
        "max_bin": args.max_bin, "truncation": args.truncation,
        # Keep the contract JSON-stable.  ``json.loads`` returns a list while
        # the parser naturally builds a tuple; preserving the tuple here made
        # an otherwise identical ``--resume`` invocation fail solely because
        # of its container type.
        "label_gains": list(gains) if gains is not None else None,
        "row_weight_scheme": args.row_weight_scheme,
        "row_weight_floor_bps": args.row_weight_floor_bps,
        "row_weight_cap_bps": args.row_weight_cap_bps,
        "early_stopping_rounds": args.early_stopping_rounds,
        "inner_validation_fraction": args.inner_validation_fraction,
        "n_jobs": args.n_jobs, "deterministic": True,
    }


def _fit_target(
    train: pd.DataFrame, fields: tuple[str, ...], name: str, group: str, inverse: bool,
    target: np.ndarray, seed: int, ranker: dict[str, object],
) -> FittedTarget:
    work = train.copy()
    work["__target__"] = np.asarray(target, dtype=np.int32)
    if work["__target__"].nunique() < 2:
        raise ValueError(f"{name}: no target class variation")
    primary_utility = _primary_weight_utility(
        work, name, work["__target__"].to_numpy(np.int32),
    )
    order, groups, weights, weight_summary = _query_weights(
        work,
        scheme=str(ranker["row_weight_scheme"]),
        floor_bps=float(ranker["row_weight_floor_bps"]),
        cap_bps=float(ranker["row_weight_cap_bps"]),
        primary_utility=primary_utility,
    )
    work = work.iloc[order["__row__"].to_numpy()].reset_index(drop=True)
    target_ordered = work["__target__"].to_numpy(np.int32)
    if len(work) < 5_000 or len(groups) < 500 or np.unique(target_ordered).size < 2:
        raise ValueError(f"{name}: insufficient query/class support after filtering")
    medians = _medians(work, fields)
    classes = int(np.max(target_ordered)) + 1
    gains = ranker["label_gains"]
    if gains is not None and len(gains) != classes:
        raise ValueError(f"{name}: label gains must have exactly {classes} entries")
    params = dict(
        objective=str(ranker["objective"]), metric="ndcg", label_gain=list(range(classes)) if gains is None else list(gains),
        lambdarank_truncation_level=int(ranker["truncation"]),
        n_estimators=int(ranker["n_estimators"]), learning_rate=float(ranker["learning_rate"]),
        max_depth=int(ranker["max_depth"]), num_leaves=int(ranker["num_leaves"]),
        min_child_samples=max(int(ranker["min_child_floor"]), int(float(ranker["min_child_fraction"]) * len(work))),
        min_split_gain=float(ranker["min_split_gain"]), subsample=float(ranker["subsample"]),
        subsample_freq=1, colsample_bytree=float(ranker["feature_fraction"]),
        reg_alpha=float(ranker["l1"]), reg_lambda=float(ranker["l2"]), max_bin=int(ranker["max_bin"]),
        random_state=seed, n_jobs=int(ranker["n_jobs"]), deterministic=True, force_col_wise=True, verbosity=-1,
    )
    matrix = _matrix(work, fields, medians)
    early_rounds = int(ranker.get("early_stopping_rounds", 0))
    validation_fraction = float(ranker.get("inner_validation_fraction", 0.0))
    if early_rounds > 0:
        # Validation is strictly the latest complete timestamp queries within
        # the training ledger.  It is used solely to choose tree count; the
        # final booster is then refit on all eligible training rows with that
        # fixed count.  No held-month row or label participates here.
        if not 0.05 <= validation_fraction < 0.5:
            raise ValueError("inner validation fraction must lie in [0.05, 0.5) when early stopping is enabled")
        query_cut = int(math.floor(len(groups) * (1.0 - validation_fraction)))
        query_cut = min(max(query_cut, 400), len(groups) - 100)
        split = int(np.sum(groups[:query_cut]))
        probe = LGBMRanker(**params).fit(
            matrix[:split], target_ordered[:split], group=groups[:query_cut], sample_weight=weights[:split],
            eval_set=[(matrix[split:], target_ordered[split:])], eval_group=[groups[query_cut:]],
            callbacks=[early_stopping(stopping_rounds=early_rounds, verbose=False)],
        )
        selected_trees = int(probe.best_iteration_ or params["n_estimators"])
        params["n_estimators"] = selected_trees
    model = LGBMRanker(**params).fit(matrix, target_ordered, group=groups, sample_weight=weights)
    raw = model.predict(_matrix(work, fields, medians))
    return FittedTarget(
        name, group, inverse, fields, medians, model,
        ScoreReference.fit(raw, f"{name}_strict_train"), classes, len(work),
        len(groups), weight_summary,
    )


def _route_rank(frame: pd.DataFrame, field: str, fraction: float) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", field]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__score__"] = pd.to_numeric(work[field], errors="coerce").fillna(-np.inf)
    work = work.sort_values(["__decision_ts__", "__score__", "candidate_id"], ascending=[True, False, True], kind="stable")
    work["__ordinal__"] = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    size = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size").to_numpy(float)
    selected = (work["__ordinal__"].to_numpy() <= np.ceil(float(fraction) * size)) & np.isfinite(work["__score__"].to_numpy(float))
    return pd.Series(selected, index=work["__row__"].to_numpy()).reindex(np.arange(len(frame))).to_numpy(bool)


def _score(
    train: pd.DataFrame, held: pd.DataFrame, fields: tuple[str, ...], primary: str,
    groups: Sequence[str], cap: int, ranker: dict[str, object], reused_aux: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int, list[dict[str, object]]]:
    # _score_month has already applied the sole deterministic whole-query cap
    # to the target-free candidate panel *before* any policy/path label is
    # joined. Reapplying it here after label filtering changes the retained
    # query set a second time and can leave an otherwise valid capped ledger
    # below the model's support floor. Keep the first cap as the causal
    # contract and assert that callers do not bypass it.
    if len(train) > cap:
        raise AssertionError("_score received an uncapped training ledger")
    target_defs = _targets(train, primary, include_auxiliary=bool(groups))
    main_name, main_inverse, main_target = target_defs["main"]
    wanted = [(str(main_name), "main", bool(main_inverse), np.asarray(main_target))]
    for name, (group, inverse, target) in target_defs.items():
        if reused_aux is None and name != "main" and group in groups:
            wanted.append((name, group, inverse, target))
    primary_rows = 0
    audit: list[dict[str, object]] = []
    scored = held.loc[:, ["candidate_id", "__decision_ts__", "side_name"]].copy()
    if reused_aux is not None:
        scored = scored.merge(
            reused_aux, on=["candidate_id", "__decision_ts__", "side_name"],
            how="left", validate="one_to_one",
        )
        if scored.filter(regex=r"^router_group_").isna().any().any():
            raise AssertionError("reused auxiliary ranks contain an unexpected null")
        audit.append({"name": "reused_A_E_auxiliary_ranks", "group": "A-E", "reused": True})
    for number, (name, group, inverse, target) in enumerate(wanted):
        model = _fit_target(
            train, fields, name, group, bool(inverse), np.asarray(target),
            SEED + 1009 + number, ranker,
        )
        _, rank = model.score(held)
        column = "router_primary_rank" if group == "main" else f"router__{name}__rank"
        scored[column] = rank
        if group == "main":
            primary_rows = model.rows
        audit.append({
            "name": name, "group": group, "inverse": bool(inverse),
            "classes": model.classes, "rows": model.rows, "queries": model.queries,
            "rank_reference": model.reference.source,
            "weight_summary": model.weight_summary,
        })
        # The held score and train-derived CDF are now materialised.  Retaining
        # sixteen fitted boosters would needlessly multiply peak memory for a
        # multi-fold experiment, and the model objects are not an artifact of
        # this target-free score producer.
        del model
        gc.collect()
    for group in groups:
        if f"router_group_{group}_rank" in scored:
            continue
        columns = [column for column in scored if column.startswith(f"router__{group}_") and column.endswith("__rank")]
        if not columns:
            raise AssertionError(f"no scored targets for requested group {group}")
        scored[f"router_group_{group}_rank"] = scored[columns].mean(axis=1).astype(np.float32)
    # Preserve a primary-only control in the exact same score receipt.  The
    # full contract gives the main economic label 50% authority and shares the
    # remaining 50% equally across retained A--E groups.  A frozen winner can
    # also request a primary-only ledger for downstream refits, after the
    # complete suite has failed its predeclared group gate.  This avoids
    # retraining outcome-path auxiliaries which have no selected authority.
    scored["router_primary_only_rank"] = scored["router_primary_rank"].astype(np.float32)
    full = .5 * scored["router_primary_rank"].to_numpy(float)
    if groups:
        aux_weight = .5 / len(groups)
        for group in groups:
            full += aux_weight * scored[f"router_group_{group}_rank"].to_numpy(float)
    else:
        full = scored["router_primary_rank"].to_numpy(float)
    scored["router_full_ae_rank"] = full.astype(np.float32)
    return scored, primary_rows, audit


def _timestamp_metric_rows(
    scores: pd.DataFrame, policy: pd.DataFrame, held_month: pd.Timestamp, route_fractions: Sequence[float],
) -> pd.DataFrame:
    """Evaluate every route equally by decision timestamp, never globally.

    The route itself is always selected on the complete target-free candidate
    population.  Invalid policy paths are excluded only *after* that decision
    when measuring realised economics; their future label completeness can
    therefore neither admit another candidate nor alter an earlier rank.
    """
    joined = scores.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    joined["__policy_net__"] = pd.to_numeric(joined["policy_net_bps"], errors="coerce")
    joined["__valid__"] = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(joined["__policy_net__"])
    output: list[pd.DataFrame] = []
    score_fields = ["router_primary_only_rank", "router_full_ae_rank"]
    for score in score_fields:
        for fraction in route_fractions:
            work = joined.loc[:, ["__decision_ts__", "candidate_id", score, "__valid__", "__policy_net__"]].copy()
            work["__selected__"] = _route_rank(work, score, fraction)
            work["__selected_valid__"] = work["__selected__"] & work["__valid__"]
            work["__selected_net__"] = np.where(work["__selected_valid__"], work["__policy_net__"], 0.0)
            work["__excess__"] = np.where(
                work["__valid__"],
                np.maximum(work["__policy_net__"].to_numpy(float) - PRIMARY_UTILITY_FLOOR_BPS, 0.0),
                0.0,
            )
            work["__selected_excess__"] = np.where(work["__selected_valid__"], work["__excess__"], 0.0)
            work["__utility__"] = np.minimum(work["__excess__"], PRIMARY_UTILITY_CAP_BPS) / PRIMARY_UTILITY_CAP_BPS
            work["__utility__"] = np.power(work["__utility__"], PRIMARY_UTILITY_POWER)
            work["__selected_utility__"] = np.where(work["__selected_valid__"], work["__utility__"], 0.0)
            work["__winner50__"] = work["__valid__"] & work["__policy_net__"].gt(PRIMARY_UTILITY_FLOOR_BPS)
            work["__winner100__"] = work["__valid__"] & work["__policy_net__"].gt(100.0)
            work["__winner200__"] = work["__valid__"] & work["__policy_net__"].gt(200.0)
            work["__selected_winner50__"] = work["__selected__"] & work["__winner50__"]
            work["__selected_winner100__"] = work["__selected__"] & work["__winner100__"]
            work["__selected_winner200__"] = work["__selected__"] & work["__winner200__"]
            grouped = work.groupby("__decision_ts__", sort=False).agg(
                candidate_rows=("candidate_id", "size"), selected_candidate_rows=("__selected__", "sum"),
                valid_rows=("__valid__", "sum"), selected_valid_rows=("__selected_valid__", "sum"),
                selected_net_sum_bps=("__selected_net__", "sum"),
                excess_sum=("__excess__", "sum"), selected_excess_sum=("__selected_excess__", "sum"),
                utility_sum=("__utility__", "sum"), selected_utility_sum=("__selected_utility__", "sum"),
                winners50=("__winner50__", "sum"), selected_winners50=("__selected_winner50__", "sum"),
                winners100=("__winner100__", "sum"), selected_winners100=("__selected_winner100__", "sum"),
                winners200=("__winner200__", "sum"), selected_winners200=("__selected_winner200__", "sum"),
            ).reset_index()
            group_count = grouped["selected_valid_rows"].to_numpy(float)
            grouped["timestamp_selected_net_ev_bps"] = np.divide(
                grouped["selected_net_sum_bps"], group_count,
                out=np.full(len(grouped), np.nan), where=group_count > 0,
            )
            denom = grouped["excess_sum"].to_numpy(float)
            grouped["timestamp_er50"] = np.divide(
                grouped["selected_excess_sum"], denom,
                out=np.full(len(grouped), np.nan), where=denom > 0,
            )
            utility = grouped["utility_sum"].to_numpy(float)
            grouped["timestamp_utility_recall"] = np.divide(
                grouped["selected_utility_sum"], utility,
                out=np.full(len(grouped), np.nan), where=utility > 0,
            )
            winners50 = grouped["winners50"].to_numpy(float)
            winners100 = grouped["winners100"].to_numpy(float)
            winners200 = grouped["winners200"].to_numpy(float)
            grouped["timestamp_recall_50bps"] = np.divide(grouped["selected_winners50"], winners50, out=np.full(len(grouped), np.nan), where=winners50 > 0)
            grouped["timestamp_recall_100bps"] = np.divide(grouped["selected_winners100"], winners100, out=np.full(len(grouped), np.nan), where=winners100 > 0)
            grouped["timestamp_recall_200bps"] = np.divide(grouped["selected_winners200"], winners200, out=np.full(len(grouped), np.nan), where=winners200 > 0)
            grouped["held_month"] = held_month.strftime("%Y-%m")
            grouped["score"] = score
            grouped["route_fraction"] = fraction
            output.append(grouped)
    return pd.concat(output, ignore_index=True)


def _metric_rows(
    scores: pd.DataFrame, policy: pd.DataFrame, held_month: pd.Timestamp, route_fractions: Sequence[float],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    timestamp = _timestamp_metric_rows(scores, policy, held_month, route_fractions)
    rows: list[dict[str, object]] = []
    for (score, fraction), work in timestamp.groupby(["score", "route_fraction"], sort=False):
        selected = float(work["selected_valid_rows"].sum())
        net_sum = float(work["selected_net_sum_bps"].sum())
        utility_eligible = work.loc[work["timestamp_utility_recall"].notna(), "utility_sum"].to_numpy(float)
        median_utility = float(np.median(utility_eligible[utility_eligible > 0])) if np.any(utility_eligible > 0) else np.nan
        weights_utility = np.zeros(len(work), dtype=float)
        if np.isfinite(median_utility) and median_utility > 0:
            weights_utility = np.minimum(
                np.power(np.maximum(work["utility_sum"].to_numpy(float), 0.0) / median_utility, PRIMARY_UTILITY_TIMESTAMP_GAMMA),
                PRIMARY_UTILITY_TIMESTAMP_CAP,
            )
        valid_utility = work["timestamp_utility_recall"].notna().to_numpy()
        utility_recall = float(np.average(work.loc[valid_utility, "timestamp_utility_recall"], weights=weights_utility[valid_utility])) if np.any(valid_utility) and weights_utility[valid_utility].sum() else np.nan
        recall50 = float(work["timestamp_recall_50bps"].mean())
        recall100 = float(work["timestamp_recall_100bps"].mean())
        recall200 = float(work["timestamp_recall_200bps"].mean())
        rows.append({
            "held_month": held_month.strftime("%Y-%m"), "score": score, "route_fraction": float(fraction),
            "timestamps": int(len(work)), "er50_timestamps": int(work["timestamp_er50"].notna().sum()),
            "recall50_timestamps": int(work["timestamp_recall_50bps"].notna().sum()),
            "recall100_timestamps": int(work["timestamp_recall_100bps"].notna().sum()),
            "recall200_timestamps": int(work["timestamp_recall_200bps"].notna().sum()),
            "utility_timestamps": int(work["timestamp_utility_recall"].notna().sum()),
            "ev_timestamps": int(work["timestamp_selected_net_ev_bps"].notna().sum()),
            "selected_rows": int(selected), "net_sum_bps": net_sum,
            "net_ev_bps_per_trade": float(work["timestamp_selected_net_ev_bps"].mean()),
            "trade_weighted_net_ev_bps_per_trade": net_sum / selected if selected else np.nan,
            "er50": float(work["timestamp_er50"].mean()),
            "utility_recall": utility_recall,
            "winner_recall_50bps": recall50,
            "winner_recall_100bps": recall100,
            "winner_recall_200bps": recall200,
            "s_router": float(.70 * utility_recall + .15 * recall50 + .15 * recall100) if np.isfinite(utility_recall) else np.nan,
        })
    return rows, timestamp


def _contract(args: argparse.Namespace, fields: Sequence[str], months: Sequence[pd.Timestamp]) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope": "research-only strict-OOF economic-recall router; no live, MC1, consensus, or execution mutation",
        "feature_roots": [str(path) for path in _normalise_feature_roots(args.feature_root)],
        "aux_root": str(args.aux_root), "policy_path": str(args.policy_path),
        "bundle": str(args.bundle), "bundle_sha256": _sha256_file(args.bundle),
        "sealed_feature_contract_source": str(args.sealed_feature_contract) if args.sealed_feature_contract else None,
        "sealed_feature_contract_source_sha256": _sha256_file(args.sealed_feature_contract) if args.sealed_feature_contract else None,
        "full_feature_contract_source": str(args.full_feature_contract) if args.full_feature_contract else None,
        "full_feature_contract_source_sha256": _sha256_file(args.full_feature_contract) if args.full_feature_contract else None,
        "feature_contract": list(fields), "feature_contract_sha256": _feature_hash(fields),
        "sealed_base_feature_contract_sha256": FEATURE_HASH if args.full_feature_contract is None else None,
        "feature_selection_source": str(args.feature_list) if args.feature_list else None,
        "primary_target": args.primary_target, "aux_groups": list(args.aux_groups),
        "score_contract": "primary_only" if args.primary_only else "primary_plus_A_E_diagnostic",
        "reused_aux_source": str(args.reuse_aux_source) if args.reuse_aux_source else None,
        "reused_aux_contract_sha256": (
            _sha256_file(args.reuse_aux_source / "run_contract.json") if args.reuse_aux_source else None
        ),
        "months": [month.strftime("%Y-%m") for month in months], "train_months": args.train_months,
        "reserve_days": args.reserve_days, "train_cap": args.train_cap, "n_jobs": args.n_jobs,
        "route_fractions": list(args.route_fractions),
        "ranker": _ranker_spec(args),
        "causality": {
            "inputs": (
                "only a frozen causal subset of the sealed 120 base fields"
                if args.full_feature_contract is None else
                "only the separately frozen causal full-universe feature contract"
            ),
            "labels": "policy and auxiliary path labels used only while fitting",
            "label_availability": "both policy and auxiliary label_available_ts must precede the reserve start",
            "embargo": f"{MAX_LABEL_HORIZON_HOURS} hours implicit in resolved-label cutoff plus {args.reserve_days}-day reserve",
            "held_receipt": "target-free score parquet is persisted before policy outcomes are joined for metrics",
            "weights": "each exact timestamp query has total loss weight one",
        },
    }


def _prepare_train(
    features: pd.DataFrame, aux: pd.DataFrame | None, policy: pd.DataFrame, reserve_start: pd.Timestamp,
) -> pd.DataFrame:
    """Join only the labels required by the selected score contract.

    The primary-only router predicts a policy-derived economic grade and has
    no semantic need for A--E path labels.  Reading those labels anyway made a
    primary-only HPO depend on an unrelated, older auxiliary schema.  More
    importantly, that unnecessary dependency could cause an otherwise valid
    causal primary experiment to fail before fitting.  Full A--E experiments
    retain the exact former join and availability requirements.
    """
    result = features.copy()
    if aux is not None:
        result = result.merge(aux, on=["candidate_id", "__decision_ts__", "side_name"], how="left", validate="one_to_one")
    result = result.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    valid = (
        result["policy_path_valid"].fillna(False).astype(bool)
        & result["policy_label_available_ts"].lt(reserve_start)
        & np.isfinite(pd.to_numeric(result["policy_net_bps"], errors="coerce"))
        & np.isfinite(pd.to_numeric(result["policy_gross_bps"], errors="coerce"))
    )
    if aux is not None:
        valid &= result["aux_path_valid"].fillna(False).astype(bool) & result["aux_label_available_ts"].lt(reserve_start)
    result = result.loc[valid].copy()
    if result.empty:
        raise AssertionError("no strict-resolved path/policy labels in training window")
    return result


def _score_month(
    *, feature_root: Path | Sequence[Path], aux_root: Path, policy: pd.DataFrame, fields: tuple[str, ...], month: pd.Timestamp,
    train_months: int, reserve_days: int, primary_target: str, aux_groups: Sequence[str], train_cap: int,
    ranker: dict[str, object], reuse_aux_source: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    held_end = _month_end(month)
    reserve_start = month - pd.Timedelta(days=reserve_days)
    train_start = reserve_start - pd.DateOffset(months=train_months)
    train_features = _window_features(feature_root, train_start, reserve_start, ("candidate_id", "__decision_ts__", "side_name", *fields))
    held_features = _window_features(feature_root, month, held_end, ("candidate_id", "__decision_ts__", "side_name", *fields))
    # A primary-only score needs only resolved policy outcomes.  Avoid both
    # reading and joining an unrelated auxiliary path label schema; the A--E
    # branch below deliberately preserves the stricter old requirements.
    aux = _window_aux(aux_root, train_start, reserve_start) if aux_groups else None
    # Cap the target-free candidate *queries before* bringing path and policy
    # labels into memory.  The hash is based only on decision timestamp, so
    # this is both causal and target-independent while avoiding a several-GB
    # merged six-month frame merely to discard most of it later.
    train_features = _deterministic_query_cap(train_features, cap=train_cap)
    selected_ids = set(train_features["candidate_id"].astype(str))
    if aux is not None:
        aux = aux.loc[aux["candidate_id"].astype(str).isin(selected_ids)].copy()
    train = _prepare_train(train_features, aux, policy, reserve_start)
    if len(train) < 20_000 or len(held_features) < 5_000:
        raise AssertionError(f"{month:%Y-%m}: inadequate strict support train={len(train)} held={len(held_features)}")
    reused_aux = None
    reused_aux_sha256 = None
    if reuse_aux_source is not None:
        reused_aux, reused_aux_sha256 = _reused_aux_scores(reuse_aux_source, month, held_features, aux_groups)
    scores, primary_rows, heads = _score(
        train, held_features, fields, primary_target, aux_groups, train_cap, ranker, reused_aux,
    )
    # Head names may legitimately describe their *training target* (for
    # example ``router__C_clipped_mfe_bps__rank``).  Token matching would
    # mistake that name for an outcome field.  Instead whitelist the held
    # target-free schema: identities plus model outputs are allowed; copied
    # labels/paths cannot satisfy this condition.
    invalid_score_columns = [
        column for column in scores.columns
        if column not in {"candidate_id", "__decision_ts__", "side_name"}
        and not (column.startswith("router_") or column.startswith("router__"))
    ]
    if invalid_score_columns:
        raise AssertionError(f"held target-free score panel has an undeclared column: {invalid_score_columns}")
    audit = {
        "held_month": month.strftime("%Y-%m"), "train_start": str(train_start), "reserve_start": str(reserve_start),
        "held_end": str(held_end), "train_rows_before_cap": int(len(train)), "train_rows": int(primary_rows),
        "held_rows": int(len(held_features)), "heads": heads, "primary_target": primary_target,
        "aux_groups": list(aux_groups), "label_valid_fraction": float(len(train) / max(len(train_features), 1)),
        "ranker": ranker, "reused_aux_source": str(reuse_aux_source) if reuse_aux_source else None,
        "reused_aux_score_sha256": reused_aux_sha256,
    }
    return scores, audit


def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (score, fraction), work in metrics.groupby(["score", "route_fraction"], sort=False):
        weights_er = work["er50_timestamps"].to_numpy(float)
        weights_utility = work["utility_timestamps"].to_numpy(float)
        weights_ev = work["ev_timestamps"].to_numpy(float)
        weights_r50 = work["recall50_timestamps"].to_numpy(float)
        weights_r100 = work["recall100_timestamps"].to_numpy(float)
        rows.append({
            "score": score, "route_fraction": float(fraction), "folds": int(work["held_month"].nunique()),
            "mean_er50": float(np.average(work["er50"], weights=weights_er)) if weights_er.sum() else np.nan,
            "min_er50": float(work["er50"].min()),
            "mean_utility_recall": float(np.average(work["utility_recall"], weights=weights_utility)) if weights_utility.sum() else np.nan,
            "min_utility_recall": float(work["utility_recall"].min()),
            "mean_recall50": float(np.average(work["winner_recall_50bps"], weights=weights_r50)) if weights_r50.sum() else np.nan,
            "min_recall50": float(work["winner_recall_50bps"].min()),
            "mean_recall100": float(np.average(work["winner_recall_100bps"], weights=weights_r100)) if weights_r100.sum() else np.nan,
            "min_recall100": float(work["winner_recall_100bps"].min()),
            "mean_s_router": float(np.average(work["s_router"], weights=weights_utility)) if weights_utility.sum() else np.nan,
            "min_s_router": float(work["s_router"].min()),
            "mean_net_ev_bps_per_trade": float(np.average(work["net_ev_bps_per_trade"], weights=weights_ev)) if weights_ev.sum() else np.nan,
            "min_net_ev_bps_per_trade": float(work["net_ev_bps_per_trade"].min()),
            "selected_rows": int(work["selected_rows"].sum()), "net_sum_bps": float(work["net_sum_bps"].sum()),
            "timestamps": int(work["timestamps"].sum()),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    months = tuple(_utc(value) for value in args.months)
    if tuple(sorted(months)) != months or len(set(months)) != len(months):
        raise ValueError("months must be unique chronological UTC month starts")
    if args.primary_only:
        if args.reuse_aux_source is not None:
            raise ValueError("primary-only downstream ledgers must not load unused auxiliary score artifacts")
        args.aux_groups = ()
    elif not set(args.aux_groups).issubset(ALL_GROUPS) or not args.aux_groups:
        raise ValueError(f"aux groups must be a non-empty subset of {ALL_GROUPS}")
    fields = _selected_fields(
        args.bundle, args.feature_list, args.sealed_feature_contract,
        args.full_feature_contract,
    )
    ranker = _ranker_spec(args)
    if args.reuse_aux_source is not None:
        _validate_aux_reuse_source(args.reuse_aux_source, fields, months, args)
    contract = _contract(args, fields, months)
    if args.out.exists():
        if not args.resume:
            raise FileExistsError(args.out)
        existing = args.out / "run_contract.json"
        if not existing.exists() or json.loads(existing.read_text()) != contract:
            raise AssertionError("refusing a resume whose immutable router contract differs")
    else:
        args.out.mkdir(parents=True)
        _write_json_exclusive(args.out / "run_contract.json", contract)
    policy = _policy_window(args.policy_path, months[0] - pd.DateOffset(months=args.train_months + 2), _month_end(months[-1]))
    score_root = args.out / "target_free_scores"
    score_root.mkdir(exist_ok=True)
    audit_root = args.out / "audit_parts"
    audit_root.mkdir(exist_ok=True)
    done = 0
    for month in months:
        score_path = score_root / f"month={month:%Y-%m}.parquet"
        audit_path = audit_root / f"month={month:%Y-%m}.json"
        if score_path.exists() and audit_path.exists():
            continue
        if score_path.exists() != audit_path.exists():
            raise AssertionError(f"partial immutable receipt exists for {month:%Y-%m}; use a new output root")
        if args.max_jobs is not None and done >= args.max_jobs:
            break
        scores, audit = _score_month(
            feature_root=args.feature_root, aux_root=args.aux_root, policy=policy, fields=fields, month=month,
            train_months=args.train_months, reserve_days=args.reserve_days, primary_target=args.primary_target,
            aux_groups=args.aux_groups, train_cap=args.train_cap, ranker=ranker,
            reuse_aux_source=args.reuse_aux_source,
        )
        scores.to_parquet(score_path, index=False, compression="zstd")
        _write_json_exclusive(audit_path, audit)
        done += 1
        print(json.dumps({"event": "scored", **audit}), flush=True)
    expected = [(score_root / f"month={month:%Y-%m}.parquet", audit_root / f"month={month:%Y-%m}.json") for month in months]
    if not all(score.exists() and audit.exists() for score, audit in expected):
        return
    # The common Router utility evaluator is deliberately streaming and is
    # the authoritative target-selection surface.  On a constrained worker,
    # constructing the legacy all-route metrics in this runner needlessly
    # retains every held score/label timestamp frame and can exceed memory
    # *after* all immutable target-free scores have already been sealed.
    # Deferral preserves the identical score receipt and lets the evaluator
    # perform the required route metrics independently without re-fitting or
    # touching a policy label during scoring.
    if args.defer_aggregate:
        return
    metrics_path = args.out / "router_metrics.parquet"
    aggregate_path = args.out / "router_metric_summary.parquet"
    manifest_path = args.out / "run_manifest.json"
    if not metrics_path.exists():
        rows: list[dict[str, object]] = []
        timestamp_rows: list[pd.DataFrame] = []
        for month in months:
            score = pd.read_parquet(score_root / f"month={month:%Y-%m}.parquet")
            month_rows, month_timestamp = _metric_rows(score, policy, month, args.route_fractions)
            rows.extend(month_rows)
            timestamp_rows.append(month_timestamp)
        metrics = pd.DataFrame(rows)
        metrics.to_parquet(metrics_path, index=False, compression="zstd")
        pd.concat(timestamp_rows, ignore_index=True).to_parquet(args.out / "router_timestamp_metrics.parquet", index=False, compression="zstd")
        _aggregate_metrics(metrics).to_parquet(aggregate_path, index=False, compression="zstd")
    if not manifest_path.exists():
        summary = pd.read_parquet(aggregate_path).to_dict(orient="records")
        _write_json_exclusive(manifest_path, {
            **contract,
            "status": "complete", "outputs": ["target_free_scores", "router_metrics.parquet", "router_timestamp_metrics.parquet", "router_metric_summary.parquet"],
            "selection_status": "development selection only; no untouched promotion claim",
            "auxiliary_contract": (
                "primary-only selected downstream ledger; complete A-E blend previously failed its gate"
                if args.primary_only else
                "complete A-E replacement; B0/efficiency/timing auxiliary targets were not trained"
            ),
            "metric_aggregation": "ER50, winner recall, and selected EV are averaged equally across decision timestamps; no metric or HPO selector pools candidate rows globally",
            "group_removal_rule": "only if router_full_ae_rank improves the predeclared timestamp-averaged ER50 score versus primary-only; then run LOO A/B/C/D/E using the same strict fold contract",
            "summary": summary,
        })


def _parse_months(value: str) -> tuple[pd.Timestamp, ...]:
    return tuple(_utc(token + "-01") for token in value.split(",") if token.strip())


def _parse_route_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not fractions or any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise argparse.ArgumentTypeError("route fractions must lie in (0, 1]")
    if tuple(sorted(set(fractions))) != fractions:
        raise argparse.ArgumentTypeError("route fractions must be unique and strictly ascending")
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root", type=Path, action="append",
        help=("explicit causal monthly feature store; repeat only for disjoint "
              "month ranges.  A month present in more than one root fails closed"),
    )
    parser.add_argument("--aux-root", type=Path, default=DEFAULT_AUX_ROOT)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--sealed-feature-contract", type=Path, help="completed immutable JSON receipt containing the exact 120-field contract")
    parser.add_argument("--feature-list", type=Path, help="JSON-selected ordered subset of the sealed 120 causal fields")
    parser.add_argument(
        "--full-feature-contract", type=Path,
        help=("explicit frozen full-universe causal feature contract from the recall selector; "
              "mutually exclusive with the 120-field feature-list path"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", type=_parse_months, default=DEFAULT_MONTHS)
    parser.add_argument("--primary-target", choices=ALL_PRIMARY_TARGETS, default="P1_net_0_50_100_150_300")
    parser.add_argument("--aux-groups", type=lambda text: tuple(item.strip() for item in text.split(",") if item.strip()), default=ALL_GROUPS)
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    parser.add_argument("--reserve-days", type=int, default=DEFAULT_RESERVE_DAYS)
    parser.add_argument("--train-cap", type=int, default=240_000)
    parser.add_argument("--route-fractions", type=_parse_route_fractions, default=DEFAULT_ROUTE_FRACTIONS)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--reuse-aux-source", type=Path)
    parser.add_argument("--primary-only", action="store_true", help="score only the selected main target; valid only after the A-E group gate rejects the full blend")
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=.045)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--min-child-fraction", type=float, default=.006)
    parser.add_argument("--min-child-floor", type=int, default=300)
    parser.add_argument("--min-split-gain", type=float, default=.001)
    parser.add_argument("--subsample", type=float, default=.82)
    parser.add_argument("--feature-fraction", type=float, default=.82)
    parser.add_argument("--l1", type=float, default=.05)
    parser.add_argument("--l2", type=float, default=6.0)
    parser.add_argument("--max-bin", type=int, default=127)
    parser.add_argument("--truncation", type=int, default=20)
    parser.add_argument("--label-gains", help="comma-separated gains; must match the target class count")
    parser.add_argument("--objective", choices=("lambdarank", "rank_xendcg"), default="lambdarank")
    parser.add_argument(
        "--row-weight-scheme", choices=_ROW_WEIGHT_SCHEMES, default="uniform",
        help=("W0 incumbent, W1 equal-timestamp, W2 bounded economic, W3 "
              "positive-recall, or predeclared W4 bounded combined authority"),
    )
    parser.add_argument("--row-weight-floor-bps", type=float, default=100.0)
    parser.add_argument("--row-weight-cap-bps", type=float, default=250.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=0, help="strictly internal chronological tree-count selection; zero disables")
    parser.add_argument("--inner-validation-fraction", type=float, default=.20)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--defer-aggregate", action="store_true",
        help=("seal only target-free monthly score/audit receipts; defer legacy "
              "aggregate metrics to the streaming Router utility evaluator"),
    )
    args = parser.parse_args()
    if args.feature_root is None:
        args.feature_root = [DEFAULT_FEATURE_ROOT]
    else:
        args.feature_root = list(_normalise_feature_roots(args.feature_root))
    # The predeclared history search starts at two months.  The downstream
    # support checks remain the real safety gate for a short ledger.
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 20_000:
        raise ValueError("train history/reserve/cap violate the predeclared strict-router minimums")
    if not 1 <= args.n_jobs <= DEFAULT_N_JOBS:
        raise ValueError(f"n_jobs must be within [1, {DEFAULT_N_JOBS}] to avoid parallel-fold oversubscription")
    if args.max_depth < 1 or args.num_leaves < 2 or args.n_estimators < 20 or args.truncation < 1:
        raise ValueError("invalid ranker geometry")
    if args.min_child_fraction <= 0 or args.min_child_floor < 1 or args.learning_rate <= 0:
        raise ValueError("invalid ranker support or learning rate")
    if not 0 < args.subsample <= 1 or not 0 < args.feature_fraction <= 1:
        raise ValueError("subsample and feature fraction must lie in (0, 1]")
    if args.row_weight_cap_bps <= 0:
        raise ValueError("row-weight cap must be positive")
    if args.early_stopping_rounds < 0:
        raise ValueError("early-stopping rounds cannot be negative")
    if args.early_stopping_rounds and not 0.05 <= args.inner_validation_fraction < .5:
        raise ValueError("inner validation fraction must lie in [0.05, 0.5) when early stopping is enabled")
    run(args)


if __name__ == "__main__":
    main()
