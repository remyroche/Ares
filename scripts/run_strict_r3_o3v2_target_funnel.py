#!/usr/bin/env python3
"""Strict-prequential O3-v2 target ablation, isolated from live inference.

The initial target screen may score all five frozen physical slots in order to
choose one slot *per target*.  Once that physical-slot contract is frozen,
all successor target runs fit and emit exactly that one slot for each target.
Resolved policy/path semantics are loaded only for training rows.  A held
score receipt is persisted before canonical policy outcomes are joined to
calculate metrics.  This runner deliberately does not call MC1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_strict_r3_enhanced_base_live_stack_challenger as parent  # noqa: E402


SCHEMA = "strict_r3_o3v2_target_funnel_v4"
SEED = 1729
TRAIN_MONTHS = 6
RESERVE_DAYS = 28
# Semantic/full-policy source history begins in April 2025.  November is the
# first held month with a complete six-month fit ending before its reserve.
DEFAULT_MONTHS = tuple(pd.date_range("2025-11-01", "2026-07-01", freq="MS", tz="UTC"))
PRIMARY_ARMS = (
    "T1_economic_residual_lambdarank",
    "T2_economic_residual_ordinal",
    "T4_hard_inversion_lambdarank",
    "T6_rank_error_ordinal",
)
ALL_ARMS = (
    *PRIMARY_ARMS,
    "T3_pair_residual_lambdarank",
    "T5_rank_error_lambdarank",
    "T7_exit4_lambdarank",
    "T8_exit5_lambdarank",
    "T9_exit5_ordinal",
)
PROHIBITED_SCORE_COLUMNS = {
    "policy_net_bps", "policy_gross_bps", "policy_path_valid", "semantic_path_valid",
    "semantic_archetype", "semantic_tbm_event", "semantic_axis_a_sequence",
    "semantic_axis_f_exit4", "semantic_axis_f_exit5", "semantic_label_available_ts",
}
ECONOMIC_RESIDUAL_EDGES = (-250.0, -100.0, -30.0, 30.0, 100.0, 250.0)
RANK_ERROR_EDGES = (-0.20, -0.05, 0.05, 0.20)
PHYSICAL_SLOTS = frozenset({
    "cap100_ordinary", "cap80_ordinary", "cap120_equal_month",
    "cap40_equal_month", "cap60_equal_month",
})


def _economic_residual_grade(values: np.ndarray) -> np.ndarray:
    """Seven ordered economic-residual states, fixed before the held folds."""
    return np.digitize(np.asarray(values, dtype=float), ECONOMIC_RESIDUAL_EDGES).astype(np.int32)


def _rank_error_grade(values: np.ndarray) -> np.ndarray:
    """Five ordered within-query base-rank-error states."""
    return np.digitize(np.asarray(values, dtype=float), RANK_ERROR_EDGES).astype(np.int32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.parquet")) if path.is_dir() else (path,):
        digest.update(str(child).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_physical_slot_contract(path: Path | None, arms: Sequence[str], *, query_mode: str) -> dict[str, str] | None:
    """Load a frozen one-slot-per-target contract without touching outcomes.

    ``None`` is retained only for the first physical-slot discovery run.  A
    successor run must name a contract so that it cannot silently restore a
    five-slot ensemble.
    """
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_o3v2_physical_slot_selection_v1":
        raise AssertionError("unknown physical-slot selection schema")
    if payload.get("query_mode") != query_mode:
        raise AssertionError(
            f"physical-slot query {payload.get('query_mode')!r} differs from requested {query_mode!r}"
        )
    slots = payload.get("selected_slots")
    if not isinstance(slots, dict):
        raise AssertionError("physical-slot contract has no selected_slots mapping")
    missing = sorted(set(arms) - set(slots))
    if missing:
        raise AssertionError(f"physical-slot contract does not cover requested arms: {missing}")
    invalid = {str(slots[arm]) for arm in arms} - PHYSICAL_SLOTS
    if invalid:
        raise AssertionError(f"physical-slot contract names unknown slots: {sorted(invalid)}")
    return {str(arm): str(slots[arm]) for arm in arms}


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _strict_train_start(month: pd.Timestamp) -> pd.Timestamp:
    """Six full calendar months ending before the embargo reserve."""
    reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
    return reserve_start - pd.DateOffset(months=TRAIN_MONTHS)


def _require_complete_feature_months(root: Path, start: pd.Timestamp, end: pd.Timestamp, *, context: str) -> None:
    """Fail closed instead of silently shortening a declared fit window."""
    periods = pd.period_range(
        start.to_period("M"),
        (end - pd.Timedelta(nanoseconds=1)).to_period("M"),
        freq="M",
    )
    missing = [
        period.strftime("%Y-%m")
        for period in periods
        if not (root / f"month={period.strftime('%Y-%m')}" / "scores_features.parquet").exists()
    ]
    if missing:
        raise AssertionError(
            f"{context}: missing source months for declared strict-prequential window: {missing}"
        )


def _window_features(root: Path, start: pd.Timestamp, end: pd.Timestamp, columns: Sequence[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    periods = pd.period_range(start.to_period("M"), (end - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M")
    for period in periods:
        source = root / f"month={period.strftime('%Y-%m')}" / "scores_features.parquet"
        if not source.exists():
            continue
        part = pd.read_parquet(source, columns=list(columns))
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        pieces.append(part.loc[part["__decision_ts__"].ge(start) & part["__decision_ts__"].lt(end)].copy())
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(columns))


def _training_window(
    feature_root: Path,
    semantic_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    columns: Sequence[str],
) -> pd.DataFrame:
    features = _window_features(feature_root, start, end, columns)
    if features.empty:
        return features
    pieces: list[pd.DataFrame] = []
    for token, part in features.groupby(features["__decision_ts__"].dt.strftime("%Y-%m"), sort=True):
        semantic = semantic_root / "parts" / f"month={token}" / "semantics.parquet"
        if not semantic.exists():
            continue
        labels = pd.read_parquet(semantic)
        labels["__decision_ts__"] = pd.to_datetime(labels["__decision_ts__"], utc=True, errors="raise")
        pieces.append(part.merge(labels, on=["candidate_id", "__decision_ts__", "side_name"], how="left", validate="one_to_one"))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(columns))


def _base_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    timestamp = out.groupby("__decision_ts__", sort=False)
    for source, target in (("base_bps", "o3v2_b0_rank"), ("efficiency_bps", "o3v2_e_rank"), ("timing_bps", "o3v2_t_rank")):
        out[target] = timestamp[source].rank(pct=True, method="average").astype(np.float32)
    coordinate = out.loc[:, ["base_bps", "efficiency_bps", "timing_bps"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    out["o3v2_coord_min"] = np.nanmin(coordinate, axis=1)
    out["o3v2_coord_max"] = np.nanmax(coordinate, axis=1)
    out["o3v2_coord_median"] = np.nanmedian(coordinate, axis=1)
    out["o3v2_coord_std"] = np.nanstd(coordinate, axis=1)
    out["o3v2_coord_range"] = out["o3v2_coord_max"] - out["o3v2_coord_min"]
    routed = out["enhanced_base_routed"].fillna(False).astype(bool)
    work = out.loc[routed, ["__decision_ts__", "enhanced_base_bps"]].copy()
    if work.empty:
        return out
    summary = work.groupby("__decision_ts__", sort=False)["enhanced_base_bps"].agg(["size", "std", "min", "max"])
    summary["o3v2_query_count"] = summary["size"]
    summary["o3v2_query_std"] = summary["std"].fillna(0.0)
    summary["o3v2_query_range"] = summary["max"] - summary["min"]
    ordered = work.sort_values(["__decision_ts__", "enhanced_base_bps"], ascending=[True, False], kind="stable")
    ordered["__next__"] = ordered.groupby("__decision_ts__", sort=False)["enhanced_base_bps"].shift(-1)
    ordered["__third__"] = ordered.groupby("__decision_ts__", sort=False)["enhanced_base_bps"].shift(-2)
    top = ordered.groupby("__decision_ts__", sort=False).first()
    summary["o3v2_query_top_gap"] = top["enhanced_base_bps"] - top["__next__"]
    summary["o3v2_query_top2_gap"] = top["__next__"] - top["__third__"]
    for column in ("o3v2_query_count", "o3v2_query_std", "o3v2_query_range", "o3v2_query_top_gap", "o3v2_query_top2_gap"):
        out[column] = out["__decision_ts__"].map(summary[column]).fillna(0.0).astype(np.float32)
    return out


def _specs(base_fields: tuple[str, ...], *, query_mode: str) -> tuple[object, ...]:
    """Reuse the frozen selected five physical slots plus O3-v2 score geometry."""
    base_specs = parent._head_specs(base_fields)
    extras = (
        "base_bps", "efficiency_bps", "timing_bps", "enhanced_base_bps", "base_rank_ts",
        "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std",
        "o3v2_b0_rank", "o3v2_e_rank", "o3v2_t_rank", "o3v2_coord_min", "o3v2_coord_max",
        "o3v2_coord_median", "o3v2_coord_std", "o3v2_coord_range", "o3v2_query_count",
        "o3v2_query_std", "o3v2_query_range", "o3v2_query_top_gap", "o3v2_query_top2_gap",
    )
    result = []
    for spec in base_specs:
        fields = tuple(dict.fromkeys((*spec.fields, *extras)))
        result.append(parent.ConsensusHeadSpec(
            name=spec.name, cap=spec.cap, weight_mode=spec.weight_mode, query=query_mode,
            fields=fields, target_edges_bps=spec.target_edges_bps, params=dict(spec.params),
        ))
    return tuple(result)


def _label_specs(specs: Sequence[object], grade: np.ndarray | None) -> tuple[object, ...]:
    """Keep frozen tree geometry while giving each declared target full gain support."""
    if grade is None:
        return tuple(specs)
    classes = int(np.nanmax(np.asarray(grade, dtype=float))) + 1
    # The economic residual arm has seven fixed classes.  The sequence below
    # preserves increasing tail emphasis without silently clipping a label to
    # the incumbent five-grade contract.
    gains = [0, 1, 2, 4, 7, 12, 20][:classes]
    return tuple(parent.ConsensusHeadSpec(
        name=spec.name, cap=spec.cap, weight_mode=spec.weight_mode, query=spec.query,
        fields=spec.fields, target_edges_bps=spec.target_edges_bps,
        params={**dict(spec.params), "label_gain": gains},
    ) for spec in specs)


def _anchor_and_targets(train: pd.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray | None, str, str]:
    policy = pd.to_numeric(train["semantic_policy_net_bps"], errors="coerce").to_numpy(float)
    base_rank = pd.to_numeric(train["base_rank_ts"], errors="coerce").to_numpy(float)
    anchor_model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(base_rank, policy)
    anchor = anchor_model.predict(base_rank)
    residual = np.clip(policy - anchor, -500.0, 500.0)
    realised_rank = train.groupby("__decision_ts__", sort=False)["semantic_policy_net_bps"].rank(pct=True, method="average").to_numpy(float)
    rank_error = realised_rank - base_rank
    exit4 = train["semantic_axis_f_exit4"].astype("string").fillna("timeout").astype(str)
    exit5 = train["semantic_axis_f_exit5"].astype("string").fillna("timeout").astype(str)
    if arm == "T1_economic_residual_lambdarank":
        return residual, _economic_residual_grade(residual), "ordinal_lambdarank", "economic_residual"
    if arm == "T2_economic_residual_ordinal":
        # Ordered regression must consume the predeclared economic states,
        # not the raw clipped bps residual.  The latter was an implementation
        # mismatch in the first exploratory receipt.
        ordinal = _economic_residual_grade(residual)
        return ordinal.astype(np.float32), ordinal, "l2_regression", "economic_residual_ordinal"
    if arm == "T3_pair_residual_lambdarank":
        return residual, _economic_residual_grade(residual), "ordinal_lambdarank", "pair_residual"
    if arm == "T4_hard_inversion_lambdarank":
        return residual, _economic_residual_grade(residual), "ordinal_lambdarank", "hard_inversion"
    if arm == "T5_rank_error_lambdarank":
        return rank_error, _rank_error_grade(rank_error), "ordinal_lambdarank", "rank_error"
    if arm == "T6_rank_error_ordinal":
        ordinal = _rank_error_grade(rank_error)
        return ordinal.astype(np.float32), ordinal, "l2_regression", "rank_error_ordinal"
    if arm == "T7_exit4_lambdarank":
        value = exit4.map({"stop": 0., "timeout": 1., "smooth_protection": 2., "trailing": 3.}).fillna(1.).to_numpy(float)
        return value, value.astype(np.int32), "ordinal_lambdarank", "exit4"
    if arm == "T8_exit5_lambdarank":
        value = exit5.map({"stop": 0., "timeout": 1., "smooth_protection": 2., "regular_trailing": 3., "large_trailing": 4.}).fillna(1.).to_numpy(float)
        return value, value.astype(np.int32), "ordinal_lambdarank", "exit5"
    if arm == "T9_exit5_ordinal":
        value = exit5.map({"stop": 0., "timeout": 1., "smooth_protection": 2., "regular_trailing": 3., "large_trailing": 4.}).fillna(1.).to_numpy(float)
        return value, value.astype(np.int32), "l2_regression", "exit5_ordinal"
    raise ValueError(arm)


def _score_columns(scored: pd.DataFrame) -> pd.DataFrame:
    head_ranks = [column for column in scored if column.startswith("head__") and column.endswith("__rank")]
    out = scored.loc[:, ["candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed", "enhanced_base_bps", "base_rank_ts", "conditional_consensus_rank", "ordinary_shadow_consensus_rank", "head_agreement_std", *head_ranks]].copy()
    out["o3v2_rank_75_25"] = 0.75 * pd.to_numeric(out["base_rank_ts"], errors="coerce") + 0.25 * pd.to_numeric(out["conditional_consensus_rank"], errors="coerce")
    return out


def _metric_rows(scores: pd.DataFrame, policy: pd.DataFrame, *, arm: str, month: pd.Timestamp) -> list[dict[str, object]]:
    joined = scores.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    valid = joined["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
    work = joined.loc[valid & joined["enhanced_base_routed"].fillna(False).astype(bool)].copy()
    result: list[dict[str, object]] = []
    for field in ("conditional_consensus_rank", "o3v2_rank_75_25"):
        score = pd.to_numeric(work[field], errors="coerce").to_numpy(float)
        outcome = pd.to_numeric(work["policy_net_bps"], errors="coerce").to_numpy(float)
        base = pd.to_numeric(work["base_rank_ts"], errors="coerce").to_numpy(float)
        valid_score = np.isfinite(score) & np.isfinite(outcome) & np.isfinite(base)
        ic = float(spearmanr(score[valid_score], outcome[valid_score]).statistic) if valid_score.sum() >= 12 else np.nan
        # A fixed, score-free base-decile residualization measures whether a
        # correction adds ordering beyond the routed upstream rank.  It is a
        # diagnostic only; no held outcome is fed back into the score.
        bucket = np.minimum(9, np.maximum(0, np.floor(base * 10))).astype(int)
        rank_value = pd.Series(score)
        outcome_value = pd.Series(outcome)
        score_residual = rank_value - rank_value.groupby(bucket, sort=False).transform("mean")
        outcome_residual = outcome_value - outcome_value.groupby(bucket, sort=False).transform("mean")
        conditional_valid = valid_score & np.isfinite(score_residual.to_numpy(float)) & np.isfinite(outcome_residual.to_numpy(float))
        conditional_ic = float(spearmanr(score_residual.to_numpy(float)[conditional_valid], outcome_residual.to_numpy(float)[conditional_valid]).statistic) if conditional_valid.sum() >= 12 else np.nan
        base_correlation = float(spearmanr(score[valid_score], base[valid_score]).statistic) if valid_score.sum() >= 12 else np.nan
        for tail in (.01, .02, .03, .05, .10):
            threshold = np.quantile(score[valid_score], 1.0 - tail, method="higher")
            selected = outcome[valid_score & (score >= threshold)]
            result.append({"arm": arm, "month": f"{month:%Y-%m}", "score": field, "tail": tail, "trades": int(len(selected)), "net_ev_bps_per_trade": float(np.mean(selected)), "net_sum_bps": float(np.sum(selected)), "policy_rank_ic": ic, "conditional_policy_rank_ic": conditional_ic, "base_rank_correlation": base_correlation})
    return result


def _control_metrics(control_root: Path, policy: pd.DataFrame, months: Sequence[pd.Timestamp]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month in months:
        source = control_root / "target_free_scores" / "O3_calibrated_residual_semantic" / f"month={month:%Y-%m}.parquet"
        if not source.exists():
            raise FileNotFoundError(source)
        raw = pd.read_parquet(source)
        raw = raw.rename(columns={"orthogonal_consensus_rank": "conditional_consensus_rank"})
        raw["o3v2_rank_75_25"] = .75 * pd.to_numeric(raw["base_rank_ts"], errors="coerce") + .25 * pd.to_numeric(raw["conditional_consensus_rank"], errors="coerce")
        raw["__symbol__"] = "control_identity_not_needed"
        rows.extend(_metric_rows(raw, policy, arm="T0_current_o3_control", month=month))
    return rows


def _write_json_exclusive(path: Path, payload: object) -> None:
    """Write an immutable small receipt, failing rather than overwriting."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _contract(
    *, feature_root: Path, semantic_root: Path, policy_path: Path, bundle_root: Path,
    control_root: Path, months: Sequence[pd.Timestamp], arms: Sequence[str], query_mode: str,
    physical_slot_selection: Path | None,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "feature_root": str(feature_root), "semantic_root": str(semantic_root),
        "policy_path": str(policy_path), "bundle_root": str(bundle_root),
        "control_root": str(control_root), "months": [f"{month:%Y-%m}" for month in months],
        "arms": list(arms), "query_mode": query_mode,
        "physical_slot_selection": str(physical_slot_selection) if physical_slot_selection else None,
        "physical_slot_selection_sha256": _sha256(physical_slot_selection) if physical_slot_selection else None,
    }


def _finalise(
    *, feature_root: Path, semantic_root: Path, policy_path: Path, bundle_root: Path,
    control_root: Path, out: Path, months: Sequence[pd.Timestamp], arms: Sequence[str],
    query_mode: str, policy: pd.DataFrame, specs_by_arm: dict[str, Sequence[object]],
    physical_slots: dict[str, str] | None, physical_slot_selection: Path | None,
) -> None:
    """Aggregate append-only per-arm/month receipts once every job is complete."""
    metrics = _control_metrics(control_root, policy, months)
    audits: list[dict[str, object]] = []
    for arm in arms:
        for month in months:
            score_path = out / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
            audit_path = out / "audit_parts" / f"{arm}__{month:%Y-%m}.json"
            if not score_path.exists() or not audit_path.exists():
                raise AssertionError(f"cannot finalise missing immutable job receipt: {arm} {month:%Y-%m}")
            score = pd.read_parquet(score_path)
            metrics.extend(_metric_rows(score, policy, arm=arm, month=month))
            audits.append(json.loads(audit_path.read_text()))
    metrics_path = out / "target_funnel_metrics.parquet"
    audit_path = out / "target_funnel_audit.parquet"
    manifest_path = out / "run_manifest.json"
    if not metrics_path.exists():
        pd.DataFrame(metrics).to_parquet(metrics_path, index=False, compression="zstd")
    if not audit_path.exists():
        pd.DataFrame(audits).to_parquet(audit_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "scope": "offline O3-v2 target research; does not update live/canonical/MC1 artifacts",
        "arms": list(arms), "control": "immutable O3 calibrated-residual control read without refitting",
        "months": [f"{month:%Y-%m}" for month in months], "feature_root": str(feature_root), "semantic_root": str(semantic_root),
        "policy_path": str(policy_path), "bundle_root": str(bundle_root), "control_root": str(control_root),
        "source_hashes": {"feature_root": _sha256(feature_root), "semantic_root": _sha256(semantic_root), "policy_path": _sha256(policy_path), "control_root": _sha256(control_root)},
        "causality": {"fit": "six full preceding resolved calendar months before reserve", "reserve": "28 days excluded from fit", "semantics": "training labels/targets only", "held": "target-free score receipt persisted before policy join", "query": query_mode, "pair_residual": "T3 pairs train-only calibrated residual, never realised raw policy net", "mc1": "not modified or refit by Stage 1"},
        "physical_slot_selection": str(physical_slot_selection) if physical_slot_selection else None,
        "physical_slot_selection_sha256": _sha256(physical_slot_selection) if physical_slot_selection else None,
        "selected_physical_slots": physical_slots,
        "head_contract": {
            arm: [
                {"name": spec.name, "fields": list(spec.fields), "query": spec.query,
                 "weight_mode": spec.weight_mode, "params": spec.params}
                for spec in specs
            ]
            for arm, specs in specs_by_arm.items()
        },
    }
    if not manifest_path.exists():
        _write_json_exclusive(manifest_path, manifest)


def run(
    *, feature_root: Path, semantic_root: Path, policy_path: Path, bundle_root: Path,
    control_root: Path, out: Path, months: Sequence[pd.Timestamp], arms: Sequence[str],
    query_mode: str, physical_slot_selection: Path | None = None, resume: bool = False,
    max_jobs: int | None = None,
) -> None:
    if query_mode not in {"exact_timestamp_side", "exact_timestamp_baseband_side", "cycle_4h_side"}:
        raise ValueError(f"unsupported O3-v2 query mode: {query_mode}")
    physical_slots = _load_physical_slot_contract(physical_slot_selection, arms, query_mode=query_mode)
    run_contract = _contract(
        feature_root=feature_root, semantic_root=semantic_root, policy_path=policy_path,
        bundle_root=bundle_root, control_root=control_root, months=months, arms=arms,
        query_mode=query_mode, physical_slot_selection=physical_slot_selection,
    )
    contract_path = out / "run_contract.json"
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        if not contract_path.exists():
            raise AssertionError("refusing to resume an output without its immutable run_contract receipt")
        if json.loads(contract_path.read_text()) != run_contract:
            raise AssertionError("resume contract does not exactly match the existing immutable target funnel")
    else:
        out.mkdir(parents=True)
        _write_json_exclusive(contract_path, run_contract)
    policy = pd.read_parquet(policy_path, columns=("candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"))
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    if policy["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy ledger contains duplicate identities")
    paths = parent.Paths(*(Path("unused") for _ in range(5)), bundle_root)
    base_fields = parent._base_fields(paths)
    all_specs = _specs(base_fields, query_mode=query_mode)
    specs_by_arm = {
        arm: tuple(spec for spec in all_specs if physical_slots is None or spec.name == physical_slots[arm])
        for arm in arms
    }
    if any(len(specs_by_arm[arm]) != (5 if physical_slots is None else 1) for arm in arms):
        raise AssertionError("physical-slot contract did not resolve the expected number of head specifications")
    source_columns = tuple(dict.fromkeys((
        "candidate_id", "__decision_ts__", "side_name", "enhanced_base_routed",
        "enhanced_base_bps", "base_rank_ts", "base_bps", "efficiency_bps", "timing_bps",
        "e_minus_t", "e_minus_b0", "t_minus_b0", "base_component_std", *base_fields,
    )))
    audit_root = out / "audit_parts"
    audit_root.mkdir(exist_ok=True)
    completed_now = 0
    for arm_index, arm in enumerate(arms):
        root = out / "target_free_scores" / arm
        root.mkdir(parents=True, exist_ok=True)
        for month in months:
            score_path = root / f"month={month:%Y-%m}.parquet"
            audit_part = audit_root / f"{arm}__{month:%Y-%m}.json"
            if score_path.exists() and audit_part.exists():
                continue
            if score_path.exists() != audit_part.exists():
                raise AssertionError(f"partial job receipt for {arm} {month:%Y-%m}; preserve it and restart with a new output root")
            if max_jobs is not None and completed_now >= max_jobs:
                break
            end = _month_end(month)
            reserve_start = month - pd.Timedelta(days=RESERVE_DAYS)
            train_start = _strict_train_start(month)
            _require_complete_feature_months(
                feature_root, train_start, reserve_start,
                context=f"{arm} {month:%Y-%m} training",
            )
            _require_complete_feature_months(
                feature_root, month, end,
                context=f"{arm} {month:%Y-%m} held",
            )
            train = _training_window(feature_root, semantic_root, start=train_start, end=reserve_start, columns=source_columns)
            held = _window_features(feature_root, month, end, source_columns)
            # Do not trust an upstream persisted route flag: legacy panels used
            # a percentile threshold and could include one excess row at a
            # timestamp.  Recompute the deterministic top-30% route from the
            # target-free upstream score for both training and held data.
            train["enhanced_base_routed"] = parent._exact_timestamp_top_fraction(
                train, "enhanced_base_bps", parent.BASE_ROUTE,
            ).to_numpy(bool)
            held["enhanced_base_routed"] = parent._exact_timestamp_top_fraction(
                held, "enhanced_base_bps", parent.BASE_ROUTE,
            ).to_numpy(bool)
            train = _base_geometry(train)
            held = _base_geometry(held)
            valid = (
                train["enhanced_base_routed"].fillna(False).astype(bool)
                & train["semantic_path_valid"].fillna(False).astype(bool)
                & train["semantic_label_available_ts"].lt(reserve_start)
                & np.isfinite(pd.to_numeric(train["semantic_policy_net_bps"], errors="coerce"))
            )
            train = train.loc[valid].copy()
            held = held.loc[held["enhanced_base_routed"].fillna(False).astype(bool)].copy()
            if len(train) < 5_000 or len(held) < 1_000:
                raise AssertionError(f"{arm} {month:%Y-%m}: insufficient strict support train={len(train)} held={len(held)}")
            target, grade, objective, mode = _anchor_and_targets(train, arm)
            fit_specs = _label_specs(specs_by_arm[arm], grade)
            # T3 is explicitly a *calibrated residual* pairwise objective.
            # The shared pair sampler reads ``policy_net_bps`` as its ranking
            # quantity, so pass the already train-only-calibrated residual
            # rather than the realised raw policy net.  This is a schema-v3
            # correction; v1/v2 raw-policy pair artifacts remain preserved
            # but are not valid evidence for the requested T3 definition.
            pair_train = train.rename(columns={"semantic_policy_net_bps": "policy_net_bps"}).copy()
            if mode == "pair_residual":
                pair_train["policy_net_bps"] = np.asarray(target, dtype=np.float32)
            if mode == "pair_residual":
                heads, pair_audit = parent._fit_heads(pair_train, target, fit_specs, objective=objective, grade=grade, pairwise_mode="near_tie_diff50")
            elif mode == "hard_inversion":
                heads, pair_audit = parent._fit_heads(pair_train, target, fit_specs, objective=objective, grade=grade, pairwise_mode="base_inversion100")
            else:
                heads, pair_audit = parent._fit_heads(train, target, fit_specs, objective=objective, grade=grade)
            score = _score_columns(parent._score_heads(held, heads))
            leak = PROHIBITED_SCORE_COLUMNS.intersection(score.columns)
            if leak:
                raise AssertionError(f"{arm} target-free score receipt leaked labels: {sorted(leak)}")
            score.to_parquet(score_path, index=False, compression="zstd")
            # Only after the immutable target-free score file is on disk can outcomes be joined for diagnostics.
            audit = {
                "arm": arm, "month": f"{month:%Y-%m}", "train_start": str(train_start), "reserve_start": str(reserve_start), "train_rows": int(len(train)), "held_rows": int(len(held)),
                "objective": objective, "mode": mode, "target_mean": float(np.nanmean(target)),
                "target_std": float(np.nanstd(target)), "classes": int(np.unique(grade).size) if grade is not None else np.nan,
                "semantic_valid_fraction": float(train["semantic_path_valid"].mean()), "pair_audit": json.dumps(pair_audit),
                "query_mode": query_mode,
                "physical_slots": [spec.name for spec in fit_specs],
            }
            _write_json_exclusive(audit_part, audit)
            completed_now += 1
            print(json.dumps({"event": "scored", **audit}), flush=True)
        if max_jobs is not None and completed_now >= max_jobs:
            break
    expected = [(arm, month) for arm in arms for month in months]
    complete = all(
        (out / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet").exists()
        and (audit_root / f"{arm}__{month:%Y-%m}.json").exists()
        for arm, month in expected
    )
    if complete:
        _finalise(
            feature_root=feature_root, semantic_root=semantic_root, policy_path=policy_path,
            bundle_root=bundle_root, control_root=control_root, out=out, months=months,
            arms=arms, query_mode=query_mode, policy=policy, specs_by_arm=specs_by_arm,
            physical_slots=physical_slots, physical_slot_selection=physical_slot_selection,
        )
        print(json.dumps({"event": "finalised", "jobs": len(expected)}), flush=True)
    else:
        completed = sum(
            (out / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet").exists()
            and (audit_root / f"{arm}__{month:%Y-%m}.json").exists()
            for arm, month in expected
        )
        progress = out / f"progress_{completed:03d}_of_{len(expected):03d}.json"
        if not progress.exists():
            _write_json_exclusive(progress, {"schema": SCHEMA, "completed_jobs": completed, "expected_jobs": len(expected), "query_mode": query_mode})
        print(json.dumps({"event": "checkpoint", "completed_jobs": completed, "expected_jobs": len(expected)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", help="YYYY-MM list; default is 2025-10..2026-07")
    parser.add_argument("--arms", help="comma-separated arms; default is high-priority Stage-1 arms")
    parser.add_argument("--query-mode", default="exact_timestamp_side", choices=("exact_timestamp_side", "exact_timestamp_baseband_side", "cycle_4h_side"), help="Causal LambdaRank query contract; schema v3 defaults to exact timestamp × side.")
    parser.add_argument(
        "--physical-slot-selection", type=Path,
        help="frozen one-slot-per-target contract; successor runs fit only the declared physical slot for each target",
    )
    parser.add_argument("--resume", action="store_true", help="resume only a matching append-only run_contract")
    parser.add_argument("--max-jobs", type=int, help="run at most this many arm/month jobs before emitting an immutable checkpoint")
    args = parser.parse_args()
    months = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in args.months.split(",")) if args.months else DEFAULT_MONTHS
    arms = tuple(args.arms.split(",")) if args.arms else PRIMARY_ARMS
    unsupported = sorted(set(arms) - set(ALL_ARMS))
    if unsupported:
        raise ValueError(f"unsupported O3-v2 target arms: {unsupported}")
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    run(
        feature_root=args.feature_root, semantic_root=args.semantic_root, policy_path=args.policy_path,
        bundle_root=args.bundle_root, control_root=args.control_root, out=args.out, months=months,
        arms=arms, query_mode=args.query_mode, physical_slot_selection=args.physical_slot_selection,
        resume=args.resume, max_jobs=args.max_jobs,
    )


if __name__ == "__main__":
    main()
