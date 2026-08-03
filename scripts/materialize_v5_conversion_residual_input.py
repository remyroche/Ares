#!/usr/bin/env python3
"""Materialize the exact March-April input for the next conversion residual."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_canonical_economic_conversion_transition_labels import (
    add_frozen_causal_score_deciles,
)

V5 = ROOT / "data_perp/artifacts/short_winner_causal_recent_ev_mapping_20260730_v5"
EXTENSION = ROOT / "data_perp/artifacts/v5_early_short_oof_extension_20260730_v1"
BASE = ROOT / "data_perp/artifacts/febapr2025_canonical_base_oof_20260727_v1"
RESIDUAL = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1"
LABELS = ROOT / "data_perp/artifacts/febapr2025_execution_ev_current_spread_12h_labels_20260727_v1"
PEAK = ROOT / "data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2"
SLOPE = ROOT / "data_perp/artifacts/febapr2025_historical_future_slope_fixed_geometry_oof_20260730_v1"
MAE = ROOT / "data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2_timing_mae"
CONTEXT = ROOT / "data_perp/artifacts/canonical_economic_conversion_transition_context_20260729_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/v5_conversion_residual_input_20260730_v3"
IDENTITY = ("candidate_id", "side_name")
EXPECTED_ROWS = 110_730
EXPECTED_MARCH_ROWS = 41_472
EXPECTED_APRIL_ROWS = 69_258

CORE_CONTEXT = (
    "context__range_24h_pct__mean",
    "context__meta_raw__volatility_zscore__mean",
    "context__trend_r2_24__mean",
    "context__jump_intensity__mean",
    "context__meta_raw__chop_score__mean",
)
CORE_TRANSITIONS = tuple(
    f"context__preentry_transition__{name}__delta_{horizon}h__mean"
    for name in (
        "range_24h_pct",
        "meta_raw__volatility_zscore",
        "trend_r2_24",
        "jump_intensity",
        "meta_raw__chop_score",
    )
    for horizon in (3, 12)
)
REGIME_CONTEXT = (
    "context__regime_source_shock_impulse_score__mean",
    "context__regime_source_execution_quality_score__mean",
    "context__regime_source_execution_risk_score__mean",
    "context__regime_source_oi_agreement_score__mean",
    "context__regime_source_compression_score__mean",
    "context__regime_source_loud_breakout_impulse_score__mean",
    "context__regime_source_dirty_shock_avoid_score__mean",
    "context__regime_source_clean_execution_context_score__mean",
)
COHORT_CONTEXT = (
    "context__side_sign",
    "context__frozen_base_score_decile",
)
APPROVED_CONTEXT = (*CORE_CONTEXT, *CORE_TRANSITIONS, *REGIME_CONTEXT, *COHORT_CONTEXT)
BASELINE_FEATURES = (
    "raw_score",
    "score_base_alpha",
    "score_residual_expected_ev",
    "direct_q25_return",
    "pred_peak_mfe_12h_atr__p_hit",
    "pred_peak_mfe_12h_atr__conditional_mean",
    "pred_peak_mfe_12h_atr__expected",
    "pred_future_slope_atr_per_hour__diagnostic",
    *APPROVED_CONTEXT,
)
OPTIONAL_RISK_FEATURES = (
    "pred_mae_before_meaningful_mfe_atr__p_hit",
    "pred_mae_before_meaningful_mfe_atr__if_hit",
    "pred_mae_before_meaningful_mfe_atr__if_no_hit",
    "pred_mae_before_meaningful_mfe_atr__expected",
)
TARGETS = (
    "target_net_positive",
    "target_favorable_net",
    "target_adverse_loss",
    "target_raw_conversion_residual",
    "target_pooled_conversion_residual",
    "target_stop_exit",
    "target_timeout_exit",
)
EVALUATION_ONLY = (
    "causal_pooled_21d",
    "causal_pooled_side_21d",
    "frozen_march_isotonic",
    "execution_gross_ev_12h",
    "execution_cost_return",
    "execution_net_ev_12h",
    "execution_exit_reason",
    "execution_exit_hour",
    "execution_mfe_return_12h",
    "execution_mae_return_12h",
    *TARGETS,
)
FORBIDDEN_FEATURE_TOKENS = (
    "execution_",
    "target_",
    "label",
    "realized",
    "exit_",
    "time_to_first",
    "bars_before",
    "target_price",
    "wait_action",
    "mapped_score",
    "map_reference",
)


class MaterializationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def verify_seal(root: Path) -> dict[str, Any]:
    manifest = root / "manifest.json"
    seal = root / "manifest.sha256"
    if not manifest.is_file() or not seal.is_file():
        raise MaterializationError(f"missing seal under {root}")
    if seal.read_text().split()[0] != sha256(manifest):
        raise MaterializationError(f"manifest seal mismatch under {root}")
    return json.loads(manifest.read_text())


def _utc(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    return result


def _unique(frame: pd.DataFrame, name: str) -> None:
    if frame.duplicated(list(IDENTITY)).any():
        raise MaterializationError(f"{name} contains duplicate candidate-side keys")


def _join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    name: str,
    columns: Sequence[str],
    timestamp_column: str = "__ts__",
) -> pd.DataFrame:
    _unique(right, name)
    selected = right.loc[:, [*IDENTITY, timestamp_column, *columns]].copy()
    selected = selected.rename(columns={timestamp_column: f"__{name}_ts__"})
    joined = left.merge(selected, on=list(IDENTITY), how="left", validate="one_to_one")
    if joined[f"__{name}_ts__"].isna().any():
        raise MaterializationError(f"{name} does not cover the exact v5 identities")
    if not joined["__ts__"].eq(joined[f"__{name}_ts__"]).all():
        raise MaterializationError(f"{name} timestamp parity failed")
    return joined.drop(columns=f"__{name}_ts__")


def context_feature_contract() -> dict[str, Any]:
    for feature in (*BASELINE_FEATURES, *OPTIONAL_RISK_FEATURES):
        lowered = feature.lower()
        if feature in APPROVED_CONTEXT:
            continue
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            raise MaterializationError(f"forbidden model feature: {feature}")
    return {
        "baseline_model_features": list(BASELINE_FEATURES),
        "optional_adverse_risk_ablation_only": list(OPTIONAL_RISK_FEATURES),
        "target_only_never_features": list(TARGETS),
        "evaluation_only_never_features": list(EVALUATION_ONLY),
        "explicitly_excluded": [
            "realised auxiliary labels and paths",
            "time-to-MFE predictions",
            "bars-before-price-stops-decreasing predictions",
            "timing, MAE, target-price and wait actions",
            "causal map reference counts, cutoffs and mapped coordinates",
            "eligible-universe cardinality as a crowding proxy",
        ],
    }


def add_execution_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Create exact-policy targets from canonical flags, never name heuristics."""

    result = panel.copy()
    result["target_net_positive"] = result.execution_net_ev_12h.gt(0).astype(np.int8)
    result["target_favorable_net"] = result.execution_net_ev_12h.clip(lower=0)
    result["target_adverse_loss"] = (-result.execution_net_ev_12h).clip(lower=0)
    result["target_raw_conversion_residual"] = (
        result.execution_net_ev_12h - result.raw_score
    )
    result["target_pooled_conversion_residual"] = (
        result.execution_net_ev_12h - result.get("causal_pooled_21d", np.nan)
    )
    reason = result.execution_exit_reason.astype(str)
    full_stop = result.exit_is_full_stop.astype(bool)
    timeout = result.exit_is_timeout.astype(bool)
    if not full_stop.eq(reason.eq("full_sl")).all():
        raise MaterializationError("canonical full-stop flag/reason mismatch")
    if not timeout.eq(reason.eq("timeout")).all():
        raise MaterializationError("canonical timeout flag/reason mismatch")
    result["target_stop_exit"] = full_stop.astype(np.int8)
    result["target_timeout_exit"] = timeout.astype(np.int8)
    return result


def load_v5(v5: Path, extension: Path) -> pd.DataFrame:
    v5_manifest = verify_seal(v5)
    if v5_manifest.get("schema") != "short_winner_causal_recent_ev_mapping_v5":
        raise MaterializationError("wrong v5 mapping source")
    extension_manifest = verify_seal(extension)
    if extension_manifest.get("schema") != "v5_early_short_oof_extension_v1":
        raise MaterializationError("wrong early-March extension source")
    extension_ledger = extension / "march_extended_oof_score_ledger.parquet"
    if (
        extension_manifest.get("outputs_sha256", {}).get(extension_ledger.name)
        != sha256(extension_ledger)
    ):
        raise MaterializationError("early-March extension ledger hash mismatch")
    march = pd.read_parquet(extension_ledger)
    april = pd.read_parquet(v5 / "april_frozen_forward_score_ledger_and_maps.parquet")
    if len(march) != EXPECTED_MARCH_ROWS:
        raise MaterializationError(f"extended March row count drift: {len(march)}")
    if len(april) != EXPECTED_APRIL_ROWS:
        raise MaterializationError(f"sealed April row count drift: {len(april)}")
    if not march.candidate_score_is_oof.astype(bool).all():
        raise MaterializationError("extended March includes a non-OOF candidate score")
    if march.groupby("side_name").size().to_dict() != {
        "long": EXPECTED_MARCH_ROWS // 2,
        "short": EXPECTED_MARCH_ROWS // 2,
    }:
        raise MaterializationError("extended March side balance drift")
    parts = [march, april]
    panel = pd.concat(parts, ignore_index=True, sort=False)
    panel = _utc(panel, ["__ts__", "execution_decision_utc", "execution_label_end_utc"])
    _unique(panel, "v5")
    if len(panel) != EXPECTED_ROWS:
        raise MaterializationError(f"v5 row count drift: {len(panel)}")
    if not (
        panel.execution_gross_ev_12h - panel.execution_cost_return
    ).sub(panel.execution_net_ev_12h).abs().le(1e-12).all():
        raise MaterializationError("v5 gross-cost-net parity failed")
    return panel


def add_score_deciles(panel: pd.DataFrame, base_root: Path) -> pd.DataFrame:
    base_oof = pd.read_parquet(
        base_root / "oof_predictions.parquet",
        columns=["candidate_id", "side_name", "__symbol__", "__ts__", "base_oof_score"],
    )
    base_oof = _utc(base_oof, ["__ts__"])
    base_oof = add_frozen_causal_score_deciles(base_oof)
    selected = base_oof.loc[
        :,
        [
            *IDENTITY,
            "__ts__",
            "base_oof_score",
            "frozen_base_score_decile",
            "frozen_base_score_decile_group_rows",
        ],
    ]
    joined = _join(
        panel,
        selected,
        name="canonical_base",
        columns=[
            "base_oof_score",
            "frozen_base_score_decile",
            "frozen_base_score_decile_group_rows",
        ],
    )
    if not np.array_equal(
        joined.score_base_alpha.to_numpy(float), joined.base_oof_score.to_numpy(float)
    ):
        raise MaterializationError("v5/base OOF score parity failed")
    return joined


def add_context(panel: pd.DataFrame, context_root: Path) -> pd.DataFrame:
    verify_seal(context_root)
    context = pd.read_parquet(context_root / "cohort_transition_context.parquet")
    context = _utc(context, ["cohort_anchor_utc"])
    keys = ["cohort_anchor_utc", "side_name", "frozen_base_score_decile"]
    if context.duplicated(keys).any():
        raise MaterializationError("context cohort keys are not unique")
    missing = set(APPROVED_CONTEXT).difference(context.columns)
    if missing:
        raise MaterializationError(f"context lacks approved fields: {sorted(missing)}")
    joined = panel.merge(
        context.loc[:, [*keys, "anchor_candidate_support", *APPROVED_CONTEXT]],
        left_on=["__ts__", "side_name", "frozen_base_score_decile"],
        right_on=keys,
        how="left",
        validate="many_to_one",
    )
    if joined.anchor_candidate_support.isna().any():
        raise MaterializationError("context does not cover the exact v5 cohorts")
    if joined.loc[:, APPROVED_CONTEXT].isna().any().any():
        raise MaterializationError("approved context contains a missing value")
    if not joined.__ts__.eq(joined.cohort_anchor_utc).all():
        raise MaterializationError("context anchor timestamp parity failed")
    return joined.drop(columns="cohort_anchor_utc")


def materialize(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = load_v5(args.v5, args.extension)
    panel = add_score_deciles(panel, args.base)
    residual = pd.read_parquet(
        args.residual / "oof_predictions.parquet",
        columns=[
            "candidate_id",
            "side_name",
            "__ts__",
            "base_oof_score",
            "residual_expected_ev",
            "residual_is_oof",
            "residual_fold",
        ],
    )
    residual = _utc(residual, ["__ts__"])
    panel = _join(
        panel,
        residual,
        name="canonical_residual",
        columns=["residual_expected_ev", "residual_is_oof", "residual_fold"],
    )
    if not panel.residual_is_oof.astype(bool).all():
        raise MaterializationError("v5 population includes a non-OOF residual score")
    if not np.array_equal(
        panel.score_residual_expected_ev.to_numpy(float),
        panel.residual_expected_ev.to_numpy(float),
    ):
        raise MaterializationError("v5/residual score parity failed")
    labels = pd.read_parquet(
        args.labels / "labels.parquet",
        columns=[
            "candidate_id",
            "side_name",
            "__ts__",
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_gross_ev_12h",
            "execution_cost_return",
            "execution_net_ev_12h",
            "execution_exit_reason",
            "execution_exit_hour",
            "execution_mfe_return_12h",
            "execution_mae_return_12h",
            "execution_expected_spread_bps",
            "execution_geometry_key",
            "execution_geometry_source",
        ],
    )
    labels = _utc(labels, ["__ts__", "execution_decision_utc", "execution_label_end_utc"])
    label_columns = [
        column for column in labels.columns if column not in {*IDENTITY, "__ts__"}
    ]
    rename_map = {
        column: f"{column}__exact_source"
        for column in label_columns
        if column in panel.columns
    }
    labels = labels.rename(
        columns=rename_map
    )
    panel = _join(
        panel,
        labels,
        name="exact_labels",
        timestamp_column="__ts__",
        columns=[rename_map.get(column, column) for column in label_columns],
    )
    if not panel.execution_decision_utc.eq(panel.execution_decision_utc__exact_source).all():
        raise MaterializationError("exact-label decision timestamp parity failed")
    if not panel.execution_label_end_utc.eq(panel.execution_label_end_utc__exact_source).all():
        raise MaterializationError("exact-label end timestamp parity failed")
    for column in ("execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h"):
        if not np.allclose(
            panel[column], panel[f"{column}__exact_source"], atol=1e-12, rtol=0
        ):
            raise MaterializationError(f"exact-label {column} parity failed")
    drop_exact = [
        column
        for column in panel.columns
        if column.endswith("__exact_source")
    ]
    panel = panel.drop(columns=drop_exact)
    peak = _utc(
        pd.read_parquet(
            args.peak / "oof_predictions.parquet",
            columns=[
                "candidate_id",
                "side_name",
                "__ts__",
                "__decision_ts__",
                "__label_end_ts__",
                "pred_peak_mfe_12h_atr__p_hit",
                "pred_peak_mfe_12h_atr__conditional_mean",
            ],
        ),
        ["__ts__", "__decision_ts__", "__label_end_ts__"],
    )
    peak_prediction_columns = [
        "pred_peak_mfe_12h_atr__p_hit",
        "pred_peak_mfe_12h_atr__conditional_mean",
    ]
    peak = peak.rename(
        columns={
            column: f"{column}__strict_source" for column in peak_prediction_columns
        }
    )
    panel = _join(
        panel,
        peak,
        name="peak_oof",
        columns=[
            "__decision_ts__",
            "__label_end_ts__",
            *[f"{column}__strict_source" for column in peak_prediction_columns],
        ],
    )
    for column in peak_prediction_columns:
        if not np.array_equal(
            panel[column].to_numpy(float),
            panel[f"{column}__strict_source"].to_numpy(float),
        ):
            raise MaterializationError(f"strict peak OOF parity failed: {column}")
    panel = panel.drop(
        columns=[f"{column}__strict_source" for column in peak_prediction_columns]
    )
    slope = _utc(
        pd.read_parquet(
            args.slope / "oof_predictions.parquet",
            columns=[
                "candidate_id",
                "side_name",
                "__ts__",
                "pred_future_slope_atr_per_hour__diagnostic",
            ],
        ),
        ["__ts__"],
    )
    slope = slope.rename(
        columns={
            "pred_future_slope_atr_per_hour__diagnostic":
            "pred_future_slope_atr_per_hour__diagnostic__strict_source"
        }
    )
    panel = _join(
        panel,
        slope,
        name="slope_oof",
        columns=["pred_future_slope_atr_per_hour__diagnostic__strict_source"],
    )
    if not np.array_equal(
        panel.pred_future_slope_atr_per_hour__diagnostic.to_numpy(float),
        panel.pred_future_slope_atr_per_hour__diagnostic__strict_source.to_numpy(float),
    ):
        raise MaterializationError("strict fixed-slope OOF parity failed")
    panel = panel.drop(
        columns="pred_future_slope_atr_per_hour__diagnostic__strict_source"
    )
    mae = _utc(
        pd.read_parquet(
            args.mae / "oof_predictions.parquet",
            columns=[
                "candidate_id",
                "side_name",
                "__ts__",
                "pred_mae_before_meaningful_mfe_atr__p_hit",
                "pred_mae_before_meaningful_mfe_atr__if_hit",
                "pred_mae_before_meaningful_mfe_atr__if_no_hit",
            ],
        ),
        ["__ts__"],
    )
    mae_prediction_columns = list(OPTIONAL_RISK_FEATURES[:-1])
    mae = mae.rename(
        columns={
            column: f"{column}__strict_source" for column in mae_prediction_columns
        }
    )
    panel = _join(
        panel,
        mae,
        name="mae_oof",
        columns=[f"{column}__strict_source" for column in mae_prediction_columns],
    )
    for column in mae_prediction_columns:
        if not np.array_equal(
            panel[column].to_numpy(float),
            panel[f"{column}__strict_source"].to_numpy(float),
        ):
            raise MaterializationError(f"strict MAE OOF parity failed: {column}")
    panel = panel.drop(
        columns=[f"{column}__strict_source" for column in mae_prediction_columns]
    )
    if not panel.__decision_ts__.eq(panel.execution_decision_utc).all():
        raise MaterializationError("auxiliary decision timestamp parity failed")
    if not panel.__label_end_ts__.eq(panel.execution_label_end_utc).all():
        raise MaterializationError("auxiliary label-end timestamp parity failed")
    panel = panel.drop(columns=["__decision_ts__", "__label_end_ts__"])
    panel["pred_peak_mfe_12h_atr__expected"] = (
        panel.pred_peak_mfe_12h_atr__p_hit
        * panel.pred_peak_mfe_12h_atr__conditional_mean
    )
    panel["pred_mae_before_meaningful_mfe_atr__expected"] = (
        panel.pred_mae_before_meaningful_mfe_atr__p_hit
        * panel.pred_mae_before_meaningful_mfe_atr__if_hit
        + (1.0 - panel.pred_mae_before_meaningful_mfe_atr__p_hit)
        * panel.pred_mae_before_meaningful_mfe_atr__if_no_hit
    )
    panel = add_context(panel, args.context)
    panel = add_execution_targets(panel)
    approved = [*BASELINE_FEATURES, *OPTIONAL_RISK_FEATURES]
    if panel.loc[:, approved].isna().any().any():
        raise MaterializationError("approved model inputs contain missing values")
    if not np.isfinite(panel.loc[:, approved].to_numpy(float)).all():
        raise MaterializationError("approved model inputs contain non-finite values")
    if len(panel) != EXPECTED_ROWS or panel.duplicated(list(IDENTITY)).any():
        raise MaterializationError("final panel identity contract failed")
    march_mask = panel.execution_decision_utc.lt(pd.Timestamp("2025-04-01T00:00:00Z"))
    panel["model_development_eligible"] = (
        march_mask & panel.candidate_score_is_oof.fillna(False).astype(bool)
    )
    panel["precalibration_training_history"] = (
        panel.model_development_eligible
        & panel.execution_decision_utc.lt(pd.Timestamp("2025-03-20T00:00:00Z"))
    )
    panel["mapping_calibration_eligible"] = (
        panel.model_development_eligible
        & panel.execution_decision_utc.ge(pd.Timestamp("2025-03-20T00:00:00Z"))
        & panel.execution_decision_utc.lt(pd.Timestamp("2025-03-23T00:00:00Z"))
    )
    panel["selection_evaluation_eligible"] = (
        panel.model_development_eligible
        & panel.execution_decision_utc.ge(pd.Timestamp("2025-03-23T00:00:00Z"))
    )
    panel["forward_diagnostic_only"] = panel.ledger_stage.eq("april_frozen_forward")
    if int(panel.model_development_eligible.sum()) != EXPECTED_MARCH_ROWS:
        raise MaterializationError("extended March development eligibility drift")
    if int(panel.mapping_calibration_eligible.sum()) != 6_912:
        raise MaterializationError("March calibration population drift")
    if int(panel.selection_evaluation_eligible.sum()) != 18_432:
        raise MaterializationError("March selection population drift")
    roles = context_feature_contract()
    return panel, roles


def period_readiness(panel: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "period": "2025-03",
            "status": "READY_DEVELOPMENT_OOF_ONLY",
            "rows": int(panel.model_development_eligible.sum()),
            "reason": "exact v5 candidate score plus strict upstream OOF scores, auxiliaries, labels and context",
        },
        {
            "period": "2025-04",
            "status": "READY_FORWARD_REDIAGNOSTIC_NOT_UNTOUCHED",
            "rows": int(panel.forward_diagnostic_only.sum()),
            "reason": "frozen forward scores but April has already been inspected repeatedly",
        },
        {
            "period": "2025-02",
            "status": "PARTIAL_NOT_JOINABLE_TO_V5_CANDIDATE_HEAD",
            "rows": 0,
            "reason": "base and exact labels exist; residual is passthrough warm-up/non-OOF and v5 candidate/auxiliary scores are absent",
        },
        {
            "period": "2025-01",
            "status": "MISSING_FAIL_CLOSED",
            "rows": 0,
            "reason": "no canonical 31/8 score stream and no compatible current-spread deployed-policy exact12h join",
        },
        {
            "period": "broader_history",
            "status": "FORBIDDEN_BRIDGE",
            "rows": 0,
            "reason": "historical_base_soft_oof/old55/hourly-no-spread sources are incompatible",
        },
    ]
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    panel, roles = materialize(args)
    readiness = period_readiness(panel)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    try:
        panel.to_parquet(stage / "panel.parquet", index=False, compression="zstd")
        readiness.to_csv(stage / "period_readiness.csv", index=False)
        write_json(stage / "feature_roles.json", roles)
        input_paths = {
            "v5_manifest": args.v5 / "manifest.json",
            "extended_march_manifest": args.extension / "manifest.json",
            "extended_march": args.extension / "march_extended_oof_score_ledger.parquet",
            "v5_april": args.v5 / "april_frozen_forward_score_ledger_and_maps.parquet",
            "canonical_base": args.base / "oof_predictions.parquet",
            "canonical_residual": args.residual / "oof_predictions.parquet",
            "exact_labels": args.labels / "labels.parquet",
            "peak_oof": args.peak / "oof_predictions.parquet",
            "slope_oof": args.slope / "oof_predictions.parquet",
            "mae_oof": args.mae / "oof_predictions.parquet",
            "context_manifest": args.context / "manifest.json",
            "context": args.context / "cohort_transition_context.parquet",
        }
        outputs = {
            path.name: sha256(path) for path in stage.iterdir() if path.is_file()
        }
        manifest = {
            "schema": "v5_conversion_residual_input_v3",
            "run_id": args.output_dir.name,
            "status": "SEALED_RESEARCH_INPUT_READY_MARCH_DEVELOPMENT_APRIL_REDIAGNOSTIC",
            "promotion_eligible": False,
            "rows": len(panel),
            "columns": len(panel.columns),
            "period_rows": {
                str(month): int(rows)
                for month, rows in panel.groupby(panel.__ts__.dt.strftime("%Y-%m")).size().items()
            },
            "side_rows": {
                str(side): int(rows) for side, rows in panel.groupby("side_name").size().items()
            },
            "join_contract": "candidate_id + side_name; __ts__ UTC equality assertion; raw symbol deliberately not a join key",
            "label_contract": "exact deployed-exit decision+12h gross - one explicit current-spread cost = net",
            "feature_contract": {
                "baseline": list(BASELINE_FEATURES),
                "optional_adverse_risk_ablation": list(OPTIONAL_RISK_FEATURES),
                "roles_file": "feature_roles.json",
                "live_status": "research only until the aggregate decile-context transform is added to strict inference",
            },
            "validation": {
                "March": (
                    "41,472 candidate-head OOF rows; March 13-19 provides genuine "
                    "older training history, March 20-22 is config-specific mapping "
                    "calibration OOF, and March 23-31 is selection evaluation"
                ),
                "April": "frozen forward rediagnostic, not untouched final test",
                "February": "excluded: residual warm-up is non-OOF and candidate-head/auxiliary scores absent",
                "January": "fail closed; prohibited historical score/cost bridges not used",
            },
            "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
            "outputs_sha256": outputs,
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        }
        write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(
            sha256(stage / "manifest.json") + "  manifest.json\n"
        )
        os.replace(stage, args.output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--v5", type=Path, default=V5)
    command.add_argument("--extension", type=Path, default=EXTENSION)
    command.add_argument("--base", type=Path, default=BASE)
    command.add_argument("--residual", type=Path, default=RESIDUAL)
    command.add_argument("--labels", type=Path, default=LABELS)
    command.add_argument("--peak", type=Path, default=PEAK)
    command.add_argument("--slope", type=Path, default=SLOPE)
    command.add_argument("--mae", type=Path, default=MAE)
    command.add_argument("--context", type=Path, default=CONTEXT)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
