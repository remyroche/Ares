#!/usr/bin/env python3
"""Materialize trailing-profit labels with side x archetype-specific geometry."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.side_aware import add_side_contract_columns  # noqa: E402
from scripts.materialize_first_touch_capture_labels import (  # noqa: E402
    OUT_SL,
    OUT_TO,
    OUT_TP,
    _infer_side,
    _monthly_stats,
    _read_manifest,
    _safe_quantile,
    _safe_rate,
    _source_copy_columns,
)
from scripts.run_label_first_touch_capture_proxy import (  # noqa: E402
    _add_regime_family_columns,
    _fetch_policy_paths,
    _first_touch_capture_outcome,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _json_safe,
    _load_feature_store_columns,
    _read_feature_list,
    _sigmoid,  # noqa: E402
)
from scripts.run_label_widestop_capture_proxy import CaptureArm  # noqa: E402
from extreme_price_movements.timestamp_contract import (  # noqa: E402
    assert_first_path_timestamp,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _policy_path_finite_mask,
)

DEFAULT_SOURCE_LABELS_DIR = Path(
    "data_perp/artifacts/"
    "20260702_184500_single_head_monthly_walkforward_bidirectional_sideaware_policy_net_economic_target_labels/"
    "labels"
)
DEFAULT_OUTPUT_RUN_ID = (
    "20260705_s59_full_long_family_conditioned_trailing_cost100bps_labels"
)
DEFAULT_REGIME_FAMILY_MIN_SCORE = 0.55
DEFAULT_REGIME_FAMILY_MIN_SCORE_GAP = 0.03


def _safe_numeric_series(values: Any, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").reindex(index)
    if values is None:
        return pd.Series(np.nan, index=index, dtype=np.float32)
    return pd.to_numeric(pd.Series(values, index=index), errors="coerce")


def _add_long_path_label_columns(
    out: pd.DataFrame, capture: pd.DataFrame, side: str
) -> pd.DataFrame:
    """Add explicit long-side path-quality labels derived from label outcomes only.

    These are not model features.  They make the long-side failure mode visible
    to base/meta training and reporting: full-path MAE, speed to profit,
    post-MFE drawdown proxy, and trailing activation/exit behavior.
    """

    index = out.index
    is_long = str(side).strip().lower() == "long"
    capture_net = _safe_numeric_series(capture.get("capture_net"), index)
    gross = _safe_numeric_series(capture.get("capture_gross"), index)
    if gross.isna().all():
        gross = capture_net + _safe_numeric_series(
            capture.get("round_trip_cost"), index
        ).fillna(0.0)
    executable_margin = _safe_numeric_series(capture.get("executable_margin"), index)
    if executable_margin.isna().all():
        executable_margin = gross - _safe_numeric_series(
            capture.get("executable_cost_floor"), index
        ).fillna(0.0)

    full_mae = _safe_numeric_series(capture.get("full_path_mae_norm"), index)
    first_mae = _safe_numeric_series(capture.get("first_touch_mae_norm"), index)
    first_mfe = _safe_numeric_series(capture.get("first_touch_mfe_norm"), index)
    full_mfe = _safe_numeric_series(capture.get("full_path_mfe_norm"), index)
    bars_to_mfe_075 = _safe_numeric_series(capture.get("bars_to_mfe_075r"), index)
    bars_to_mfe_1 = _safe_numeric_series(capture.get("bars_to_mfe_1r"), index)
    activation_bar = _safe_numeric_series(capture.get("trailing_activation_bar"), index)
    trailing_activated = _safe_numeric_series(
        capture.get("trailing_activated"), index
    ).fillna(0.0)
    hit = _safe_numeric_series(capture.get("capture_hit"), index).fillna(0.0)
    timeout = _safe_numeric_series(capture.get("capture_timeout"), index).fillna(0.0)
    stop = _safe_numeric_series(capture.get("capture_stop"), index).fillna(0.0)
    mae_before_mfe = _safe_numeric_series(
        capture.get("mae_1r_before_mfe_1r"), index
    ).fillna(0.0)
    mfe_before_mae = _safe_numeric_series(
        capture.get("mfe_1r_before_mae_1r"), index
    ).fillna(0.0)
    underwater_bars = _safe_numeric_series(
        capture.get("underwater_bars_before_mfe_1r"), index
    )
    underwater_fraction = _safe_numeric_series(
        capture.get("underwater_fraction_before_mfe_1r"), index
    )

    post_mfe_drawdown = (full_mae - first_mae).clip(lower=0.0)
    time_to_profit = bars_to_mfe_1.where(bars_to_mfe_1.notna(), bars_to_mfe_075)
    trailing_success = (
        trailing_activated.gt(0.5) & hit.gt(0.5) & executable_margin.gt(0.0)
    )
    slow_profit = time_to_profit.gt(16.0) | activation_bar.gt(16.0)
    full_bad = full_mae.ge(1.0)
    post_mfe_bad = post_mfe_drawdown.ge(0.50)
    clean = (
        executable_margin.gt(0.0)
        & full_bad.fillna(False).eq(False)
        & timeout.lt(0.5)
        & stop.lt(0.5)
        & mae_before_mfe.lt(0.5)
        & (mfe_before_mae.gt(0.5) | trailing_activated.gt(0.5))
        & slow_profit.fillna(False).eq(False)
        & post_mfe_bad.fillna(False).eq(False)
    )
    dirty_positive = executable_margin.gt(0.0) & (
        full_bad.fillna(False)
        | timeout.gt(0.5)
        | stop.gt(0.5)
        | mae_before_mfe.gt(0.5)
        | slow_profit.fillna(False)
        | post_mfe_bad.fillna(False)
    )
    quality_soft = (
        0.35 * executable_margin.gt(0.0).astype(float)
        + 0.20 * trailing_success.astype(float)
        + 0.15 * mfe_before_mae.clip(0.0, 1.0)
        + 0.10 * full_mfe.fillna(0.0).clip(0.0, 2.0) / 2.0
        - 0.20 * full_bad.fillna(False).astype(float)
        - 0.12 * post_mfe_bad.fillna(False).astype(float)
        - 0.10 * slow_profit.fillna(False).astype(float)
        - 0.08 * timeout.clip(0.0, 1.0)
        - 0.08 * mae_before_mfe.clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    long_mask = np.full(len(out), bool(is_long), dtype=bool)
    out["__long_path_full_bad_mae_1r__"] = np.where(
        long_mask, full_bad.fillna(False).astype(float), np.nan
    ).astype(np.float32)
    out["__long_path_time_to_profit_bars__"] = np.where(
        long_mask, time_to_profit, np.nan
    ).astype(np.float32)
    out["__long_path_slow_profit__"] = np.where(
        long_mask, slow_profit.fillna(False).astype(float), np.nan
    ).astype(np.float32)
    out["__long_path_post_mfe_drawdown_norm__"] = np.where(
        long_mask, post_mfe_drawdown, np.nan
    ).astype(np.float32)
    out["__long_path_post_mfe_bad_drawdown__"] = np.where(
        long_mask, post_mfe_bad.fillna(False).astype(float), np.nan
    ).astype(np.float32)
    out["__long_trailing_activated__"] = np.where(
        long_mask, trailing_activated, np.nan
    ).astype(np.float32)
    out["__long_trailing_success__"] = np.where(
        long_mask, trailing_success.astype(float), np.nan
    ).astype(np.float32)
    out["__long_path_clean_exec_label__"] = np.where(
        long_mask, clean.astype(float), np.nan
    ).astype(np.float32)
    out["__long_path_dirty_positive_label__"] = np.where(
        long_mask, dirty_positive.astype(float), np.nan
    ).astype(np.float32)
    out["__long_path_quality_soft__"] = np.where(
        long_mask, quality_soft, np.nan
    ).astype(np.float32)
    out["__path_full_bad_mae_1r__"] = full_bad.fillna(False).astype(np.float32)
    out["__path_time_to_profit_bars__"] = time_to_profit.astype(np.float32)
    out["__path_post_mfe_drawdown_norm__"] = post_mfe_drawdown.astype(np.float32)
    out["__path_trailing_success__"] = trailing_success.astype(np.float32)
    return out


def _default_policy_manifest() -> dict[str, Any]:
    """Current evidence-backed geometry policy from the S55/S57 sweeps."""
    return {
        "version": "s59_full_long_family_side_archetype_trailing_v1",
        "created_from": [
            "s55_short_breakout_pathquality_refine_v1",
            "s55_short_mixed_pathquality_refine_v1",
            "s57_long_wide_slow_geometry_ablation_v1",
        ],
        "benchmark_policy": "s58_side_archetype_conditioned_trailing_v1",
        "default": {
            "short": {
                "policy_key": "short_default_clean_path",
                "semantic_role": "short_clean_path_fallback",
                "confidence": "fallback",
                "tp_r": 0.70,
                "sl_r": 1.00,
                "trail_r": 0.25,
                "max_bars_to_mfe": 16.0,
                "max_barrier": 0.05,
            },
            "long": {
                "policy_key": "long_default_wideslow_pathquality",
                "semantic_role": "long_wideslow_path_quality_fallback",
                "confidence": "fallback",
                "tp_r": 0.40,
                "sl_r": 1.00,
                "trail_r": 0.25,
                "max_bars_to_mfe": 24.0,
                "max_barrier": 0.05,
            },
        },
        "overrides": [
            {
                "side": "short",
                "archetype": "breakout_impulse",
                "policy_key": "short_breakout_precision",
                "semantic_role": "short_breakout_precision_opportunity",
                "confidence": "promoted",
                "tp_r": 0.50,
                "sl_r": 0.75,
                "trail_r": 0.30,
                "max_bars_to_mfe": 16.0,
                "max_barrier": 0.05,
            },
            {
                "side": "short",
                "archetype": "mixed",
                "policy_key": "short_mixed_clean_path",
                "semantic_role": "short_mixed_clean_path",
                "confidence": "promoted",
                "tp_r": 0.70,
                "sl_r": 1.00,
                "trail_r": 0.25,
                "max_bars_to_mfe": 16.0,
                "max_barrier": 0.05,
            },
            {
                "side": "long",
                "archetype": "vol_compression",
                "policy_key": "long_volcompression_wideslow_candidate",
                "semantic_role": "long_vol_compression_promotable_candidate",
                "confidence": "promotable_candidate",
                "source_holdout_ev_weighted_precision": 0.648654,
                "source_holdout_precision": 0.363636,
                "source_holdout_mean_net": -0.003808,
                "tp_r": 0.50,
                "sl_r": 1.00,
                "trail_r": 0.20,
                "max_bars_to_mfe": 24.0,
                "max_barrier": 0.05,
            },
            {
                "side": "long",
                "archetype": "mixed",
                "policy_key": "long_mixed_wideslow_tentative",
                "semantic_role": "long_mixed_tentative_candidate",
                "confidence": "tentative_candidate",
                "source_holdout_ev_weighted_precision": 0.656161,
                "source_holdout_precision": 0.321244,
                "source_holdout_mean_net": 0.01193,
                "tp_r": 0.40,
                "sl_r": 1.50,
                "trail_r": 0.20,
                "max_bars_to_mfe": 24.0,
                "max_barrier": 0.05,
            },
            {
                "side": "long",
                "archetype": "breakout_impulse",
                "policy_key": "long_breakout_diagnostic_candidate",
                "semantic_role": "long_breakout_diagnostic_candidate",
                "confidence": "diagnostic_candidate",
                "source_holdout_ev_weighted_precision": 0.567733,
                "source_holdout_precision": 0.472222,
                "source_holdout_mean_net": 0.00362,
                "tp_r": 0.60,
                "sl_r": 1.25,
                "trail_r": 0.15,
                "max_bars_to_mfe": 24.0,
                "max_barrier": 0.05,
            },
            {
                "side": "long",
                "archetype": "dirty_avoid",
                "policy_key": "long_dirtyavoid_sparse_questionable",
                "semantic_role": "long_dirty_avoid_questionable_opportunity",
                "confidence": "questionable_sparse_candidate",
                "source_holdout_ev_weighted_precision": 0.934423,
                "source_holdout_precision": 0.60,
                "source_holdout_mean_net": 0.020142,
                "tp_r": 0.40,
                "sl_r": 1.00,
                "trail_r": 0.25,
                "max_bars_to_mfe": 24.0,
                "max_barrier": 0.05,
            },
        ],
    }


def _load_policy_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _default_policy_manifest()
    return json.loads(path.read_text(encoding="utf-8"))


def _policy_for(
    manifest: dict[str, Any], *, side: str, archetype: str
) -> dict[str, Any]:
    side_key = str(side).strip().lower()
    family = str(archetype).strip()
    for row in manifest.get("overrides", []) or []:
        if (
            str(row.get("side", "")).strip().lower() == side_key
            and str(row.get("archetype", "")).strip() == family
        ):
            return dict(row)
    defaults = manifest.get("default", {}) or {}
    if side_key not in defaults:
        raise KeyError(f"No default side policy for {side_key}")
    row = dict(defaults[side_key])
    row.setdefault("side", side_key)
    row.setdefault("archetype", family)
    return row


def _arm_from_policy(policy: dict[str, Any]) -> CaptureArm:
    return CaptureArm(
        name=str(policy.get("policy_key") or "conditioned_trailing"),
        tp_r=float(policy["tp_r"]),
        sl_r=float(policy["sl_r"]),
        trail_r=float(policy.get("trail_r", 0.50)),
        max_bars_to_mfe=float(policy.get("max_bars_to_mfe", 24.0)),
        max_barrier=float(policy.get("max_barrier", 0.05)),
    )


def _copy_capture_columns(
    *,
    out: pd.DataFrame,
    capture: pd.DataFrame,
    side: str,
    timeframe: str,
    round_trip_cost: float,
    policy_label_center: float,
    policy_label_temperature: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    for col in _source_copy_columns(out):
        out[f"__source{col}"] = out[col].to_numpy(copy=False)

    capture_net = pd.to_numeric(capture["capture_net"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    policy_soft = _sigmoid(
        (
            np.nan_to_num(capture_net, nan=float(policy_label_center))
            - float(policy_label_center)
        )
        / max(float(policy_label_temperature), 1e-12)
    ).astype(np.float32)
    hit = (
        pd.to_numeric(capture["capture_hit"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    stop = (
        pd.to_numeric(capture["capture_stop"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    timeout = (
        pd.to_numeric(capture["capture_timeout"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    eligible = (
        pd.to_numeric(capture["capture_eligible"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    label_code = np.full(len(out), OUT_SL, dtype=np.int8)
    label_code[timeout > 0.5] = OUT_TO
    label_code[hit > 0.5] = OUT_TP

    out["__y_lbl__"] = label_code
    out["__y_outcome__"] = label_code
    out["__y_bin__"] = hit.astype(np.float32)
    out["__y_ret__"] = capture_net
    out["__is_timeout__"] = timeout.astype(np.float32)
    out["__tp__"] = pd.to_numeric(
        capture["effective_tp_abs"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    out["__sl__"] = pd.to_numeric(
        capture["effective_sl_abs"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    out["__u_policy_net__"] = capture_net
    out["__r_policy_net__"] = capture_net

    out["__first_touch_target_soft__"] = pd.to_numeric(
        capture["target_soft"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    out["__first_touch_policy_soft__"] = policy_soft
    out["__first_touch_capture_net__"] = capture_net
    out["__first_touch_round_trip_cost__"] = np.full(
        len(out), float(round_trip_cost), dtype=np.float32
    )
    out["__first_touch_hit__"] = hit.astype(np.float32)
    out["__first_touch_stop__"] = stop.astype(np.float32)
    out["__first_touch_timeout__"] = timeout.astype(np.float32)
    out["__first_touch_eligible__"] = eligible.astype(np.float32)
    out["__first_touch_valid_path__"] = (
        pd.to_numeric(capture["capture_valid_path"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    out["__first_touch_net_positive__"] = (capture_net > 0.0).astype(np.float32)
    out["__first_touch_bar__"] = pd.to_numeric(
        capture["first_touch_bar"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    out["__first_touch_same_bar_both__"] = (
        pd.to_numeric(capture["same_bar_both_hit"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    # Keep the repository's legacy support aliases aligned with the same
    # causal path as the promoted first-touch target.  These names are still
    # accepted by diagnostic and auxiliary-label code, so carrying them from
    # the source artifact would reintroduce the pre-signal-close path.
    barrier = (
        pd.to_numeric(out["__barrier_pct__"], errors="coerce")
        .abs()
        .to_numpy(dtype=np.float32)
    )
    full_mfe_norm = pd.to_numeric(
        capture.get("full_path_mfe_norm"), errors="coerce"
    ).to_numpy(dtype=np.float32)
    full_mae_norm = pd.to_numeric(
        capture.get("full_path_mae_norm"), errors="coerce"
    ).to_numpy(dtype=np.float32)
    corrected_mfe = (full_mfe_norm * barrier).astype(np.float32, copy=False)
    corrected_mae = (full_mae_norm * barrier).astype(np.float32, copy=False)
    out["__mfe__"] = corrected_mfe
    out["__mae__"] = corrected_mae
    out["__mfe_ret__"] = corrected_mfe
    out["__mae_ret__"] = corrected_mae
    out["__bars_to_mfe__"] = pd.to_numeric(
        capture.get("bars_to_mfe_1r"), errors="coerce"
    ).to_numpy(dtype=np.float32)
    out["__bars_to_mae__"] = pd.to_numeric(
        capture.get("bars_to_mae_1r"), errors="coerce"
    ).to_numpy(dtype=np.float32)
    out["__quality__"] = pd.to_numeric(
        capture["target_soft"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    # Current training constructs its explicit target-strength weights from
    # permitted train rows.  A neutral alias is safer than retaining weights
    # fitted to the superseded source target.
    out["__w__"] = np.ones(len(out), dtype=np.float32)
    out["__first_touch_effective_tp_abs__"] = out["__tp__"]
    out["__first_touch_effective_sl_abs__"] = out["__sl__"]
    if "effective_trail_abs" in capture.columns:
        out["__first_touch_effective_trail_abs__"] = pd.to_numeric(
            capture["effective_trail_abs"], errors="coerce"
        ).to_numpy(dtype=np.float32)
    if "trailing_activated" in capture.columns:
        out["__trailing_profit_activated__"] = (
            pd.to_numeric(capture["trailing_activated"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    if "trailing_activation_bar" in capture.columns:
        out["__trailing_profit_activation_bar__"] = pd.to_numeric(
            capture["trailing_activation_bar"], errors="coerce"
        ).to_numpy(dtype=np.float32)

    for source_col, output_col in (
        ("mae_to_sl", "__first_touch_mae_to_sl__"),
        ("mfe_to_tp", "__first_touch_mfe_to_tp__"),
        ("first_touch_mae_norm", "__first_touch_mae_norm__"),
        ("first_touch_mfe_norm", "__first_touch_mfe_norm__"),
        ("full_path_mae_to_sl", "__first_touch_full_path_mae_to_sl__"),
        ("full_path_mfe_to_tp", "__first_touch_full_path_mfe_to_tp__"),
        ("full_path_mae_norm", "__first_touch_full_path_mae_norm__"),
        ("full_path_mfe_norm", "__first_touch_full_path_mfe_norm__"),
    ):
        if source_col in capture.columns:
            out[output_col] = pd.to_numeric(
                capture[source_col], errors="coerce"
            ).to_numpy(dtype=np.float32)

    for col in (
        "bars_to_mfe_05r",
        "bars_to_mfe_075r",
        "bars_to_mfe_1r",
        "bars_to_mfe_125r",
        "bars_to_mfe_15r",
        "bars_to_mae_05r",
        "bars_to_mae_075r",
        "bars_to_mae_1r",
        "bars_to_mae_15r",
        "mfe_1r_before_mae_05r",
        "mfe_1r_before_mae_075r",
        "mfe_1r_before_mae_1r",
        "mae_05r_before_mfe_1r",
        "mae_075r_before_mfe_1r",
        "mae_1r_before_mfe_1r",
        "max_adverse_before_mfe_1r",
        "underwater_bars_before_mfe_1r",
        "underwater_fraction_before_mfe_1r",
        "area_underwater_before_mfe_1r",
    ):
        if col in capture.columns:
            out[f"__{col}__"] = pd.to_numeric(capture[col], errors="coerce").to_numpy(
                dtype=np.float32
            )

    out = add_side_contract_columns(
        out,
        side=side,
        timestamp_col="__ts__",
        asset_col="__symbol__",
        timeframe=timeframe,
        copy=False,
    )
    return out, policy_soft


def _conditioned_capture(
    *,
    df: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    side: str,
    families: pd.Series,
    policy_manifest: dict[str, Any],
    outcome_mode: str,
    round_trip_cost: float,
    first_outcome_bar: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for family in families.fillna("mixed").astype(str).unique():
        policy = _policy_for(policy_manifest, side=side, archetype=str(family))
        key = (
            str(policy.get("policy_key")),
            float(policy["tp_r"]),
            float(policy["sl_r"]),
            float(policy.get("trail_r", 0.50)),
            float(policy.get("max_bars_to_mfe", 24.0)),
            float(policy.get("max_barrier", 0.05)),
        )
        assignments.append({"family": str(family), "policy": policy, "key": key})
        assignment_rows.append(
            {
                "side": str(side),
                "archetype": str(family),
                "policy_key": str(policy.get("policy_key")),
                "semantic_role": str(policy.get("semantic_role", "")),
                "tp_r": float(policy["tp_r"]),
                "sl_r": float(policy["sl_r"]),
                "trail_r": float(policy.get("trail_r", 0.50)),
                "max_bars_to_mfe": float(policy.get("max_bars_to_mfe", 24.0)),
                "max_barrier": float(policy.get("max_barrier", 0.05)),
                "rows": int(families.astype(str).eq(str(family)).sum()),
            }
        )

    policy_by_key = {row["key"]: row["policy"] for row in assignments}
    captures_by_key = {
        key: _first_touch_capture_outcome(
            df,
            paths,
            _arm_from_policy(policy),
            side_name=side,
            outcome_mode=outcome_mode,
            round_trip_cost=float(round_trip_cost),
            first_outcome_bar=int(first_outcome_bar),
        )
        for key, policy in policy_by_key.items()
    }
    capture = pd.DataFrame(
        index=df.index, columns=next(iter(captures_by_key.values())).columns
    )
    policy_key = pd.Series("", index=df.index, dtype=object)
    role = pd.Series("", index=df.index, dtype=object)
    confidence = pd.Series("", index=df.index, dtype=object)
    tp_r = pd.Series(np.nan, index=df.index, dtype=np.float32)
    sl_r = pd.Series(np.nan, index=df.index, dtype=np.float32)
    trail_r = pd.Series(np.nan, index=df.index, dtype=np.float32)
    max_bars = pd.Series(np.nan, index=df.index, dtype=np.float32)
    max_barrier = pd.Series(np.nan, index=df.index, dtype=np.float32)

    fam_s = families.fillna("mixed").astype(str)
    for row in assignments:
        mask = fam_s.eq(str(row["family"]))
        key = row["key"]
        policy = row["policy"]
        capture.loc[mask, :] = captures_by_key[key].loc[mask, :]
        policy_key.loc[mask] = str(policy.get("policy_key"))
        role.loc[mask] = str(policy.get("semantic_role", ""))
        confidence.loc[mask] = str(policy.get("confidence", ""))
        tp_r.loc[mask] = float(policy["tp_r"])
        sl_r.loc[mask] = float(policy["sl_r"])
        trail_r.loc[mask] = float(policy.get("trail_r", 0.50))
        max_bars.loc[mask] = float(policy.get("max_bars_to_mfe", 24.0))
        max_barrier.loc[mask] = float(policy.get("max_barrier", 0.05))

    capture = capture.infer_objects(copy=False)
    capture["__archetype_policy_key__"] = policy_key
    capture["__archetype_policy_role__"] = role
    capture["__archetype_policy_confidence__"] = confidence
    capture["__archetype_policy_tp_r__"] = tp_r.astype(np.float32)
    capture["__archetype_policy_sl_r__"] = sl_r.astype(np.float32)
    capture["__archetype_policy_trail_r__"] = trail_r.astype(np.float32)
    capture["__archetype_policy_max_bars_to_mfe__"] = max_bars.astype(np.float32)
    capture["__archetype_policy_max_barrier__"] = max_barrier.astype(np.float32)
    return capture, pd.DataFrame(assignment_rows)


def _dataset_chunk_windows(
    source_path: Path, chunk_frequency: str
) -> list[tuple[str, str, str]]:
    mode = str(chunk_frequency or "none").strip().lower()
    if mode in {"", "none"}:
        return [("", "", "")]
    if mode != "monthly":
        raise ValueError(f"Unsupported chunk frequency: {chunk_frequency!r}")
    ts = pd.to_datetime(
        pd.read_parquet(source_path, columns=["__ts__"])["__ts__"],
        utc=True,
        errors="coerce",
    )
    ts = ts.dropna()
    if ts.empty:
        raise RuntimeError(
            f"{source_path}: cannot build monthly chunks; __ts__ is empty or invalid"
        )
    start_month = ts.min().to_period("M").to_timestamp(how="start").tz_localize("UTC")
    end_month = (
        (ts.max().to_period("M") + 1).to_timestamp(how="start").tz_localize("UTC")
    )
    starts = pd.date_range(start_month, end_month, freq="MS", inclusive="left")
    out: list[tuple[str, str, str]] = []
    for start in starts:
        end = start + pd.offsets.MonthBegin(1)
        suffix = start.strftime("%Y_%m")
        out.append((suffix, start.isoformat(), end.isoformat()))
    return out


def _chunked_dataset_name(dataset_name: str, suffix: str) -> str:
    return str(dataset_name) if not suffix else f"{dataset_name}_{suffix}"


def _chunked_file_name(file_name: str, suffix: str) -> str:
    if not suffix:
        return str(file_name)
    path = Path(file_name)
    return f"{path.stem}_{suffix}{path.suffix}"


def _materialize_dataset(
    *,
    source_path: Path,
    output_path: Path,
    dataset_name: str,
    side: str,
    data_root: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    market_mode: str,
    exchange: str,
    timeframe: str,
    path_len: int,
    apply_delayed_entry: bool,
    entry_delay_hours: int,
    policy_label_center: float,
    policy_label_temperature: float,
    outcome_mode: str,
    round_trip_cost: float,
    policy_manifest: dict[str, Any],
    regime_family_min_score: float,
    regime_family_min_score_gap: float,
    start_ts: str | None = None,
    end_ts: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    df = pd.read_parquet(source_path).reset_index(drop=True)
    required = {"__ts__", "__symbol__", "__barrier_pct__"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"{source_path}: missing required columns {missing}")
    if start_ts is not None or end_ts is not None:
        ts = pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
        mask = pd.Series(True, index=df.index)
        if start_ts is not None:
            mask &= ts >= pd.Timestamp(start_ts, tz="UTC")
        if end_ts is not None:
            mask &= ts < pd.Timestamp(end_ts, tz="UTC")
        df = df.loc[mask.to_numpy()].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(
                f"{source_path}: no rows after timestamp filter start_ts={start_ts!r} end_ts={end_ts!r}"
            )

    if feature_list_csv.exists():
        selected_features = _read_feature_list(
            feature_list_csv, max_features=max_feature_store_features
        )
        feature_matrix, feature_report = _load_feature_store_columns(
            df,
            feature_dir=feature_dir,
            selected_features=selected_features,
        )
    else:
        selected_features = []
        feature_matrix = pd.DataFrame(index=df.index)
        feature_report = {
            "source": "embedded_pre_entry_columns",
            "reason": "feature_list_csv_missing",
            "feature_list_csv": str(feature_list_csv),
            "embedded_columns": int(len(df.columns)),
        }
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in df.columns]
        if new_cols:
            df = pd.concat(
                [
                    df,
                    feature_matrix.loc[:, new_cols]
                    .reset_index(drop=True)
                    .astype(np.float32, copy=False),
                ],
                axis=1,
                copy=False,
            )

    regime_report = _add_regime_family_columns(
        df,
        min_score=float(regime_family_min_score),
        min_score_gap=float(regime_family_min_score_gap),
        legacy_min_score=float(regime_family_min_score),
    )
    families = df["__regime_family__"].fillna("mixed").astype(str)
    rows_exec, paths, path_stats = _fetch_policy_paths(
        df,
        labels_path=source_path,
        side=side,
        data_root=data_root,
        market_mode=market_mode,
        exchange=exchange,
        path_len=path_len,
        apply_delayed_entry=apply_delayed_entry,
        entry_delay_hours=int(entry_delay_hours),
        timeframe=timeframe,
    )
    minimum_path_coverage = float(
        os.environ.get("EPM_LABEL_MIN_PATH_COVERAGE", "0.95")
    )
    finite_path_coverage = float(path_stats.get("finite_path_coverage", 0.0) or 0.0)
    if finite_path_coverage < minimum_path_coverage:
        raise RuntimeError(
            f"{source_path}: causal execution path coverage "
            f"{finite_path_coverage:.4%} is below required "
            f"{minimum_path_coverage:.4%}; labels were not written"
        )
    finite_path_mask = _policy_path_finite_mask(paths)
    dropped_unresolved_paths = int((~finite_path_mask).sum())
    if dropped_unresolved_paths:
        # Rows without a complete causal path do not have an observable label.
        # Keep the shard-level coverage audit above, then exclude those rows
        # instead of persisting NaT path provenance or synthetic outcomes.
        keep = np.flatnonzero(finite_path_mask)
        df = df.iloc[keep].reset_index(drop=True)
        rows_exec = rows_exec.iloc[keep].reset_index(drop=True)
        families = families.iloc[keep].reset_index(drop=True)
        paths = tuple(
            np.asarray(path)[keep]
            for path in paths
        )
        path_stats["dropped_unresolved_path_rows"] = dropped_unresolved_paths
        path_stats["materialized_rows"] = int(len(df))
    capture, assignment = _conditioned_capture(
        df=df,
        paths=paths,
        side=side,
        families=families,
        policy_manifest=policy_manifest,
        outcome_mode=outcome_mode,
        round_trip_cost=float(round_trip_cost),
        first_outcome_bar=1 if apply_delayed_entry else 0,
    )

    out = df.copy()
    out["__archetype_label_family__"] = families.to_numpy(dtype=object, copy=False)
    out["__archetype_label_source__"] = "observable_regime_family"
    for col in (
        "__archetype_policy_key__",
        "__archetype_policy_role__",
        "__archetype_policy_confidence__",
        "__archetype_policy_tp_r__",
        "__archetype_policy_sl_r__",
        "__archetype_policy_trail_r__",
        "__archetype_policy_max_bars_to_mfe__",
        "__archetype_policy_max_barrier__",
    ):
        out[col] = capture[col].to_numpy(copy=False)

    out, policy_soft = _copy_capture_columns(
        out=out,
        capture=capture,
        side=side,
        timeframe=timeframe,
        round_trip_cost=float(round_trip_cost),
        policy_label_center=float(policy_label_center),
        policy_label_temperature=float(policy_label_temperature),
    )
    out["__signal_ts__"] = pd.to_datetime(df["__ts__"], utc=True, errors="coerce")
    out["__decision_ts__"] = pd.to_datetime(
        rows_exec["decision_ts"], utc=True, errors="coerce"
    )
    if "delayed_entry_effective_ts" in rows_exec.columns:
        effective_entry = pd.to_datetime(
            rows_exec["delayed_entry_effective_ts"], utc=True, errors="coerce"
        )
    else:
        effective_entry = pd.Series(
            pd.NaT, index=rows_exec.index, dtype="datetime64[ns, UTC]"
        )
    out["__entry_ts__"] = effective_entry.fillna(out["__decision_ts__"])
    out["__first_path_ts__"] = pd.to_datetime(
        rows_exec.get("first_path_timestamp", out["__entry_ts__"]),
        utc=True,
        errors="coerce",
    )
    assert_first_path_timestamp(
        first_path_ts=out["__first_path_ts__"],
        signal_ts=out["__signal_ts__"],
        timeframe=timeframe,
    )
    resolved_path = out["__first_path_ts__"].notna() & out["__entry_ts__"].notna()
    if bool((out.loc[resolved_path, "__first_path_ts__"] < out.loc[resolved_path, "__entry_ts__"]).any()):
        raise AssertionError(
            "first_path_timestamp must be at or after the executable entry timestamp"
        )
    out = _add_long_path_label_columns(out, capture, side)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    capture_net = pd.to_numeric(capture["capture_net"], errors="coerce").to_numpy(
        dtype=np.float32
    )
    finite = np.isfinite(capture_net)
    hit = pd.to_numeric(capture["capture_hit"], errors="coerce").fillna(0.0)
    stop = pd.to_numeric(capture["capture_stop"], errors="coerce").fillna(0.0)
    timeout = pd.to_numeric(capture["capture_timeout"], errors="coerce").fillna(0.0)
    eligible = pd.to_numeric(capture["capture_eligible"], errors="coerce").fillna(0.0)
    bad_mae = pd.to_numeric(capture["mae_to_sl"], errors="coerce").ge(1.0)
    full_bad_mae = _safe_numeric_series(
        capture.get("full_path_mae_norm"), capture.index
    ).ge(1.0)
    long_clean = pd.to_numeric(
        out.get("__long_path_clean_exec_label__"), errors="coerce"
    )
    long_dirty = pd.to_numeric(
        out.get("__long_path_dirty_positive_label__"), errors="coerce"
    )
    long_post_mfe = pd.to_numeric(
        out.get("__long_path_post_mfe_drawdown_norm__"), errors="coerce"
    )
    long_time_to_profit = pd.to_numeric(
        out.get("__long_path_time_to_profit_bars__"), errors="coerce"
    )
    side_family_rows: list[dict[str, Any]] = []
    for (fam, pkey), idx in out.groupby(
        ["__archetype_label_family__", "__archetype_policy_key__"]
    ).indices.items():
        pos = np.asarray(idx, dtype=np.int64)
        net = pd.Series(capture_net[pos])
        side_family_rows.append(
            {
                "dataset": str(dataset_name),
                "side": str(side),
                "archetype": str(fam),
                "policy_key": str(pkey),
                "rows": int(len(pos)),
                "symbols": int(out.iloc[pos]["__symbol__"].nunique(dropna=True)),
                "capture_net_mean": float(net.mean())
                if len(net.dropna())
                else float("nan"),
                "capture_net_q10": float(net.quantile(0.10))
                if len(net.dropna())
                else float("nan"),
                "hit_rate": _safe_rate(hit.iloc[pos]),
                "stop_rate": _safe_rate(stop.iloc[pos]),
                "timeout_rate": _safe_rate(timeout.iloc[pos]),
                "bad_mae_to_sl_rate": _safe_rate(bad_mae.iloc[pos]),
                "full_path_bad_mae_1r_rate": _safe_rate(full_bad_mae.iloc[pos]),
                "long_path_clean_exec_rate": _safe_rate(long_clean.iloc[pos])
                if str(side) == "long"
                else float("nan"),
                "long_path_dirty_positive_rate": _safe_rate(long_dirty.iloc[pos])
                if str(side) == "long"
                else float("nan"),
                "long_path_mean_post_mfe_drawdown_norm": float(
                    long_post_mfe.iloc[pos].mean()
                )
                if str(side) == "long" and long_post_mfe.iloc[pos].notna().any()
                else float("nan"),
                "long_path_mean_time_to_profit_bars": float(
                    long_time_to_profit.iloc[pos].mean()
                )
                if str(side) == "long" and long_time_to_profit.iloc[pos].notna().any()
                else float("nan"),
                "eligible_rate": _safe_rate(eligible.iloc[pos]),
                "net_positive_rate": _safe_rate(net > 0.0),
                "policy_soft_mean": float(np.mean(policy_soft[pos]))
                if len(pos)
                else float("nan"),
            }
        )

    summary = {
        "dataset": str(dataset_name),
        "source_file": str(source_path),
        "output_file": str(output_path),
        "side": str(side),
        "start_ts": str(start_ts) if start_ts is not None else None,
        "end_ts": str(end_ts) if end_ts is not None else None,
        "min_ts": str(pd.to_datetime(out["__ts__"], errors="coerce").min()),
        "max_ts": str(pd.to_datetime(out["__ts__"], errors="coerce").max()),
        "columns": list(out.columns),
        "rows": int(len(out)),
        "finite": int(np.sum(finite)),
        "finite_frac": float(np.mean(finite)) if len(finite) else 0.0,
        "capture_net_mean": float(np.nanmean(capture_net))
        if np.any(finite)
        else float("nan"),
        "capture_net_std": float(np.nanstd(capture_net))
        if np.any(finite)
        else float("nan"),
        "capture_net_p10": float(np.nanpercentile(capture_net, 10))
        if np.any(finite)
        else float("nan"),
        "capture_net_p90": float(np.nanpercentile(capture_net, 90))
        if np.any(finite)
        else float("nan"),
        "hit_rate": _safe_rate(hit),
        "stop_rate": _safe_rate(stop),
        "timeout_rate": _safe_rate(timeout),
        "bad_mae_to_sl_rate": _safe_rate(bad_mae),
        "full_path_bad_mae_1r_rate": _safe_rate(full_bad_mae),
        "long_path_clean_exec_rate": _safe_rate(long_clean)
        if str(side) == "long"
        else float("nan"),
        "long_path_dirty_positive_rate": _safe_rate(long_dirty)
        if str(side) == "long"
        else float("nan"),
        "long_path_mean_post_mfe_drawdown_norm": float(long_post_mfe.mean())
        if str(side) == "long" and long_post_mfe.notna().any()
        else float("nan"),
        "long_path_mean_time_to_profit_bars": float(long_time_to_profit.mean())
        if str(side) == "long" and long_time_to_profit.notna().any()
        else float("nan"),
        "eligible_rate": _safe_rate(eligible),
        "net_positive_rate": _safe_rate(capture_net > 0.0),
        "policy_soft_mean": float(np.mean(policy_soft))
        if len(policy_soft)
        else float("nan"),
        "policy_soft_std": float(np.std(policy_soft))
        if len(policy_soft)
        else float("nan"),
        "effective_tp_abs_p90": _safe_quantile(
            out["__first_touch_effective_tp_abs__"], 0.90
        ),
        "effective_sl_abs_p90": _safe_quantile(
            out["__first_touch_effective_sl_abs__"], 0.90
        ),
        "effective_trail_abs_p90": _safe_quantile(
            out.get("__first_touch_effective_trail_abs__"), 0.90
        )
        if "__first_touch_effective_trail_abs__" in out.columns
        else float("nan"),
        "trailing_activated_rate": _safe_rate(out["__trailing_profit_activated__"])
        if "__trailing_profit_activated__" in out.columns
        else float("nan"),
        "outcome_mode": str(outcome_mode),
        "round_trip_cost": float(round_trip_cost),
        "path_fetch": path_stats,
        "feature_store": feature_report,
        "regime_report": regime_report,
        "policy_assignments": assignment.to_dict(orient="records"),
        "monthly": _monthly_stats(out, capture, policy_soft),
        "side_archetype": side_family_rows,
    }
    return summary, pd.DataFrame(side_family_rows)


def run_materialization(
    *,
    source_labels_dir: Path,
    output_labels_dir: Path,
    output_run_id: str,
    data_root: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    market_mode: str,
    exchange: str,
    timeframe: str,
    path_len: int,
    apply_delayed_entry: bool,
    entry_delay_hours: int,
    policy_label_center: float,
    policy_label_temperature: float,
    outcome_mode: str,
    round_trip_cost: float,
    policy_manifest: dict[str, Any],
    regime_family_min_score: float,
    regime_family_min_score_gap: float,
    chunk_frequency: str,
    chunk_months: set[str] | None,
    dataset_regex: str | None,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if str(market_mode).strip().lower() == "perps":
        os.environ.setdefault("EPM_SIMPLE_POLICY_15M_CHART_ONLY", "1")
    if (
        output_labels_dir.exists()
        and any(output_labels_dir.iterdir())
        and not overwrite
        and not resume
    ):
        raise FileExistsError(
            f"{output_labels_dir} already exists; pass --overwrite to replace files"
        )
    source_manifest = _read_manifest(source_labels_dir)
    datasets = source_manifest.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError(
            f"No datasets found in {source_labels_dir / 'labels_manifest.json'}"
        )

    output_labels_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = {
        "run_id": str(output_run_id),
        "source_labels_dir": str(source_labels_dir),
        "source_manifest": str(source_labels_dir / "labels_manifest.json"),
        "datasets": {},
        "materialized_side_archetype_trailing_labels": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root),
            "feature_dir": str(feature_dir),
            "feature_list_csv": str(feature_list_csv),
            "max_feature_store_features": max_feature_store_features,
            "market_mode": str(market_mode),
            "exchange": str(exchange),
            "timeframe": str(timeframe),
            "path_len": int(path_len),
            "apply_delayed_entry": bool(apply_delayed_entry),
            "entry_delay_hours": int(entry_delay_hours),
            "path_start_contract": (
                "signal_timestamp_plus_timeframe_then_optional_delayed_execution"
            ),
            "label_entry_contract": (
                "mandatory_signal_close_plus_one_signal_timeframe"
                if not apply_delayed_entry
                else "mandatory_offset_plus_optional_delayed_execution_diagnostic"
            ),
            "delayed_slippage_contract_owner": (
                "simple_policy_optimiser_and_live_inference"
                if not apply_delayed_entry
                else "label_diagnostic_and_simple_policy_optimiser"
            ),
            "minimum_path_coverage": float(
                os.environ.get("EPM_LABEL_MIN_PATH_COVERAGE", "0.95")
            ),
            "native_15m_chart_only": str(
                os.environ.get("EPM_SIMPLE_POLICY_15M_CHART_ONLY", "")
            )
            .strip()
            .lower()
            not in {"0", "false", "no", "n", "off"},
            "policy_label_center": float(policy_label_center),
            "policy_label_temperature": float(policy_label_temperature),
            "outcome_mode": str(outcome_mode),
            "round_trip_cost": float(round_trip_cost),
            "regime_family_min_score": float(regime_family_min_score),
            "regime_family_min_score_gap": float(regime_family_min_score_gap),
            "chunk_frequency": str(chunk_frequency),
            "chunk_months": sorted(chunk_months) if chunk_months else None,
            "dataset_regex": str(dataset_regex) if dataset_regex else None,
            "resume": bool(resume),
            "policy_manifest": policy_manifest,
        },
    }
    summaries: list[dict[str, Any]] = []
    side_archetype_frames: list[pd.DataFrame] = []
    dataset_pattern = re.compile(str(dataset_regex)) if dataset_regex else None
    for dataset_name, meta in datasets.items():
        if not isinstance(meta, dict):
            continue
        if dataset_pattern is not None and not dataset_pattern.search(
            str(dataset_name)
        ):
            continue
        file_name = str(meta.get("file") or "")
        if not file_name or not file_name.endswith(".parquet"):
            continue
        source_path = source_labels_dir / file_name
        side = _infer_side(str(dataset_name), file_name, None)
        for suffix, start_ts, end_ts in _dataset_chunk_windows(
            source_path, chunk_frequency
        ):
            if chunk_months and suffix not in chunk_months:
                continue
            chunk_dataset_name = _chunked_dataset_name(str(dataset_name), suffix)
            chunk_file_name = _chunked_file_name(file_name, suffix)
            output_path = output_labels_dir / chunk_file_name
            if resume and output_path.exists():
                print(
                    f"[materialize_archetype_conditioned_trailing_labels] skip existing "
                    f"dataset={chunk_dataset_name} output={output_path}",
                    flush=True,
                )
                existing = pd.read_parquet(output_path)
                out_meta = dict(meta)
                out_meta["file"] = chunk_file_name
                out_meta["rows"] = int(len(existing))
                out_meta["columns"] = list(existing.columns)
                out_meta["source_dataset"] = str(dataset_name)
                out_meta["chunk_frequency"] = str(chunk_frequency)
                out_meta["chunk_suffix"] = str(suffix)
                out_meta["min_ts"] = str(
                    pd.to_datetime(existing["__ts__"], errors="coerce").min()
                )
                out_meta["max_ts"] = str(
                    pd.to_datetime(existing["__ts__"], errors="coerce").max()
                )
                out_manifest["datasets"][chunk_dataset_name] = out_meta
                continue
            print(
                f"[materialize_archetype_conditioned_trailing_labels] start dataset={chunk_dataset_name} "
                f"side={side} source={source_path} output={output_path} start_ts={start_ts or 'ALL'} "
                f"end_ts={end_ts or 'ALL'}",
                flush=True,
            )
            summary, side_archetype = _materialize_dataset(
                source_path=source_path,
                output_path=output_path,
                dataset_name=chunk_dataset_name,
                side=side,
                data_root=data_root,
                feature_dir=feature_dir,
                feature_list_csv=feature_list_csv,
                max_feature_store_features=max_feature_store_features,
                market_mode=market_mode,
                exchange=exchange,
                timeframe=timeframe,
                path_len=path_len,
                apply_delayed_entry=apply_delayed_entry,
                entry_delay_hours=int(entry_delay_hours),
                policy_label_center=policy_label_center,
                policy_label_temperature=policy_label_temperature,
                outcome_mode=outcome_mode,
                round_trip_cost=float(round_trip_cost),
                policy_manifest=policy_manifest,
                regime_family_min_score=float(regime_family_min_score),
                regime_family_min_score_gap=float(regime_family_min_score_gap),
                start_ts=start_ts or None,
                end_ts=end_ts or None,
            )
            summaries.append(summary)
            side_archetype_frames.append(side_archetype)
            out_meta = dict(meta)
            out_meta["file"] = chunk_file_name
            out_meta["rows"] = int(summary["rows"])
            out_meta["columns"] = list(summary.get("columns", []))
            out_meta["source_dataset"] = str(dataset_name)
            out_meta["chunk_frequency"] = str(chunk_frequency)
            out_meta["chunk_suffix"] = str(suffix)
            out_meta["min_ts"] = summary.get("min_ts")
            out_meta["max_ts"] = summary.get("max_ts")
            out_meta["start_ts"] = summary.get("start_ts")
            out_meta["end_ts"] = summary.get("end_ts")
            out_manifest["datasets"][chunk_dataset_name] = out_meta
            print(
                f"[materialize_archetype_conditioned_trailing_labels] done dataset={chunk_dataset_name} "
                f"rows={summary['rows']} finite_path_coverage="
                f"{summary.get('path_fetch', {}).get('finite_path_coverage', np.nan)}",
                flush=True,
            )

    if not summaries and not out_manifest["datasets"]:
        raise RuntimeError(
            f"No parquet datasets were materialized from {source_labels_dir}"
        )

    summary_path = (
        output_labels_dir / "side_archetype_trailing_materialization_summary.json"
    )
    manifest_path = output_labels_dir / "labels_manifest.json"
    policy_path = output_labels_dir / "side_archetype_label_manifest.json"
    policy_csv_path = output_labels_dir / "side_archetype_label_manifest.csv"
    side_archetype_path = output_labels_dir / "side_archetype_label_metrics.csv"

    policy_rows: list[dict[str, Any]] = []
    for side in ("long", "short"):
        defaults = policy_manifest.get("default", {}) or {}
        if side in defaults:
            row = dict(defaults[side])
            row.update({"side": side, "archetype": "__default__"})
            policy_rows.append(row)
    policy_rows.extend(list(policy_manifest.get("overrides", []) or []))
    pd.DataFrame(policy_rows).to_csv(policy_csv_path, index=False)
    if side_archetype_frames:
        pd.concat(side_archetype_frames, ignore_index=True).to_csv(
            side_archetype_path, index=False
        )
    else:
        pd.DataFrame().to_csv(side_archetype_path, index=False)

    summary_path.write_text(
        json.dumps(_json_safe({"datasets": summaries}), indent=2), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(_json_safe(out_manifest), indent=2), encoding="utf-8"
    )
    policy_path.write_text(
        json.dumps(_json_safe(policy_manifest), indent=2), encoding="utf-8"
    )
    return {
        "output_labels_dir": str(output_labels_dir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "side_archetype_label_manifest_json": str(policy_path),
        "side_archetype_label_manifest_csv": str(policy_csv_path),
        "side_archetype_label_metrics": str(side_archetype_path),
        "datasets": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-labels-dir", type=Path, default=DEFAULT_SOURCE_LABELS_DIR
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument(
        "--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV
    )
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--output-labels-dir", type=Path, default=None)
    parser.add_argument("--policy-manifest", type=Path, default=None)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument(
        "--entry-delay-hours",
        type=int,
        default=1,
        help=(
            "Deprecated compatibility check. It must equal the signal timeframe; "
            "the signal-close offset is always enforced."
        ),
    )
    parser.add_argument(
        "--apply-delayed-entry",
        dest="apply_delayed_entry",
        action="store_true",
        help=(
            "Apply the additional 1m delayed/slipped fill model after the mandatory "
            "signal-close offset."
        ),
    )
    parser.add_argument(
        "--no-delayed-entry",
        dest="apply_delayed_entry",
        action="store_false",
        help=(
            "Keep the base label on the mandatory next-bar decision anchor. "
            "The additional delayed/slipped fill remains a downstream policy "
            "execution adjustment."
        ),
    )
    parser.add_argument("--policy-label-center", type=float, default=0.0)
    parser.add_argument("--policy-label-temperature", type=float, default=0.004)
    parser.add_argument(
        "--outcome-mode",
        choices=("fixed_tp", "trailing_profit"),
        default="trailing_profit",
    )
    parser.add_argument("--round-trip-cost", type=float, default=0.01)
    parser.add_argument(
        "--regime-family-min-score", type=float, default=DEFAULT_REGIME_FAMILY_MIN_SCORE
    )
    parser.add_argument(
        "--regime-family-min-score-gap",
        type=float,
        default=DEFAULT_REGIME_FAMILY_MIN_SCORE_GAP,
    )
    parser.add_argument(
        "--chunk-frequency", choices=("none", "monthly"), default="none"
    )
    parser.add_argument(
        "--chunk-months",
        default=None,
        help="Optional comma-separated monthly chunk filter, for example 2026-07,2026-08.",
    )
    parser.add_argument("--dataset-regex", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(apply_delayed_entry=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_labels_dir = args.output_labels_dir
    if output_labels_dir is None:
        output_labels_dir = (
            args.data_root / "artifacts" / str(args.output_run_id) / "labels"
        )
    result = run_materialization(
        source_labels_dir=args.source_labels_dir,
        output_labels_dir=output_labels_dir,
        output_run_id=str(args.output_run_id),
        data_root=args.data_root,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        market_mode=str(args.market_mode),
        exchange=str(args.exchange),
        timeframe=str(args.timeframe),
        path_len=int(args.path_len),
        apply_delayed_entry=bool(args.apply_delayed_entry),
        entry_delay_hours=int(args.entry_delay_hours),
        policy_label_center=float(args.policy_label_center),
        policy_label_temperature=float(args.policy_label_temperature),
        outcome_mode=str(args.outcome_mode),
        round_trip_cost=float(args.round_trip_cost),
        policy_manifest=_load_policy_manifest(args.policy_manifest),
        regime_family_min_score=float(args.regime_family_min_score),
        regime_family_min_score_gap=float(args.regime_family_min_score_gap),
        chunk_frequency=str(args.chunk_frequency),
        chunk_months=(
            {
                token.strip().replace("-", "_")
                for token in str(args.chunk_months or "").split(",")
                if token.strip()
            }
            or None
        ),
        dataset_regex=args.dataset_regex,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(_json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
