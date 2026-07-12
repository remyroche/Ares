#!/usr/bin/env python3
"""Backcast one frozen base/meta champion contract over Jan 2025-Jul 10 2026.

This is a fixed-model comparability diagnostic.  January-June rows precede the
bundle fit cutoff and are therefore retrospective/in-sample backcasts, not
chronological OOS evidence.  July rows are the frozen post-fit evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.features_gmm_ae import transform_ae_gmm_features
from extreme_price_movements.regime_ev_calibration import (
    apply_regime_ev_calibration,
    load_regime_ev_calibration,
)
from extreme_price_movements.regime_ev_calibration import (
    required_feature_columns as regime_required_feature_columns,
)
from scripts.backfill_complete_july_meta_predictions import _source_tags
from scripts.report_meta_residual_champion_long_history import (
    _autocorrelation_rows,
    _daily_cells,
)
from scripts.run_train_meta_residual_archetype_enhancement import (
    _add_reference_fold_features,
)
from scripts.score_compare_meta_residual_july_oos import _append_store_features

ROOT = Path("data_perp/reports/train_meta_residual_archetype_enhancement_20260711_v1")
DEFAULT_LABELS = Path(
    "data_perp/artifacts/"
    "20260708_s59_h5_2025start_monthly_v6_15mchart_trailing_cost100bps_labels/labels"
)
DEFAULT_FEATURES = Path("data_perp/features/20260710_170000")
DEFAULT_FROZEN = Path("data_perp/artifacts/s59_s52_frozen_inference_bundle_20260709")
DEFAULT_BUNDLE_DIR = ROOT / "inference_bundle_residual_pca8_globaloverlay_shock"
DEFAULT_REFERENCE = ROOT / "cache/compact_reference_with_lifecycle.parquet"
DEFAULT_JULY_EARLY = (
    ROOT / "july_predictions_through_20260711/july_predictions_combined.parquet"
)
DEFAULT_JULY_COMPLETE = (
    ROOT / "july_complete_08_10/july_08_10_complete_predictions.parquet"
)
DEFAULT_OUTPUT = ROOT / "champion_frozen_single_source_202501_20260710"
DEFAULT_THRESHOLD_POLICY = DEFAULT_FROZEN / "policy_params/threshold_basis_policy.json"
DEFAULT_REGIME_CALIBRATION = DEFAULT_FROZEN / "policy_params/regime_ev_calibration.json"
DEFAULT_SOURCE_MANIFEST = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v6_15mchart_base_frozenfs_fixedparams_may_july_combined_20260708/"
    "s52_trailing_regime_meta_handoff_top30_allsafe_20260708/manifest.json"
)
DEFAULT_BASE_REFERENCE = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v6_15mchart_base_frozenfs_fixedparams_"
    "trainthroughjun_scorejul_20260709_savedmodels_fromcache/best_oos_scored_ledger.parquet"
)

KEYS = ["__ts__", "__symbol__", "side_name"]
OUTCOME_COLUMNS = [
    "ev_after_1pct",
    "exec_margin",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
]
LABEL_OUTCOME_COLUMNS = [
    "__first_touch_capture_net__",
    "__first_touch_valid_path__",
    "__first_touch_mae_norm__",
    "__first_touch_full_path_mae_norm__",
    "__first_touch_timeout__",
    "__mfe_1r_before_mae_1r__",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _columns(path: Path) -> list[str]:
    return [str(name) for name in pq.ParquetFile(path).schema_arrow.names]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _label_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    valid = pd.to_numeric(frame["__first_touch_valid_path__"], errors="coerce").gt(0.5)
    capture = pd.to_numeric(frame["__first_touch_capture_net__"], errors="coerce")
    first_mae = pd.to_numeric(frame["__first_touch_mae_norm__"], errors="coerce")
    full_mae = pd.to_numeric(
        frame["__first_touch_full_path_mae_norm__"], errors="coerce"
    )
    timeout = pd.to_numeric(frame["__first_touch_timeout__"], errors="coerce").fillna(
        0.0
    )
    mfe_first = pd.to_numeric(
        frame["__mfe_1r_before_mae_1r__"], errors="coerce"
    ).fillna(0.0)
    out = pd.DataFrame(index=frame.index)
    out["exec_margin"] = capture.where(valid)
    out["ev_after_1pct"] = (capture - 0.01).where(valid)
    out["first_touch_bad_mae_1r"] = first_mae.ge(1.0).astype(np.float32).where(valid)
    out["full_path_bad_mae_1r"] = full_mae.ge(1.0).astype(np.float32).where(valid)
    out["timeout"] = timeout.gt(0.5).astype(np.float32).where(valid)
    out["clean_exec"] = (
        (capture.gt(0.0) & first_mae.lt(1.0) & timeout.lt(0.5) & mfe_first.gt(0.5))
        .astype(np.float32)
        .where(valid)
    )
    out["dirty_positive"] = (
        (capture.gt(0.0) & (first_mae.ge(1.0) | full_mae.ge(1.0) | timeout.gt(0.5)))
        .astype(np.float32)
        .where(valid)
    )
    return out


def _append_missing_store_features(
    frame: pd.DataFrame,
    feature_root: Path,
    requested: Iterable[str],
) -> pd.DataFrame:
    missing = [
        str(name)
        for name in dict.fromkeys(requested)
        if str(name) not in frame.columns
        and not str(name).startswith("meta_resid_pca_")
    ]
    if not missing:
        return frame
    enriched, _coverage = _append_store_features(frame, feature_root, missing)
    return enriched


def _transform_base_state_chunked(
    frame: pd.DataFrame,
    state: dict[str, Any],
    *,
    batch_rows: int = 50_000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    columns = list(state["feature_columns"])
    for start in range(0, len(frame), int(batch_rows)):
        stop = min(start + int(batch_rows), len(frame))
        part = transform_ae_gmm_features(
            frame.iloc[start:stop].reindex(columns=columns),
            state,
            index=frame.index[start:stop],
        )
        parts.append(part)
    return pd.concat(parts, axis=0).reindex(frame.index)


def _load_label_month(
    labels_dir: Path,
    month: str,
    requested: Iterable[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    year, number = month.split("-")
    for side in ("long", "short"):
        path = labels_dir / f"train_global_{side}_5_{year}_{number}.parquet"
        available = set(_columns(path))
        identity = [
            *KEYS,
            "side",
            "__side__",
            "__archetype_policy_key__",
            "__archetype_label_family__",
            *LABEL_OUTCOME_COLUMNS,
        ]
        columns = [
            name for name in dict.fromkeys([*identity, *requested]) if name in available
        ]
        part = pd.read_parquet(path, columns=columns)
        part["__ts__"] = pd.to_datetime(part["__ts__"], utc=True, errors="coerce")
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True, copy=False)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["archetype_policy_key"] = frame["__archetype_policy_key__"].astype(str)
    return frame


def _prepare_month_candidates(
    *,
    month: str,
    labels_dir: Path,
    feature_root: Path,
    base_columns: list[str],
    base_model: Any,
    base_state: dict[str, Any],
    bundle: Any,
    base_cutoff: float,
    source_edges: list[float | None],
    output: Path,
) -> dict[str, Any]:
    raw_requested = list(
        dict.fromkeys(
            [
                *base_columns,
                *base_state["feature_columns"],
                *bundle.required_input_features(),
                *bundle.raw_selected_features,
            ]
        )
    )
    frame = _load_label_month(labels_dir, month, raw_requested)
    outcome = _label_outcomes(frame)
    for name in outcome.columns:
        frame[name] = outcome[name].to_numpy(copy=False)
    frame = _append_missing_store_features(
        frame,
        feature_root,
        [
            *base_columns,
            *base_state["feature_columns"],
            *bundle.required_input_features(),
        ],
    )
    generated = _transform_base_state_chunked(frame, base_state)
    for name in generated.columns:
        if name in base_columns or name in bundle.required_input_features():
            frame[name] = generated[name].to_numpy(dtype=np.float32, copy=False)
    matrix = frame.reindex(columns=base_columns).replace([np.inf, -np.inf], np.nan)
    frame["score"] = np.asarray(base_model.predict(matrix), dtype=np.float32)
    frame = frame.loc[
        pd.to_numeric(frame["score"], errors="coerce").ge(base_cutoff)
    ].copy()
    frame["selected_top30"] = True
    frame["source_tag"] = _source_tags(
        frame["score"], frame["side_name"], source_edges
    ).to_numpy()
    frame["row_id"] = (
        month.replace("-", "")
        + "__"
        + pd.Series(np.arange(len(frame), dtype=np.int64), index=frame.index).astype(
            str
        )
    )
    keep = list(
        dict.fromkeys(
            [
                "row_id",
                *KEYS,
                "archetype_policy_key",
                "source_tag",
                "score",
                "selected_top30",
                *OUTCOME_COLUMNS,
                *bundle.required_input_features(),
                *bundle.raw_selected_features,
            ]
        )
    )
    result = frame[[name for name in keep if name in frame.columns]].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    summary = {
        "month": month,
        "label_rows": int(len(matrix)),
        "candidate_rows": int(len(result)),
        "timestamps": int(result["__ts__"].nunique()),
        "symbols": int(result["__symbol__"].nunique()),
    }
    del frame, matrix, generated, result, outcome
    gc.collect()
    return summary


def _reference_features(
    reference_path: Path,
    candidate_paths: list[Path],
    output: Path,
) -> list[str]:
    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "source_tag",
        "score",
        "selected_top30",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
        "exec_margin",
    ]
    available = set(_columns(reference_path))
    train = pd.read_parquet(
        reference_path, columns=[name for name in columns if name in available]
    )
    train["__ts__"] = pd.to_datetime(train["__ts__"], utc=True, errors="coerce")
    candidate_parts = [
        pd.read_parquet(
            path,
            columns=[
                name for name in ["row_id", *columns] if name in set(_columns(path))
            ],
        )
        for path in candidate_paths
    ]
    valid = pd.concat(candidate_parts, ignore_index=True, copy=False)
    _train_enriched, valid_enriched = _add_reference_fold_features(train, valid)
    generated = [
        name
        for name in valid_enriched.columns
        if name.startswith("base_") or name.startswith("rel_")
    ]
    valid_enriched[["row_id", *generated]].to_parquet(
        output, index=False, compression="zstd"
    )
    del train, valid, _train_enriched, valid_enriched, candidate_parts
    gc.collect()
    return generated


def _score_month(
    *,
    candidate_path: Path,
    reference_features: pd.DataFrame,
    bundle: Any,
    output: Path,
    batch_rows: int = 50_000,
) -> dict[str, Any]:
    frame = pd.read_parquet(candidate_path)
    frame = frame.merge(
        reference_features, on="row_id", how="left", validate="one_to_one"
    )
    scored_parts: list[pd.DataFrame] = []
    for start in range(0, len(frame), int(batch_rows)):
        stop = min(start + int(batch_rows), len(frame))
        scored_parts.append(bundle.predict(frame.iloc[start:stop]))
    scores = pd.concat(scored_parts, axis=0).reindex(frame.index)
    for name in scores.columns:
        frame[name] = scores[name].to_numpy(dtype=np.float32, copy=False)
    frame["selected_for_monitor"] = pd.to_numeric(
        frame["historical_rank"], errors="coerce"
    ).ge(0.90)
    frame["prediction_evidence"] = "frozen_champion_retrospective_backcast"
    keep = [
        "row_id",
        *KEYS,
        "archetype_policy_key",
        "source_tag",
        "score",
        "score_lifecycle_only",
        "score_residual_overlay",
        "score_shock_adjusted",
        "shock_composite_raw",
        "shock_composite_local",
        "hit_probability",
        "historical_rank",
        "selected_for_monitor",
        *OUTCOME_COLUMNS,
        "prediction_evidence",
    ]
    result = frame[[name for name in keep if name in frame.columns]].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    selected = result.loc[result["selected_for_monitor"]]
    summary = {
        "month": str(result["__ts__"].dt.strftime("%Y-%m").iloc[0]),
        "candidate_rows": int(len(result)),
        "selected_rows": int(len(selected)),
        "mean_ev_after_1pct": float(
            pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean()
        ),
        "clean_exec_precision": float(
            pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
        ),
    }
    del frame, result, selected, scores, scored_parts
    gc.collect()
    return summary


def _july_uniform(
    early_path: Path, complete_path: Path, output: Path
) -> dict[str, Any]:
    early = pd.read_parquet(early_path)
    complete = pd.read_parquet(complete_path)
    for frame in (early, complete):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    early = early.loc[
        early["__ts__"].ge(pd.Timestamp("2026-07-01", tz="UTC"))
        & early["__ts__"].lt(pd.Timestamp("2026-07-08", tz="UTC"))
    ].copy()
    complete = complete.loc[
        complete["__ts__"].ge(pd.Timestamp("2026-07-08", tz="UTC"))
        & complete["__ts__"].lt(pd.Timestamp("2026-07-11", tz="UTC"))
    ].copy()
    combined = pd.concat([early, complete], ignore_index=True, sort=False, copy=False)
    combined = combined.sort_values(KEYS, kind="stable").drop_duplicates(
        KEYS, keep="last"
    )
    combined["selected_for_monitor"] = pd.to_numeric(
        combined["historical_rank"], errors="coerce"
    ).ge(0.90)
    combined["prediction_evidence"] = "frozen_champion_postfit_oos"
    combined["row_id"] = "202607__" + pd.Series(
        np.arange(len(combined), dtype=np.int64)
    ).astype(str)
    combined.to_parquet(output, index=False, compression="zstd")
    selected = combined.loc[combined["selected_for_monitor"]]
    return {
        "month": "2026-07",
        "candidate_rows": int(len(combined)),
        "selected_rows": int(len(selected)),
        "mean_ev_after_1pct": float(
            pd.to_numeric(selected["ev_after_1pct"], errors="coerce").mean()
        ),
        "clean_exec_precision": float(
            pd.to_numeric(selected["clean_exec"], errors="coerce").mean()
        ),
    }


def _rank_against_reference(values: pd.Series, reference: pd.Series) -> np.ndarray:
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    ref = np.sort(ref[np.isfinite(ref)])
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    result = np.full(raw.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(raw)
    if ref.size and finite.any():
        result[finite] = (
            np.searchsorted(ref, raw[finite], side="right") / float(ref.size)
        ).astype(np.float32)
    return result


def _threshold_for_target_ev(
    score: pd.Series,
    outcome: pd.Series,
    *,
    target_ev: float,
    min_rows: int,
) -> float:
    score_values = pd.to_numeric(score, errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    ev_values = pd.to_numeric(outcome, errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    valid = np.isfinite(score_values) & np.isfinite(ev_values)
    if valid.sum() < int(min_rows) or not np.isfinite(target_ev):
        return float("nan")
    score_values = score_values[valid]
    ev_values = ev_values[valid]
    grid = np.unique(np.nanquantile(score_values, np.linspace(0.70, 0.99, 60)))
    best_threshold = float("nan")
    best_gap = float("inf")
    for threshold in grid:
        chosen = score_values >= float(threshold)
        if chosen.sum() < int(min_rows):
            continue
        mean_ev = float(np.mean(ev_values[chosen]))
        gap = abs(mean_ev - float(target_ev))
        if mean_ev >= float(target_ev) and gap < best_gap:
            best_gap = gap
            best_threshold = float(threshold)
    if not np.isfinite(best_threshold):
        best_threshold = float(np.nanquantile(score_values, 0.99))
    return best_threshold


def _append_regime_calibrated_scores(
    frame: pd.DataFrame,
    *,
    candidate_dir: Path,
    feature_root: Path,
    artifact_path: Path,
) -> pd.DataFrame:
    """Apply the frozen regime calibrator without rerunning base/meta models."""
    out = frame.copy()
    base_score = pd.to_numeric(out.get("score_shock_adjusted"), errors="coerce")
    alternative_score = pd.to_numeric(
        out.get("score_regime_alternative"), errors="coerce"
    )
    out["score_regime_calibrated"] = alternative_score.where(
        alternative_score.notna(), base_score
    ).astype(np.float32)
    out["regime_ev_risk_score"] = np.float32(0.0)
    out["regime_ev_effect_count"] = np.int16(0)
    artifact = load_regime_ev_calibration(artifact_path)
    artifact["_raise_model_prediction_errors"] = True
    required = regime_required_feature_columns(artifact)
    # The saved calibrator has no active effects before May 2026.  Earlier rows
    # retain the frozen champion score, while May/June receive the exact frozen
    # side/archetype regime adjustment. July already carries its scored value.
    for month in ("2026-05", "2026-06"):
        mask = out["__ts__"].dt.strftime("%Y-%m").eq(month)
        if not bool(mask.any()):
            continue
        candidate_path = candidate_dir / f"candidates_{month}.parquet"
        available = set(_columns(candidate_path))
        columns = [
            name
            for name in ["row_id", *KEYS, "archetype_policy_key", *required]
            if name in available
        ]
        candidate = pd.read_parquet(candidate_path, columns=columns)
        scores = out.loc[mask, ["row_id", "score_shock_adjusted"]].copy()
        score_lookup = scores.set_index("row_id")["score_shock_adjusted"]
        update_columns = [
            "row_id",
            "score_regime_calibrated",
            "regime_ev_risk_score",
            "regime_ev_effect_count",
        ]
        calibrated_parts: list[pd.DataFrame] = []
        for _symbol, raw_positions in candidate.groupby(
            "__symbol__", sort=False
        ).indices.items():
            positions = np.asarray(raw_positions, dtype=np.int64)
            part = candidate.iloc[positions].copy().reset_index(drop=True)
            missing = [name for name in required if name not in part.columns]
            if missing:
                part = _append_missing_store_features(part, feature_root, missing)
            part["score_shock_adjusted"] = part["row_id"].map(score_lookup)
            part["score_meta_base_soft_label"] = pd.to_numeric(
                part["score_shock_adjusted"], errors="coerce"
            ).astype(np.float32)
            part = apply_regime_ev_calibration(
                part,
                artifact,
                source_score_col="score_meta_base_soft_label",
                adjusted_score_col="score_regime_calibrated",
                side_col="side_name",
                archetype_col="archetype_policy_key",
                copy=False,
            )
            if not calibrated_parts:
                feature_coverage = {
                    name: float(
                        pd.to_numeric(part[name], errors="coerce").notna().mean()
                    )
                    for name in required
                    if name in part.columns
                }
                print(
                    json.dumps(
                        {
                            "event": "regime_calibration_first_symbol",
                            "month": month,
                            "symbol": str(_symbol),
                            "rows": int(len(part)),
                            "score_rows": int(
                                part["score_shock_adjusted"].notna().sum()
                            ),
                            "effect_rows": int(
                                pd.to_numeric(
                                    part["regime_ev_effect_count"], errors="coerce"
                                )
                                .fillna(0)
                                .gt(0)
                                .sum()
                            ),
                            "timestamp_min": str(part["__ts__"].min()),
                            "timestamp_max": str(part["__ts__"].max()),
                            "archetypes": part["archetype_policy_key"]
                            .value_counts()
                            .to_dict(),
                            "zero_coverage_features": sorted(
                                name
                                for name, coverage in feature_coverage.items()
                                if coverage == 0.0
                            ),
                            "minimum_feature_coverage": min(
                                feature_coverage.values(), default=0.0
                            ),
                        }
                    ),
                    flush=True,
                )
            calibrated_parts.append(part[update_columns])
        calibrated = pd.concat(calibrated_parts, ignore_index=True, copy=False)
        calibrated_effect_rows = int(
            pd.to_numeric(calibrated["regime_ev_effect_count"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        updates = out.loc[mask, ["row_id"]].merge(
            calibrated[update_columns],
            on="row_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        effect_rows = int(
            pd.to_numeric(updates["regime_ev_effect_count"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        if effect_rows == 0:
            raise ValueError(
                f"Frozen regime calibration produced no active effects for {month}."
            )
        out.loc[mask, "score_regime_calibrated"] = pd.to_numeric(
            updates["score_regime_calibrated"], errors="coerce"
        ).to_numpy(dtype=np.float32, copy=False)
        out.loc[mask, "regime_ev_risk_score"] = (
            pd.to_numeric(updates["regime_ev_risk_score"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32, copy=False)
        )
        out.loc[mask, "regime_ev_effect_count"] = (
            pd.to_numeric(updates["regime_ev_effect_count"], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int16, copy=False)
        )
        print(
            json.dumps(
                {
                    "event": "regime_calibration_month",
                    "month": month,
                    "rows": int(len(updates)),
                    "effect_rows": effect_rows,
                    "calibrated_effect_rows": calibrated_effect_rows,
                }
            ),
            flush=True,
        )
        del candidate, calibrated, calibrated_parts, updates, scores
        gc.collect()
    return out


def _apply_causal_reachable_ev_policy(
    frame: pd.DataFrame,
    *,
    policy: dict[str, Any],
    score_col: str = "score_regime_calibrated",
    preserve_materialized_policy: bool = True,
) -> pd.DataFrame:
    """Apply the promoted 8d HR-off policy using only prior-day evidence.

    The production artifact contains only its recent live reference parquet.
    For the longer backcast we rebuild the same reference fields causally from
    this one frozen score stream. Thresholds are fixed for each UTC day, so no
    outcome from the current day can affect another decision on that day.
    """
    out = frame.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    exact_policy_mask = pd.Series(False, index=out.index)
    if preserve_materialized_policy and "threshold_alternative_rank" in out.columns:
        exact_policy_mask = pd.to_numeric(
            out["threshold_alternative_rank"], errors="coerce"
        ).notna()
    if score_col not in out.columns:
        raise ValueError(f"Causal reachable-EV score column is missing: {score_col}")
    return_col = "ev_after_1pct"
    window_days = int(policy.get("window_days") or 8)
    min_rows = int(policy.get("min_reference_rows") or 40)
    arch_min_rows = int(policy.get("arch_min_reference_rows") or max(8, min_rows // 4))
    top_band_floor = float(policy.get("top_band_floor") or 0.90)
    out["policy_baseline_rank"] = np.float32(np.nan)
    out["threshold_basis_rank_score"] = np.float32(0.0)
    out["threshold_basis_selected"] = False
    out["threshold_basis_dynamic_ev_target"] = np.float32(np.nan)
    out["threshold_basis_dynamic_score_threshold"] = np.float32(np.nan)
    out["threshold_basis_recent_reference_rows"] = np.int32(0)
    out["threshold_basis_baseline_activity_count"] = np.int16(0)
    # Force nanoseconds explicitly. Pandas can preserve Arrow's datetime64[us]
    # unit through both astype(int64) and DatetimeIndex.asi8.
    timestamp_ns = out["__ts__"].to_numpy(dtype="datetime64[ns]").view("int64")
    day_ns = out["__ts__"].dt.floor("D").to_numpy(dtype="datetime64[ns]").view("int64")
    day_values = np.unique(day_ns)
    one_day_ns = int(pd.Timedelta(days=1).value)
    global_sum = 0.0
    global_count = 0
    arch_sum: dict[str, float] = {}
    arch_count: dict[str, int] = {}
    total_days = int(len(day_values))
    for day_number, day_value in enumerate(day_values, start=1):
        start = int(np.searchsorted(timestamp_ns, day_value, side="left"))
        stop = int(np.searchsorted(timestamp_ns, day_value + one_day_ns, side="left"))
        recent_start = int(
            np.searchsorted(
                timestamp_ns,
                day_value - window_days * one_day_ns,
                side="left",
            )
        )
        current = out.iloc[start:stop]
        recent = out.iloc[recent_start:start]
        if len(recent) >= min_rows:
            baseline_rank = _rank_against_reference(
                current[score_col], recent[score_col]
            )
            out.loc[current.index, "policy_baseline_rank"] = baseline_rank
            current = out.loc[current.index]
            global_target = (
                global_sum / global_count
                if global_count >= max(1, min_rows // 2)
                else float("nan")
            )
            global_threshold = _threshold_for_target_ev(
                recent[score_col],
                recent[return_col],
                target_ev=global_target,
                min_rows=min_rows,
            )
            local_rank = pd.Series(np.nan, index=current.index, dtype="float64")
            local_threshold = pd.Series(
                global_threshold, index=current.index, dtype="float64"
            )
            local_target = pd.Series(
                global_target, index=current.index, dtype="float64"
            )
            for archetype, sub in current.groupby("archetype_policy_key", dropna=False):
                key = str(archetype)
                ref_arch = recent.loc[
                    recent["archetype_policy_key"].astype(str).eq(key)
                ]
                target = (
                    arch_sum.get(key, 0.0) / arch_count.get(key, 0)
                    if arch_count.get(key, 0) >= max(1, arch_min_rows // 2)
                    else float("nan")
                )
                threshold = _threshold_for_target_ev(
                    ref_arch[score_col],
                    ref_arch[return_col],
                    target_ev=target,
                    min_rows=arch_min_rows,
                )
                if not np.isfinite(threshold):
                    threshold = global_threshold
                    target = global_target
                    rank_reference = recent[score_col]
                else:
                    rank_reference = ref_arch[score_col]
                local_rank.loc[sub.index] = _rank_against_reference(
                    sub[score_col], rank_reference
                )
                local_threshold.loc[sub.index] = threshold
                local_target.loc[sub.index] = target
            day_selected = np.zeros(len(current), dtype=bool)
            day_rank_score = np.zeros(len(current), dtype=np.float32)
            day_target = np.full(len(current), np.nan, dtype=np.float32)
            day_threshold = np.full(len(current), np.nan, dtype=np.float32)
            day_activity = np.zeros(len(current), dtype=np.int16)
            for _timestamp, batch in current.groupby("__ts__", sort=True):
                baseline_count = int(
                    pd.to_numeric(batch["policy_baseline_rank"], errors="coerce")
                    .ge(0.90)
                    .sum()
                )
                batch_positions = (
                    batch.index.to_numpy(dtype=np.int64, copy=False) - start
                )
                day_activity[batch_positions] = baseline_count
                if baseline_count <= 0:
                    continue
                eligible = batch.index[
                    pd.to_numeric(batch[score_col], errors="coerce").to_numpy()
                    >= local_threshold.loc[batch.index].to_numpy()
                ]
                ranked = local_rank.loc[batch.index].sort_values(
                    ascending=False, kind="stable"
                )
                if len(eligible) >= baseline_count:
                    chosen = (
                        local_rank.loc[eligible]
                        .sort_values(ascending=False, kind="stable")
                        .head(baseline_count)
                        .index
                    )
                else:
                    chosen = ranked.head(baseline_count).index
                if len(chosen) == 0:
                    continue
                rank_values = local_rank.loc[chosen].rank(method="first", pct=True)
                positions = chosen.to_numpy(dtype=np.int64, copy=False) - start
                day_selected[positions] = True
                day_rank_score[positions] = (
                    top_band_floor + (1.0 - top_band_floor) * rank_values
                ).to_numpy(dtype=np.float32, copy=False)
                day_target[positions] = local_target.loc[chosen].to_numpy(
                    dtype=np.float32, copy=False
                )
                day_threshold[positions] = local_threshold.loc[chosen].to_numpy(
                    dtype=np.float32, copy=False
                )
            current_index = current.index
            out.loc[current_index, "threshold_basis_recent_reference_rows"] = int(
                len(recent)
            )
            out.loc[current_index, "threshold_basis_baseline_activity_count"] = (
                day_activity
            )
            out.loc[current_index, "threshold_basis_selected"] = day_selected
            out.loc[current_index, "threshold_basis_rank_score"] = day_rank_score
            out.loc[current_index, "threshold_basis_dynamic_ev_target"] = day_target
            out.loc[current_index, "threshold_basis_dynamic_score_threshold"] = (
                day_threshold
            )
        # Update reachable-EV targets only after the complete UTC day. This is
        # deliberately stricter than row-level timestamp filtering.
        day_rows = out.iloc[start:stop]
        baseline = day_rows.loc[
            pd.to_numeric(day_rows["policy_baseline_rank"], errors="coerce").ge(0.90)
            & pd.to_numeric(day_rows[return_col], errors="coerce").notna()
        ]
        if not baseline.empty:
            values = pd.to_numeric(baseline[return_col], errors="coerce")
            global_sum += float(values.sum())
            global_count += int(values.count())
            for archetype, sub in baseline.groupby(
                "archetype_policy_key", dropna=False
            ):
                key = str(archetype)
                local_values = pd.to_numeric(sub[return_col], errors="coerce")
                arch_sum[key] = arch_sum.get(key, 0.0) + float(local_values.sum())
                arch_count[key] = arch_count.get(key, 0) + int(local_values.count())
        if day_number == 1 or day_number % 30 == 0 or day_number == total_days:
            print(
                json.dumps(
                    {
                        "event": "threshold_policy_progress",
                        "days_complete": day_number,
                        "days_total": total_days,
                        "day": str(pd.Timestamp(day_value, unit="ns", tz="UTC").date()),
                        "selected_so_far": int(out["threshold_basis_selected"].sum()),
                    }
                ),
                flush=True,
            )
    out["selected_for_monitor"] = out["threshold_basis_selected"].astype(bool)
    out["historical_rank"] = out["threshold_basis_rank_score"].astype(np.float32)
    # July scorer artifacts already contain the exact promoted-policy decision.
    # Preserve those values instead of rebuilding their rank from a score stream
    # whose scale differs from the historical backcast.
    if preserve_materialized_policy and bool(exact_policy_mask.any()):
        exact_selected = (
            out.loc[exact_policy_mask, "threshold_alternative_selected"]
            .fillna(False)
            .astype(bool)
        )
        exact_rank = pd.to_numeric(
            out.loc[exact_policy_mask, "threshold_alternative_rank"],
            errors="coerce",
        ).fillna(0.0)
        out.loc[exact_policy_mask, "threshold_basis_selected"] = (
            exact_selected.to_numpy()
        )
        out.loc[exact_policy_mask, "threshold_basis_rank_score"] = exact_rank.to_numpy(
            dtype=np.float32, copy=False
        )
        for target, source in (
            (
                "threshold_basis_dynamic_ev_target",
                "threshold_alternative_dynamic_ev_target",
            ),
            (
                "threshold_basis_dynamic_score_threshold",
                "threshold_alternative_dynamic_score_threshold",
            ),
            (
                "threshold_basis_recent_reference_rows",
                "threshold_alternative_recent_reference_rows",
            ),
            (
                "threshold_basis_baseline_activity_count",
                "threshold_alternative_baseline_activity_count",
            ),
        ):
            if source in out.columns:
                out.loc[exact_policy_mask, target] = out.loc[
                    exact_policy_mask, source
                ].to_numpy(copy=False)
        out.loc[exact_policy_mask, "selected_for_monitor"] = exact_selected.to_numpy()
        out.loc[exact_policy_mask, "historical_rank"] = exact_rank.to_numpy(
            dtype=np.float32, copy=False
        )
    out["monitor_selection_contract"] = str(
        policy.get("policy_id")
        or (
            "ev_target_archetype_reachable_match_current_activity_8d_hr_off_regimecal_v1"
            if score_col == "score_regime_calibrated"
            else "ev_target_archetype_reachable_match_current_activity_8d_hr_off"
        )
    )
    return out


def _materialize_policy_selection(
    ledger_path: Path,
    *,
    candidate_dir: Path,
    feature_root: Path,
    threshold_policy_path: Path,
    regime_calibration_path: Path,
) -> dict[str, Any]:
    frame = pd.read_parquet(ledger_path)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    regime_cache_path = ledger_path.parent / "frozen_regime_calibrated_scores.parquet"
    regime_cols = [
        "score_regime_calibrated",
        "regime_ev_risk_score",
        "regime_ev_effect_count",
    ]
    if regime_cache_path.exists():
        cached = pd.read_parquet(regime_cache_path)
        frame = frame.drop(columns=regime_cols, errors="ignore").merge(
            cached,
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        frame["score_regime_calibrated"] = pd.to_numeric(
            frame["score_regime_calibrated"], errors="coerce"
        ).fillna(pd.to_numeric(frame["score_shock_adjusted"], errors="coerce"))
    else:
        frame = _append_regime_calibrated_scores(
            frame,
            candidate_dir=candidate_dir,
            feature_root=feature_root,
            artifact_path=regime_calibration_path,
        )
        frame[["row_id", *regime_cols]].to_parquet(
            regime_cache_path, index=False, compression="zstd"
        )
    policy = _read_json(threshold_policy_path)
    if bool(policy.get("hr_rank50", True)):
        raise ValueError("The requested threshold policy must keep HR rank50 disabled.")
    frame = _apply_causal_reachable_ev_policy(frame, policy=policy)
    frame.to_parquet(ledger_path, index=False, compression="zstd")
    return {
        "policy_id": policy.get("policy_id"),
        "selected_rows": int(frame["selected_for_monitor"].sum()),
        "candidate_rows": int(len(frame)),
        "selection_rate": float(frame["selected_for_monitor"].mean()),
        "regime_adjusted_rows": int(
            pd.to_numeric(frame["regime_ev_effect_count"], errors="coerce").gt(0).sum()
        ),
    }


def _write_reports(ledger_path: Path, output_dir: Path) -> dict[str, Any]:
    frame = pd.read_parquet(ledger_path)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["selected_for_monitor"] = (
        frame["selected_for_monitor"].fillna(False).astype(bool)
    )
    if "monitor_selection_contract" not in frame.columns:
        frame["monitor_selection_contract"] = (
            "ev_target_archetype_reachable_match_current_activity_8d_"
            "hr_off_regimecal_v1"
        )
    frame["evidence_phase"] = np.where(
        frame["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC")),
        "retrospective_frozen_backcast",
        "frozen_postfit_oos",
    )
    daily = _daily_cells(frame)
    daily["coverage_status"] = "covered"
    significant = daily.loc[daily["significant"].fillna(False).astype(bool)].copy()
    autocorr = _autocorrelation_rows(daily) if not daily.empty else pd.DataFrame()
    autocorr_phase = (
        _autocorrelation_rows(daily, group_by_phase=True)
        if not daily.empty
        else pd.DataFrame()
    )
    daily.to_csv(output_dir / "daily_surprise_calendar_all_cells.csv", index=False)
    significant.to_csv(output_dir / "significant_surprise_calendar.csv", index=False)
    autocorr.to_csv(
        output_dir / "side_archetype_daily_autocorrelation.csv", index=False
    )
    autocorr_phase.to_csv(
        output_dir / "side_archetype_daily_autocorrelation_by_evidence.csv", index=False
    )
    selected = frame.loc[
        frame["selected_for_monitor"] & frame["ev_after_1pct"].notna()
    ].copy()
    selected["month"] = selected["__ts__"].dt.strftime("%Y-%m")
    selected["week_start"] = selected["__ts__"].dt.floor("D") - pd.to_timedelta(
        selected["__ts__"].dt.weekday.to_numpy(), unit="D"
    )
    month = (
        selected.groupby("month", observed=True)
        .agg(
            selected_rows=("__ts__", "size"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            clean_exec_precision=("clean_exec", "mean"),
            full_path_bad_mae_rate=("full_path_bad_mae_1r", "mean"),
            timeout_rate=("timeout", "mean"),
        )
        .reset_index()
    )
    week = (
        selected.groupby("week_start", observed=True)
        .agg(
            selected_rows=("__ts__", "size"),
            mean_ev_after_1pct=("ev_after_1pct", "mean"),
            clean_exec_precision=("clean_exec", "mean"),
        )
        .reset_index()
    )
    month.to_csv(output_dir / "month_metrics.csv", index=False)
    week.to_csv(output_dir / "week_metrics.csv", index=False)
    coverage = (
        frame.assign(day=frame["__ts__"].dt.floor("D"))
        .groupby("day", observed=True)
        .agg(
            candidate_rows=("__ts__", "size"),
            timestamps=("__ts__", "nunique"),
            symbols=("__symbol__", "nunique"),
            outcome_rows=("ev_after_1pct", "count"),
        )
        .reset_index()
    )
    coverage.to_csv(output_dir / "daily_coverage.csv", index=False)
    return {
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "mean_ev_after_1pct": float(selected["ev_after_1pct"].mean()),
        "clean_exec_precision": float(selected["clean_exec"].mean()),
        "worst_week_ev": float(week["mean_ev_after_1pct"].min()),
        "worst_month_ev": float(month["mean_ev_after_1pct"].min()),
        "significant_calendar_rows": int(len(significant)),
        "coverage_days": int(coverage["day"].nunique()),
        "min_timestamp": str(frame["__ts__"].min()),
        "max_timestamp": str(frame["__ts__"].max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--frozen-root", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--base-reference", type=Path, default=DEFAULT_BASE_REFERENCE)
    parser.add_argument("--july-early", type=Path, default=DEFAULT_JULY_EARLY)
    parser.add_argument("--july-complete", type=Path, default=DEFAULT_JULY_COMPLETE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold-policy", type=Path, default=DEFAULT_THRESHOLD_POLICY
    )
    parser.add_argument(
        "--regime-calibration", type=Path, default=DEFAULT_REGIME_CALIBRATION
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "candidate_shards"
    prediction_dir = args.output_dir / "prediction_shards"
    candidate_dir.mkdir(exist_ok=True)
    prediction_dir.mkdir(exist_ok=True)
    base_dir = args.frozen_root / "models/base/2026-07-01_2026-07-16"
    base_contract = _read_json(base_dir / "columns.json")
    base_columns = list(base_contract["feature_names"])
    base_model = joblib.load(base_dir / "base_model.joblib")
    with (args.bundle_dir / "base_ae_gmm_state.pkl").open("rb") as handle:
        base_state = pickle.load(handle)
    bundle = joblib.load(
        args.bundle_dir / "alternative_meta_residual_pca8_shock_bundle.joblib"
    )
    base_reference = pd.read_parquet(
        args.base_reference, columns=["score", "selected_top30"]
    )
    base_cutoff = float(
        pd.to_numeric(
            base_reference.loc[base_reference["selected_top30"].astype(bool), "score"],
            errors="coerce",
        ).min()
    )
    source_edges = _read_json(args.source_manifest)["source_contract"]["edges"]
    months = [str(period) for period in pd.period_range("2025-01", "2026-06", freq="M")]
    preparation: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    for month in months:
        path = candidate_dir / f"candidates_{month}.parquet"
        candidate_paths.append(path)
        if args.force or not path.exists():
            summary = _prepare_month_candidates(
                month=month,
                labels_dir=args.labels_dir,
                feature_root=args.feature_root,
                base_columns=base_columns,
                base_model=base_model,
                base_state=base_state,
                bundle=bundle,
                base_cutoff=base_cutoff,
                source_edges=source_edges,
                output=path,
            )
        else:
            summary = {"month": month, "status": "candidate_cache_hit"}
        preparation.append(summary)
        print(json.dumps({"event": "frozen_candidate_month", **summary}), flush=True)
    reference_feature_path = args.output_dir / "frozen_reference_features.parquet"
    if args.force or not reference_feature_path.exists():
        reference_columns = _reference_features(
            args.reference, candidate_paths, reference_feature_path
        )
    else:
        reference_columns = [
            name for name in _columns(reference_feature_path) if name != "row_id"
        ]
    reference_features = pd.read_parquet(reference_feature_path)
    scoring: list[dict[str, Any]] = []
    prediction_paths: list[Path] = []
    for month, candidate_path in zip(months, candidate_paths, strict=True):
        path = prediction_dir / f"predictions_{month}.parquet"
        prediction_paths.append(path)
        if args.force or not path.exists():
            summary = _score_month(
                candidate_path=candidate_path,
                reference_features=reference_features,
                bundle=bundle,
                output=path,
            )
        else:
            summary = {"month": month, "status": "prediction_cache_hit"}
        scoring.append(summary)
        print(json.dumps({"event": "frozen_prediction_month", **summary}), flush=True)
    july_path = prediction_dir / "predictions_2026-07.parquet"
    july_summary = _july_uniform(args.july_early, args.july_complete, july_path)
    prediction_paths.append(july_path)
    scoring.append(july_summary)
    ledger_path = args.output_dir / "frozen_champion_single_source_ledger.parquet"
    union_glob = (prediction_dir / "predictions_*.parquet").as_posix()
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{union_glob}', union_by_name=true) "
        f"WHERE CAST(__ts__ AS TIMESTAMPTZ) >= TIMESTAMPTZ '2025-01-01' "
        f"AND CAST(__ts__ AS TIMESTAMPTZ) < TIMESTAMPTZ '2026-07-11') "
        f"TO '{ledger_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()
    policy_selection = _materialize_policy_selection(
        ledger_path,
        candidate_dir=candidate_dir,
        feature_root=args.feature_root,
        threshold_policy_path=args.threshold_policy,
        regime_calibration_path=args.regime_calibration,
    )
    metrics = _write_reports(ledger_path, args.output_dir)
    manifest = {
        "schema": "frozen_champion_single_source_history_v1",
        "model_contract": "one frozen July-fitted base model plus sparse-shock champion meta bundle",
        "base_feature_contract_hash": base_contract.get("feature_contract_hash"),
        "base_cutoff": base_cutoff,
        "base_ae_gmm_state": str(args.bundle_dir / "base_ae_gmm_state.pkl"),
        "champion_bundle": str(
            args.bundle_dir / "alternative_meta_residual_pca8_shock_bundle.joblib"
        ),
        "historical_rank_contract": (
            "causal prior-8d global rank feeding the promoted reachable-EV "
            "side/archetype threshold; activity matched per timestamp"
        ),
        "threshold_basis_policy": policy_selection,
        "regime_calibration_artifact": str(args.regime_calibration),
        "cost_contract": "ev_after_1pct = materialized trailing capture net minus 1% comparison floor",
        "period": "2025-01-01 through 2026-07-10",
        "fit_cutoff": bundle.fit_through,
        "evidence_contract": {
            "2025-01-01_through_2026-06-30": "retrospective fixed-model backcast; comparable but not OOS",
            "2026-07-01_through_2026-07-10": "frozen post-fit OOS",
        },
        "reference_feature_columns": reference_columns,
        "preparation": preparation,
        "scoring": scoring,
        "metrics": metrics,
        "outputs": {
            "ledger": str(ledger_path),
            "autocorrelation": str(
                args.output_dir / "side_archetype_daily_autocorrelation.csv"
            ),
            "calendar": str(args.output_dir / "daily_surprise_calendar_all_cells.csv"),
            "significant_calendar": str(
                args.output_dir / "significant_surprise_calendar.csv"
            ),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_json_safe(manifest), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
