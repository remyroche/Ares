#!/usr/bin/env python3
"""Run the fixed 4m->6m base and 3m->3m residual label ablation.

The job is intentionally separate from production promotion runners.  It
reuses the frozen per-side Pack-B feature/model contracts, but fits every base
challenger on Sep-Dec 2025 only, scores Jan-Jun 2026 once, develops the
residual and label recipe on Jan-Mar only, and reports Apr-Jun untouched.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_residual_label_ablation import (  # noqa: E402
    CALIBRATION_DAYS,
    ROUND_TRIP_COST,
    SCHEMA,
    FixedWindowCalendar,
    LabelRecipe,
    build_soft_label,
    default_label_recipes,
    economic_metrics,
    label_components,
    label_hpo_objective,
    rank_mask,
)
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from extreme_price_movements.universe import (  # noqa: E402
    _normalize_symbol,
    load_spread_cost_excluded_symbols,
)
from scripts.run_packb_side_local_residual_oof import (  # noqa: E402
    DEFAULT_AE_ROOT,
    _side_loader,
)

DEFAULT_LABELS = (
    ROOT / "data_perp/artifacts/"
    "20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
DEFAULT_PATH_LABELS = (
    ROOT / "data_perp/artifacts/"
    "20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels"
)
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_BASE_CONTRACT = (
    ROOT / "data_perp/artifacts/packb_side_local_outer_oof_20260724_v1_31_8"
)
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/base_residual_label_ablation_20260725_v1"
TARGET = "__first_touch_target_soft__"
WEIGHT = "__w__"
ECONOMIC = "__first_touch_capture_net__"
SIDES = ("long", "short")
PATH_COLUMNS = (
    "__peak_mfe_atr_12h__",
    "__time_to_first_meaningful_mfe_hours_12h__",
    "__mae_before_meaningful_mfe_atr_12h__",
    "__future_slope_atr_per_hour_12h__",
    "__mfe_2h_over_mfe_12h__",
    "__bars_to_80pct_peak__",
    "__mfe_before_60m_atr__",
    "__adverse_trough_within_60m__",
    "__adverse_trough_within_120m__",
    "__meaningful_mfe_reached_12h__",
    "__path_auxiliary_target_valid__",
)


def spread_exclusion_mask(
    symbols: pd.Series, excluded_symbols: set[str]
) -> np.ndarray:
    """Apply the exact inference-universe symbol normalization before matching."""

    normalized_excluded = {_normalize_symbol(value) for value in excluded_symbols}
    normalized = symbols.astype(str).map(_normalize_symbol)
    return normalized.isin(normalized_excluded).to_numpy(dtype=bool)


class AblationError(RuntimeError):
    pass


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_safe(dict(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_files(root: Path, side: str) -> list[Path]:
    files = sorted(root.glob(f"train_global_{side}_5_*.parquet"))
    if not files:
        raise AblationError(f"no canonical {side} labels found under {root}")
    return files


def _path_file(root: Path, side: str) -> Path:
    path = root / f"train_global_{side}_3.parquet"
    if not path.is_file():
        raise AblationError(f"missing path-label file: {path}")
    return path


def _load_side_labels(
    *,
    labels_root: Path,
    path_labels_root: Path,
    side: str,
    calendar: FixedWindowCalendar,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import duckdb

    files = _canonical_files(labels_root, side)
    path_file = _path_file(path_labels_root, side)
    projected = ",\n".join(f"a.{column}" for column in PATH_COLUMNS)
    connection = duckdb.connect(database=":memory:")
    try:
        frame = connection.execute(
            f"""
            SELECT
                b.candidate_id,
                lower(b.side_name) AS side_name,
                b.__symbol__,
                b.__ts__,
                b.{TARGET},
                b.{WEIGHT},
                b.{ECONOMIC},
                {projected}
            FROM read_parquet(?, union_by_name=true) AS b
            INNER JOIN read_parquet(?) AS a USING (candidate_id)
            WHERE cast(b.__ts__ AS TIMESTAMP) >= cast(? AS TIMESTAMP)
              AND cast(b.__ts__ AS TIMESTAMP) < cast(? AS TIMESTAMP)
              AND lower(b.side_name) = ?
              AND a.__path_auxiliary_target_valid__ = 1
            ORDER BY b.__ts__, b.__symbol__, b.candidate_id
            """,
            [
                list(map(str, files)),
                str(path_file),
                calendar.base_train_start.tz_localize(None).to_pydatetime(),
                calendar.base_oos_end.tz_localize(None).to_pydatetime(),
                side,
            ],
        ).fetchdf()
        total = connection.execute(
            """
            SELECT count(*)
            FROM read_parquet(?, union_by_name=true)
            WHERE cast(__ts__ AS TIMESTAMP) >= cast(? AS TIMESTAMP)
              AND cast(__ts__ AS TIMESTAMP) < cast(? AS TIMESTAMP)
              AND lower(side_name) = ?
            """,
            [
                list(map(str, files)),
                calendar.base_train_start.tz_localize(None).to_pydatetime(),
                calendar.base_oos_end.tz_localize(None).to_pydatetime(),
                side,
            ],
        ).fetchone()[0]
    finally:
        connection.close()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise AblationError(f"{side} joined label rows are not unique")
    for column in (TARGET, WEIGHT, ECONOMIC, *PATH_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = frame.loc[:, [TARGET, WEIGHT, ECONOMIC, *PATH_COLUMNS]]
    finite = np.isfinite(required.to_numpy(np.float64)).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)
    if len(frame) < 100_000:
        raise AblationError(f"{side} has insufficient paired path-label support")
    return frame, {
        "canonical_rows_in_calendar": int(total),
        "joined_valid_path_rows": int(len(frame)),
        "paired_support_fraction": float(len(frame) / max(int(total), 1)),
        "canonical_file_count": len(files),
        "path_labels": str(path_file),
        "path_labels_sha256": _sha256(path_file),
        "paired_comparison_policy": (
            "all arms use the same rows with complete exact 12h path targets"
        ),
    }


def _contract(root: Path, side: str) -> tuple[list[str], dict[str, Any]]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    side_summary = summary["sides"][side]
    features = list(map(str, side_summary["features"]))
    params = dict(side_summary["parameters"])
    return features, params


def _time_spread_indices(
    frame: pd.DataFrame, mask: np.ndarray, maximum: int
) -> np.ndarray:
    positions = np.flatnonzero(mask)
    if maximum <= 0 or len(positions) <= maximum:
        return positions
    local = np.linspace(0, len(positions) - 1, num=int(maximum), dtype=np.int64)
    return positions[np.unique(local)]


def _load_feature_matrix(
    loader: Any,
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    guard: TrainingResourceGuard,
    side: str,
    batch_rows: int = 150_000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    keys = ("candidate_id", "side_name", "__symbol__", "__ts__")
    for batch_index, start in enumerate(range(0, len(frame), int(batch_rows))):
        end = min(start + int(batch_rows), len(frame))
        guard.checkpoint(
            f"base_residual_label_ablation:{side}:feature_batch_{batch_index:03d}"
        )
        part = loader(frame.iloc[start:end].loc[:, list(keys)], features)
        parts.append(part.reset_index(drop=True))
    matrix = pd.concat(parts, ignore_index=True, copy=False)
    if len(matrix) != len(frame) or list(matrix.columns) != list(features):
        raise AblationError(
            f"{side} chunked feature loading changed row/column contract"
        )
    return matrix


def _fit_base(
    x: pd.DataFrame,
    target: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    params: Mapping[str, Any],
    *,
    seed: int,
) -> lgb.LGBMRegressor:
    model_params = {
        **dict(params),
        "objective": "regression",
        "verbosity": -1,
        "random_state": int(seed),
        "n_jobs": 6,
    }
    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        x.iloc[indices],
        target[indices],
        sample_weight=weights[indices],
        categorical_feature="auto",
    )
    return model


def _base_rank(frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    work = pd.DataFrame(
        {
            "position": np.arange(len(frame)),
            "ts": frame["__ts__"].to_numpy(),
            "score": prediction,
            "symbol": frame["__symbol__"].astype(str).to_numpy(),
        }
    ).sort_values(
        ["ts", "score", "symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    grouped = work.groupby("ts", sort=False)
    work["rank"] = grouped.cumcount() + 1
    work["rows"] = grouped["position"].transform("size")
    work["rank_pct"] = work["rank"] / work["rows"]
    result = np.empty(len(frame), dtype=np.float32)
    result[work["position"].to_numpy(int)] = work["rank_pct"].to_numpy(np.float32)
    return result


def _fit_ev_map(
    score: np.ndarray, economic: np.ndarray, weight: np.ndarray
) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(score, economic, sample_weight=weight)
    return model


def _fit_residual(
    matrix: pd.DataFrame,
    prediction: np.ndarray,
    rank_pct: np.ndarray,
    economic: np.ndarray,
    weights: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
    max_rows: int,
) -> tuple[IsotonicRegression, lgb.LGBMRegressor]:
    if len(train_indices) > max_rows > 0:
        local = np.linspace(
            0, len(train_indices) - 1, num=int(max_rows), dtype=np.int64
        )
        train_indices = train_indices[np.unique(local)]
    ev_map = _fit_ev_map(
        prediction[train_indices], economic[train_indices], weights[train_indices]
    )
    base_ev = ev_map.predict(prediction[train_indices])
    residual_x = matrix.iloc[train_indices].copy()
    residual_x["base_prediction"] = prediction[train_indices]
    residual_x["base_rank_pct_timestamp_side"] = rank_pct[train_indices]
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=5,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.25,
        reg_lambda=3.0,
        verbosity=-1,
        random_state=int(seed),
        n_jobs=6,
    )
    model.fit(
        residual_x,
        economic[train_indices] - base_ev,
        sample_weight=weights[train_indices],
    )
    return ev_map, model


def _residual_predict(
    ev_map: IsotonicRegression,
    model: lgb.LGBMRegressor,
    matrix: pd.DataFrame,
    prediction: np.ndarray,
    rank_pct: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    residual_x = matrix.iloc[indices].copy()
    residual_x["base_prediction"] = prediction[indices]
    residual_x["base_rank_pct_timestamp_side"] = rank_pct[indices]
    return ev_map.predict(prediction[indices]) + model.predict(residual_x)


def _residual_oof(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    prediction: np.ndarray,
    *,
    seed: int,
    max_train_rows: int,
) -> pd.DataFrame:
    rank_pct = _base_rank(frame, prediction)
    economics = frame[ECONOMIC].to_numpy(np.float64)
    weights = frame[WEIGHT].to_numpy(np.float64)
    top40 = rank_mask(frame, prediction, fraction=0.40, scope="timestamp_side")
    outputs: list[pd.DataFrame] = []
    for fold_index, (start, end) in enumerate(
        (("2026-02-01", "2026-03-01"), ("2026-03-01", "2026-04-01"))
    ):
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        train = np.flatnonzero(
            top40
            & frame["__ts__"].lt(start_ts).to_numpy()
            & frame["__ts__"].ge(pd.Timestamp("2026-01-01", tz="UTC")).to_numpy()
        )
        valid = np.flatnonzero(
            top40
            & frame["__ts__"].ge(start_ts).to_numpy()
            & frame["__ts__"].lt(end_ts).to_numpy()
        )
        ev_map, model = _fit_residual(
            matrix,
            prediction,
            rank_pct,
            economics,
            weights,
            train,
            seed=seed + fold_index,
            max_rows=max_train_rows,
        )
        score = _residual_predict(ev_map, model, matrix, prediction, rank_pct, valid)
        part = (
            frame.iloc[valid]
            .loc[
                :,
                ["candidate_id", "side_name", "__ts__", "__symbol__", ECONOMIC, WEIGHT],
            ]
            .copy()
        )
        part["residual_score"] = score
        part["fold"] = f"{start}/{end}"
        outputs.append(part)
        del model, ev_map
        gc.collect()
    return pd.concat(outputs, ignore_index=True)


def _fit_admission_calibrator(
    oof: pd.DataFrame,
) -> tuple[IsotonicRegression, pd.Timestamp, dict[str, Any]]:
    ordered_days = (
        pd.to_datetime(oof["__ts__"], utc=True)
        .dt.floor("D")
        .drop_duplicates()
        .sort_values()
    )
    if len(ordered_days) < CALIBRATION_DAYS + 7:
        raise AblationError(
            "residual OOF has insufficient days for admission calibration"
        )
    cutoff = pd.Timestamp(ordered_days.iloc[CALIBRATION_DAYS - 1]) + pd.Timedelta(
        days=1
    )
    fit_mask = pd.to_datetime(oof["__ts__"], utc=True).lt(cutoff)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(
        oof.loc[fit_mask, "residual_score"],
        oof.loc[fit_mask, ECONOMIC],
        sample_weight=oof.loc[fit_mask, WEIGHT],
    )
    return (
        calibrator,
        cutoff,
        {
            "fit_days": int(CALIBRATION_DAYS),
            "fit_rows": int(fit_mask.sum()),
            "fit_end_exclusive": cutoff.isoformat(),
            "source": "residual OOF only",
            "admission_rule": "calibrated_expected_net_return_gt_0",
        },
    )


def _month_metrics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    start: pd.Timestamp | None = None,
    admitted: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    local = frame.copy()
    if start is not None:
        local = local.loc[local["__ts__"].ge(start)].copy()
        if admitted is not None:
            admitted = np.asarray(admitted)[local.index.to_numpy()]
    local["_month"] = local["__ts__"].dt.strftime("%Y-%m")
    records = []
    for month, part in local.groupby("_month", sort=True):
        metrics = economic_metrics(
            part.reset_index(drop=True),
            part[score_column].to_numpy(),
            admitted=(
                admitted[part.index.to_numpy()] if admitted is not None else None
            ),
        )
        records.append({"month": month, **metrics})
    return records


def _evaluate_recipe(
    *,
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    components: pd.DataFrame,
    recipe: LabelRecipe,
    params: Mapping[str, Any],
    train_indices: np.ndarray,
    oos_indices: np.ndarray,
    base_train_rows: int,
    residual_train_rows: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame, IsotonicRegression]:
    target, hard = build_soft_label(components, recipe)
    model = _fit_base(
        matrix,
        target,
        frame[WEIGHT].to_numpy(np.float64),
        _time_spread_indices(
            frame, np.isin(np.arange(len(frame)), train_indices), base_train_rows
        ),
        params,
        seed=seed,
    )
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    prediction[oos_indices] = model.predict(matrix.iloc[oos_indices])
    oos_frame = frame.iloc[oos_indices].reset_index(drop=True)
    oos_matrix = matrix.iloc[oos_indices].reset_index(drop=True)
    oos_prediction = prediction[oos_indices]
    residual_oof = _residual_oof(
        oos_frame,
        oos_matrix,
        oos_prediction,
        seed=seed + 100,
        max_train_rows=residual_train_rows,
    )
    calibrator, calibration_cutoff, calibration_contract = _fit_admission_calibrator(
        residual_oof
    )
    residual_oof["calibrated_ev"] = calibrator.predict(residual_oof["residual_score"])
    post_mask = residual_oof["__ts__"].ge(calibration_cutoff)
    promotion = residual_oof.loc[post_mask].reset_index(drop=True)
    raw_by_month = _month_metrics(promotion, "residual_score")
    admitted_by_month = []
    for month, part in promotion.assign(
        _month=promotion["__ts__"].dt.strftime("%Y-%m")
    ).groupby("_month", sort=True):
        admitted = part["calibrated_ev"].to_numpy() > 0.0
        admitted_by_month.append(
            {
                "month": month,
                **economic_metrics(
                    part.reset_index(drop=True),
                    part["residual_score"].to_numpy(),
                    admitted=admitted,
                ),
            }
        )
    objective = label_hpo_objective(raw_by_month)
    admitted_values = [
        row["global_top10_mean_net_return"]
        for row in admitted_by_month
        if row["global_top10_rows"] > 0
    ]
    if admitted_values:
        objective += 0.20 * float(np.median(admitted_values))
    report = {
        "recipe": recipe.manifest(),
        "objective": float(objective),
        "base_train_rows": int(len(train_indices)),
        "base_train_rows_sampled": int(
            min(len(train_indices), base_train_rows)
            if base_train_rows > 0
            else len(train_indices)
        ),
        "base_oos_rows": int(len(oos_indices)),
        "soft_target_mean_train": float(np.mean(target[train_indices])),
        "hard_target_rate_train": float(np.mean(hard[train_indices])),
        "raw_residual_oof_by_month": raw_by_month,
        "post_21d_admission_by_month": admitted_by_month,
        "calibration": calibration_contract,
        "promotion_uses_only": "2026-02 through 2026-03 residual OOF; final OOS untouched",
    }
    del model
    gc.collect()
    return report, prediction, target, residual_oof, calibrator


def _final_evaluation(
    *,
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    base_prediction: np.ndarray,
    calibrator: IsotonicRegression,
    residual_train_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], tuple[IsotonicRegression, lgb.LGBMRegressor]]:
    rank_pct = _base_rank(frame, base_prediction)
    top40 = rank_mask(frame, base_prediction, fraction=0.40, scope="timestamp_side")
    meta_train = np.flatnonzero(
        top40
        & frame["__ts__"].ge(pd.Timestamp("2026-01-01", tz="UTC")).to_numpy()
        & frame["__ts__"].lt(pd.Timestamp("2026-04-01", tz="UTC")).to_numpy()
    )
    meta_oos = np.flatnonzero(
        top40
        & frame["__ts__"].ge(pd.Timestamp("2026-04-01", tz="UTC")).to_numpy()
        & frame["__ts__"].lt(pd.Timestamp("2026-07-01", tz="UTC")).to_numpy()
    )
    economics = frame[ECONOMIC].to_numpy(np.float64)
    weights = frame[WEIGHT].to_numpy(np.float64)
    ev_map, model = _fit_residual(
        matrix,
        base_prediction,
        rank_pct,
        economics,
        weights,
        meta_train,
        seed=seed,
        max_rows=residual_train_rows,
    )
    score = _residual_predict(
        ev_map, model, matrix, base_prediction, rank_pct, meta_oos
    )
    scored = (
        frame.iloc[meta_oos]
        .loc[:, ["candidate_id", "side_name", "__ts__", "__symbol__", ECONOMIC, WEIGHT]]
        .copy()
    )
    scored["base_prediction"] = base_prediction[meta_oos]
    scored["base_rank_pct_timestamp_side"] = rank_pct[meta_oos]
    scored["residual_score"] = score
    scored["calibrated_ev"] = calibrator.predict(score)
    scored["admitted_after_21d_calibrator"] = scored["calibrated_ev"] > 0.0
    raw = economic_metrics(scored, scored["residual_score"])
    admitted = economic_metrics(
        scored,
        scored["residual_score"],
        admitted=scored["admitted_after_21d_calibrator"],
    )
    by_month = []
    for month, part in scored.assign(
        _month=scored["__ts__"].dt.strftime("%Y-%m")
    ).groupby("_month", sort=True):
        by_month.append(
            {
                "month": month,
                "raw": economic_metrics(part, part["residual_score"]),
                "post_21d_admission": economic_metrics(
                    part,
                    part["residual_score"],
                    admitted=part["admitted_after_21d_calibrator"],
                ),
            }
        )
    return (
        scored,
        {
            "raw": raw,
            "post_21d_admission": admitted,
            "by_month": by_month,
            "meta_train_rows": int(len(meta_train)),
            "meta_oos_rows": int(len(meta_oos)),
            "top40_handoff": True,
            "cost": "1% already embedded in __first_touch_capture_net__; not subtracted again",
        },
        (ev_map, model),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    stage = args.output.parent / f".{args.output.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    calendar = FixedWindowCalendar.from_first_oos_month("2026-01")
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=stage,
        telemetry_path=stage / "training_resource_telemetry.jsonl",
    )
    guard.preflight("base_residual_label_ablation:preflight")
    spread_excluded = load_spread_cost_excluded_symbols()
    recipes = default_label_recipes(args.seed)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "calendar": calendar.manifest(),
        "round_trip_cost": ROUND_TRIP_COST,
        "feature_selection_exception": (
            "reuses the approved current 31/8 per-side feature contract; label "
            "promotion remains confined to Jan-Mar and Apr-Jun is untouched"
        ),
        "label_hpo": {
            "trials": len(recipes),
            "side_local": True,
            "side_is_not_a_search_axis": True,
            "selection_period": "residual OOF after the first 21 calibration days, ending 2026-04-01",
            "selection_policy": "one pooled-global top-10 after the 21-day admission calibrator; timestamp-side top-k is diagnostic only",
        },
        "spread_exclusion": {
            "threshold_bps": 70.0,
            "current_excluded_symbols": len(spread_excluded),
            "classification": (
                "diagnostic_non_PIT: current baseline was built Jun-Jul 2026 and "
                "cannot support a causal Sep-2025 training exclusion claim"
            ),
        },
        "sides": {},
    }
    all_scored: list[pd.DataFrame] = []
    named_scored: dict[str, list[pd.DataFrame]] = {
        "baseline_24h": [],
        "timeout_12h": [],
        "time_aware_12h": [],
    }
    try:
        for side_index, side in enumerate(SIDES):
            guard.checkpoint(f"base_residual_label_ablation:{side}:labels")
            frame, label_evidence = _load_side_labels(
                labels_root=args.labels,
                path_labels_root=args.path_labels,
                side=side,
                calendar=calendar,
            )
            masks = calendar.masks(frame["__ts__"])
            features, params = _contract(args.base_contract, side)
            loader, candidates, loader_evidence = _side_loader(
                side=side,
                ae_root=args.ae_root,
                feature_store=args.feature_store,
                guard=guard,
            )
            missing = sorted(set(features) - set(candidates))
            if missing:
                raise AblationError(f"{side} feature loader misses {missing}")
            guard.checkpoint(f"base_residual_label_ablation:{side}:features")
            matrix = _load_feature_matrix(
                loader,
                frame,
                features,
                guard=guard,
                side=side,
            )
            complete = np.isfinite(matrix.to_numpy(np.float32)).all(axis=1)
            frame = frame.loc[complete].reset_index(drop=True)
            matrix = matrix.loc[complete].reset_index(drop=True)
            masks = calendar.masks(frame["__ts__"])
            train_indices = np.flatnonzero(masks["base_train"])
            oos_indices = np.flatnonzero(masks["base_oos"])
            if len(train_indices) < 25_000 or len(oos_indices) < 50_000:
                raise AblationError(f"{side} has insufficient complete model rows")
            components = label_components(frame)
            trial_reports = []
            trial_state: dict[
                str, tuple[np.ndarray, np.ndarray, pd.DataFrame, IsotonicRegression]
            ] = {}
            for trial_index, recipe in enumerate(recipes):
                guard.checkpoint(
                    f"base_residual_label_ablation:{side}:{recipe.recipe_id}:start"
                )
                report, prediction, target, residual_oof, calibrator = _evaluate_recipe(
                    frame=frame,
                    matrix=matrix,
                    components=components,
                    recipe=recipe,
                    params=params,
                    train_indices=train_indices,
                    oos_indices=oos_indices,
                    base_train_rows=args.base_train_rows,
                    residual_train_rows=args.residual_train_rows,
                    seed=args.seed + side_index * 1000 + trial_index * 20,
                )
                trial_reports.append(report)
                trial_state[recipe.recipe_id] = (
                    prediction,
                    target,
                    residual_oof,
                    calibrator,
                )
                guard.checkpoint(
                    f"base_residual_label_ablation:{side}:{recipe.recipe_id}:complete"
                )
            ordered = sorted(
                trial_reports,
                key=lambda item: (
                    float(item["objective"]),
                    item["recipe"]["recipe_id"],
                ),
                reverse=True,
            )
            winner_id = str(ordered[0]["recipe"]["recipe_id"])
            winner_prediction, winner_target, winner_oof, winner_calibrator = (
                trial_state[winner_id]
            )
            side_root = stage / side
            side_root.mkdir(parents=True)
            comparison_ids = list(
                dict.fromkeys(
                    ["baseline_24h", "timeout_12h", "time_aware_12h", winner_id]
                )
            )
            comparison_metrics: dict[str, Any] = {}
            comparison_models: dict[
                str, tuple[IsotonicRegression, lgb.LGBMRegressor]
            ] = {}
            comparison_predictions: dict[str, pd.DataFrame] = {}
            for comparison_index, comparison_id in enumerate(comparison_ids):
                base_prediction, _, _, calibrator = trial_state[comparison_id]
                scored_arm, metrics_arm, models_arm = _final_evaluation(
                    frame=frame,
                    matrix=matrix,
                    base_prediction=base_prediction,
                    calibrator=calibrator,
                    residual_train_rows=args.residual_train_rows,
                    seed=(args.seed + side_index * 1000 + 900 + comparison_index * 10),
                )
                scored_arm["label_recipe"] = comparison_id
                scored_arm["asset_exclusion_arm"] = "all_current_training_universe"
                comparison_metrics[comparison_id] = metrics_arm
                comparison_models[comparison_id] = models_arm
                comparison_predictions[comparison_id] = scored_arm
                scored_arm.to_parquet(
                    side_root / f"meta_oos_predictions_{comparison_id}.parquet",
                    index=False,
                )
                if comparison_id in named_scored:
                    named_scored[comparison_id].append(scored_arm)
            scored = comparison_predictions[winner_id]
            final_metrics = comparison_metrics[winner_id]
            models = comparison_models[winner_id]
            all_scored.append(scored)
            pd.DataFrame(
                [
                    {
                        "recipe_id": row["recipe"]["recipe_id"],
                        "objective": row["objective"],
                        **{
                            key: value
                            for key, value in row["recipe"].items()
                            if key != "recipe_id"
                        },
                    }
                    for row in ordered
                ]
            ).to_csv(side_root / "label_hpo_trials.csv", index=False)
            scored.to_parquet(side_root / "meta_oos_predictions.parquet", index=False)
            joblib.dump(winner_calibrator, side_root / "admission_calibrator.joblib")
            joblib.dump(models[0], side_root / "base_ev_map.joblib")
            models[1].booster_.save_model(str(side_root / "residual_model.txt"))
            # Static spread exclusion is a non-PIT diagnostic only.  Report its
            # exact row impact without using it to select or promote the model.
            excluded_mask = spread_exclusion_mask(
                frame["__symbol__"], spread_excluded
            )
            spread_eval = scored.loc[
                ~spread_exclusion_mask(scored["__symbol__"], spread_excluded)
            ].reset_index(drop=True)
            spread_metrics = (
                economic_metrics(spread_eval, spread_eval["residual_score"])
                if len(spread_eval)
                else {}
            )
            summary["sides"][side] = {
                "label_evidence": label_evidence,
                "loader_evidence": loader_evidence,
                "feature_count": len(features),
                "features": features,
                "feature_complete_rows": int(len(frame)),
                "base_train_rows": int(len(train_indices)),
                "base_oos_rows": int(len(oos_indices)),
                "winner_recipe": winner_id,
                "winner_recipe_contract": ordered[0]["recipe"],
                "label_hpo_trials": ordered,
                "final_meta_oos": final_metrics,
                "final_meta_oos_named_ablations": comparison_metrics,
                "spread_exclusion_diagnostic": {
                    "rows_in_full_calendar_matching_current_exclusion": int(
                        excluded_mask.sum()
                    ),
                    "final_oos_rows_before": int(len(scored)),
                    "final_oos_rows_after": int(len(spread_eval)),
                    "metrics_after_static_exclusion": spread_metrics,
                    "promotion_eligible": False,
                    "reason": "future-derived current spread baseline is not PIT",
                },
            }
            _atomic_json(side_root / "summary.json", summary["sides"][side])
            del (
                matrix,
                frame,
                components,
                trial_state,
                models,
                comparison_models,
                comparison_predictions,
            )
            gc.collect()
        combined = pd.concat(all_scored, ignore_index=True)
        combined.to_parquet(stage / "meta_oos_predictions.parquet", index=False)
        summary["combined_final_meta_oos"] = {
            "raw": economic_metrics(combined, combined["residual_score"]),
            "post_21d_admission": economic_metrics(
                combined,
                combined["residual_score"],
                admitted=combined["admitted_after_21d_calibrator"],
            ),
        }
        summary["combined_named_ablations"] = {}
        for arm_id, parts in named_scored.items():
            arm_frame = pd.concat(parts, ignore_index=True)
            summary["combined_named_ablations"][arm_id] = {
                "raw": economic_metrics(arm_frame, arm_frame["residual_score"]),
                "post_21d_admission": economic_metrics(
                    arm_frame,
                    arm_frame["residual_score"],
                    admitted=arm_frame["admitted_after_21d_calibrator"],
                ),
            }
        _atomic_json(stage / "summary.json", summary)
        os.replace(stage, args.output)
        return summary
    except Exception:
        _atomic_json(stage / "failure.json", summary)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--path-labels", type=Path, default=DEFAULT_PATH_LABELS)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--base-contract", type=Path, default=DEFAULT_BASE_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-train-rows", type=int, default=150_000)
    parser.add_argument("--residual-train-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(json.dumps(_safe(summary["combined_final_meta_oos"]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
