#!/usr/bin/env python3
"""Rebuild the side-local residual on the exact cost-aware H12 target.

This is a sealed diagnostic, not a promotion runner.  Feature selection and
HPO use only February--April 2025 exact-H12 rows.  The chosen side-specific
contracts are then frozen for expanding May, June and July 2026 OOF folds.
Every fold fits its own side-local base-score EV map and residual model using
only labels resolved before the fold starts.  Selection metrics are one pooled
global monthly top-k across timestamps and sides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "exact_h12_side_local_residual_oof_v2"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
SIDES = ("long", "short")
TARGET = "execution_net_ev_12h"
PROTECTED_CONTEXT = (
    "base_oof_score",
    "base_rank_pct_timestamp_side",
    "base_score_z_timestamp_side",
)
FOLDS = (
    ("may_2026", "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
    ("june_2026", "2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("july_2026_through_10", "2026-07-01T00:00:00Z", "2026-07-11T00:00:00Z"),
)
ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0, 1.25)
DEFAULT_DATASET = (
    ROOT
    / "data_perp/artifacts/cross_era_tail_payoff_dataset_20260730_v3"
)
DEFAULT_WATERFALL = (
    ROOT
    / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/exact_h12_side_local_residual_oof_20260730_v2"
)


class ExactH12ResidualError(RuntimeError):
    """Raised when an exact-H12 residual contract cannot be proven."""


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def rank_ic(score: Sequence[float], target: Sequence[float]) -> float:
    left = np.asarray(score, dtype=float)
    right = np.asarray(target, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.unique(left[finite]).size < 2:
        return np.nan
    value = spearmanr(left[finite], right[finite]).statistic
    return float(value) if np.isfinite(value) else np.nan


def stable_top(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
    *,
    secondary_column: str | None = None,
) -> pd.DataFrame:
    """Select once across the supplied pooled rows with deterministic ties."""
    take = max(1, int(math.ceil(float(fraction) * len(frame))))
    keys: list[np.ndarray] = [frame["candidate_id"].astype(str).to_numpy()]
    if secondary_column is not None:
        keys.append(
            -pd.to_numeric(frame[secondary_column], errors="raise").to_numpy(float)
        )
    keys.append(-pd.to_numeric(frame[score_column], errors="raise").to_numpy(float))
    order = np.lexsort(tuple(keys))
    return frame.iloc[order[:take]].copy()


def _utc_week_start(timestamp: pd.Series) -> pd.Series:
    day = pd.to_datetime(timestamp, utc=True, errors="raise").dt.floor("D")
    return day - pd.to_timedelta(day.dt.dayofweek, unit="D")


def _weekly_top10_objective(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    secondary: np.ndarray,
) -> dict[str, float]:
    local = frame.loc[:, ["candidate_id", "__ts__", TARGET]].copy()
    local["_score"] = np.asarray(score, dtype=float)
    local["_secondary"] = np.asarray(secondary, dtype=float)
    local["_week"] = _utc_week_start(local["__ts__"])
    weekly: list[float] = []
    for _, week in local.groupby("_week", sort=True):
        selected = stable_top(
            week,
            "_score",
            0.10,
            secondary_column="_secondary",
        )
        weekly.append(float(selected[TARGET].mean() * 1e4))
    values = np.asarray(weekly, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    worst = float(values.min())
    selected = stable_top(
        local,
        "_score",
        0.10,
        secondary_column="_secondary",
    )
    return {
        "objective_bps": mean - 0.5 * std + 0.25 * worst,
        "mean_week_top10_net_bps": mean,
        "std_week_top10_net_bps": std,
        "worst_week_top10_net_bps": worst,
        "month_top10_net_bps": float(selected[TARGET].mean() * 1e4),
        "rank_ic_exact_net": rank_ic(local["_score"], local[TARGET]),
    }


def _normalise_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str)
    result["__symbol__"] = (
        result["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    )
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    return result


def _load_inputs(
    dataset_dir: Path,
    waterfall_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset_manifest_path = dataset_dir / "manifest.json"
    feature_contract_path = dataset_dir / "feature_contract.json"
    dataset_path = dataset_dir / "cross_era_tail_payoff_dataset.parquet"
    waterfall_manifest_path = waterfall_dir / "manifest.json"
    waterfall_path = waterfall_dir / "allscore_waterfall.parquet"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    waterfall_manifest = json.loads(
        waterfall_manifest_path.read_text(encoding="utf-8")
    )
    if dataset_manifest.get("schema") != "cross_era_tail_payoff_dataset_v3":
        raise ExactH12ResidualError("unexpected cross-era dataset schema")
    if (
        dataset_manifest.get("outputs", {}).get("dataset", {}).get("sha256")
        != sha256(dataset_path)
    ):
        raise ExactH12ResidualError("cross-era dataset hash changed")
    if (
        waterfall_manifest.get("schema")
        != "mayjul2026_exact_allscore_ic_ev_waterfall_v1"
        or waterfall_manifest.get("outputs", {})
        .get("allscore_waterfall", {})
        .get("sha256")
        != sha256(waterfall_path)
    ):
        raise ExactH12ResidualError("exact-H12 waterfall binding changed")
    feature_contract = json.loads(feature_contract_path.read_text(encoding="utf-8"))
    raw_features = list(map(str, feature_contract["feature_columns"]))
    context = list(map(str, feature_contract["candidate_context_columns"]))
    if len(raw_features) != 256 or len(set(raw_features)) != 256:
        raise ExactH12ResidualError("expected frozen 256-feature contract")
    features = list(dict.fromkeys([*raw_features, *context]))
    forbidden = {
        TARGET,
        "execution_gross_ev_12h",
        "execution_cost_return",
        "positive_net",
        "negative_net",
        "event_class",
        "clean_first",
        "adverse_first",
        "timeout_event",
        "soft_upper_hit",
        "soft_lower_hit",
        "meaningful_mfe_reached",
        "peak_mfe_atr",
        "time_to_meaningful_mfe_hours",
        "mae_before_meaningful_mfe_atr",
        "future_slope_atr_per_hour",
        "label_resolution_utc",
    }
    overlap = sorted(set(features).intersection(forbidden))
    if overlap:
        raise ExactH12ResidualError(f"outcome fields entered feature contract: {overlap}")
    history_columns = [
        *IDENTITY,
        "era",
        "label_resolution_utc",
        *features,
        "execution_gross_ev_12h",
        "execution_cost_return",
        TARGET,
    ]
    history = _normalise_identity(
        pd.read_parquet(dataset_path, columns=history_columns)
    )
    history["label_resolution_utc"] = pd.to_datetime(
        history["label_resolution_utc"], utc=True, errors="raise"
    )
    waterfall = _normalise_identity(pd.read_parquet(waterfall_path))
    for frame, name in ((history, "history"), (waterfall, "waterfall")):
        if frame.duplicated(list(IDENTITY)).any():
            raise ExactH12ResidualError(f"{name} has duplicate candidate identities")
        if not set(frame["side_name"].unique()) <= set(SIDES):
            raise ExactH12ResidualError(f"{name} has invalid side")
    if not (
        history["label_resolution_utc"]
        .eq(history["__ts__"] + pd.Timedelta(hours=13))
        .all()
    ):
        raise ExactH12ResidualError("cross-era labels are not signal+13h")
    gross = pd.to_numeric(history["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(history["execution_cost_return"], errors="raise")
    net = pd.to_numeric(history[TARGET], errors="raise")
    if not np.allclose(gross - cost, net, rtol=0.0, atol=1e-10):
        raise ExactH12ResidualError("cross-era gross-cost-net reconciliation failed")
    expected_decision = waterfall["__ts__"] + pd.Timedelta(hours=1)
    expected_end = expected_decision + pd.Timedelta(hours=12)
    if not pd.to_datetime(
        waterfall["execution_decision_utc"], utc=True, errors="raise"
    ).eq(expected_decision).all():
        raise ExactH12ResidualError("waterfall decision timestamp mismatch")
    if not pd.to_datetime(
        waterfall["execution_label_end_utc"], utc=True, errors="raise"
    ).eq(expected_end).all():
        raise ExactH12ResidualError("waterfall label endpoint mismatch")
    evaluation = history.merge(
        waterfall,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
        suffixes=("_dataset", ""),
    )
    if len(evaluation) != len(waterfall):
        raise ExactH12ResidualError(
            f"waterfall identity coverage incomplete: {len(evaluation)} != {len(waterfall)}"
        )
    base_left_f64 = pd.to_numeric(
        evaluation["base_oof_score"], errors="raise"
    ).to_numpy(np.float64)
    base_right_f64 = pd.to_numeric(
        evaluation["score_base_alpha"], errors="raise"
    ).to_numpy(np.float64)
    # The cross-era feature surface intentionally persists model features as
    # float32 while the score waterfall retains the upstream float64 values.
    # Demand bit identity at the declared feature-storage precision; do not use
    # an unconstrained tolerance that could conceal a lineage mismatch.
    base_left = base_left_f64.astype(np.float32)
    base_right = base_right_f64.astype(np.float32)
    if not np.array_equal(base_left, base_right):
        raise ExactH12ResidualError(
            "base OOF score differs at declared float32 feature precision"
        )
    base_score_max_storage_delta = float(
        np.max(np.abs(base_left_f64 - base_right_f64))
    )
    for column in (
        "execution_gross_ev_12h",
        "execution_cost_return",
        TARGET,
    ):
        left = pd.to_numeric(
            evaluation[f"{column}_dataset"], errors="raise"
        ).to_numpy(np.float64)
        right = pd.to_numeric(evaluation[column], errors="raise").to_numpy(np.float64)
        if not np.array_equal(left, right):
            raise ExactH12ResidualError(f"{column} differs across exact inputs")
    start, end = pd.Timestamp(FOLDS[0][1]), pd.Timestamp(FOLDS[-1][2])
    if not (
        evaluation["__ts__"].ge(start) & evaluation["__ts__"].lt(end)
    ).all():
        raise ExactH12ResidualError("waterfall evaluation calendar changed")
    expected_counts = {
        ("2026-05", "long"): 29_848,
        ("2026-05", "short"): 33_503,
        ("2026-06", "long"): 24_357,
        ("2026-06", "short"): 24_902,
        ("2026-07", "long"): 7_413,
        ("2026-07", "short"): 7_754,
    }
    observed = (
        evaluation.assign(month=evaluation["__ts__"].dt.strftime("%Y-%m"))
        .groupby(["month", "side_name"], observed=True)
        .size()
        .to_dict()
    )
    if observed != expected_counts:
        raise ExactH12ResidualError(f"evaluation support changed: {observed}")
    evidence = {
        "dataset": binding(dataset_path),
        "dataset_manifest": binding(dataset_manifest_path),
        "feature_contract": binding(feature_contract_path),
        "waterfall": binding(waterfall_path),
        "waterfall_manifest": binding(waterfall_manifest_path),
        "feature_columns": features,
        "raw_feature_count": len(raw_features),
        "context_feature_count": len(context),
        "evaluation_rows": len(evaluation),
        "base_score_identity": {
            "status": "bit_identical_after_declared_float32_feature_storage",
            "max_precast_abs_delta": base_score_max_storage_delta,
        },
        "evaluation_support": {
            f"{month}:{side}": rows
            for (month, side), rows in expected_counts.items()
        },
    }
    return history, evaluation, evidence


def _feature_coverage(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    period: str,
    side: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        rows.append(
            {
                "period": period,
                "side_name": side,
                "feature": feature,
                "rows": len(frame),
                "finite_rows": int(finite.sum()),
                "finite_fraction": float(finite.mean()),
                "unique_finite": int(np.unique(values[finite]).size),
            }
        )
    return pd.DataFrame(rows)


def _feature_screen(
    frame: pd.DataFrame,
    residual_bps: np.ndarray,
    features: Sequence[str],
    *,
    max_count: int,
) -> tuple[list[str], pd.DataFrame]:
    matrix = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    coverage = matrix.notna().mean()
    variance = matrix.var()
    candidates = list(
        coverage.index[(coverage >= 0.95) & (variance > 1e-12)]
    )
    if not candidates:
        raise ExactH12ResidualError("no admissible residual features")
    values = matrix[candidates]
    midpoint = max(1, len(values) // 2)
    y = pd.Series(np.asarray(residual_bps, dtype=float), index=values.index)

    def correlations(local: pd.DataFrame, target: pd.Series) -> pd.Series:
        return local.corrwith(target).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    full = correlations(values, y)
    early = correlations(values.iloc[:midpoint], y.iloc[:midpoint])
    late = correlations(values.iloc[midpoint:], y.iloc[midpoint:])
    stable = pd.Series(
        np.where(
            early.to_numpy() * late.to_numpy() > 0.0,
            np.minimum(np.abs(early.to_numpy()), np.abs(late.to_numpy())),
            0.10 * np.abs(full.to_numpy()),
        ),
        index=values.columns,
    )
    ordered = sorted(
        candidates,
        key=lambda column: (-float(stable[column]), str(column)),
    )
    selected: list[str] = [
        feature for feature in PROTECTED_CONTEXT if feature in candidates
    ]
    for feature in ordered:
        if feature in selected:
            continue
        if len(selected) >= max_count:
            break
        candidate = values[feature]
        correlated = False
        for previous in selected:
            correlation = candidate.corr(values[previous])
            if np.isfinite(correlation) and abs(correlation) >= 0.95:
                correlated = True
                break
        if not correlated:
            selected.append(feature)
    if len(selected) < min(16, max_count):
        raise ExactH12ResidualError("residual feature screen collapsed")
    report = pd.DataFrame(
        {
            "feature": candidates,
            "coverage": [float(coverage[name]) for name in candidates],
            "variance": [float(variance[name]) for name in candidates],
            "full_corr": [float(full[name]) for name in candidates],
            "early_corr": [float(early[name]) for name in candidates],
            "late_corr": [float(late[name]) for name in candidates],
            "stable_score": [float(stable[name]) for name in candidates],
            "selected_order": [
                selected.index(name) if name in selected else -1 for name in candidates
            ],
        }
    ).sort_values(
        ["selected_order", "stable_score"],
        ascending=[True, False],
        kind="stable",
    )
    return selected, report.reset_index(drop=True)


def _configs_for_side(side: str) -> tuple[dict[str, Any], ...]:
    common = {
        "objective": "regression_l2",
        "verbosity": -1,
        "num_threads": max(1, min(6, int(os.cpu_count() or 1))),
        "bagging_freq": 1,
    }
    legacy = (
        {
            "feature_count": 64,
            "rounds": 524,
            "learning_rate": 0.008401344966231013,
            "num_leaves": 16,
            "max_depth": 6,
            "min_data_in_leaf": 628,
            "feature_fraction": 0.5526098190398928,
            "bagging_fraction": 0.9414633662175064,
            "lambda_l1": 0.06669625805934529,
            "lambda_l2": 0.8851989155340616,
        }
        if side == "long"
        else {
            "feature_count": 64,
            "rounds": 642,
            "learning_rate": 0.06169209574497716,
            "num_leaves": 8,
            "max_depth": 3,
            "min_data_in_leaf": 485,
            "feature_fraction": 0.9851166482864842,
            "bagging_fraction": 0.7394121354773147,
            "lambda_l1": 1.796227973809244,
            "lambda_l2": 6.426041288770094,
        }
    )
    return (
        {
            "name": "shallow_24",
            "feature_count": 24,
            "rounds": 120,
            **common,
            "learning_rate": 0.045,
            "num_leaves": 15,
            "max_depth": 5,
            "min_data_in_leaf": 300,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.85,
            "lambda_l1": 0.0,
            "lambda_l2": 20.0,
        },
        {
            "name": "regularised_40",
            "feature_count": 40,
            "rounds": 150,
            **common,
            "learning_rate": 0.035,
            "num_leaves": 23,
            "max_depth": 6,
            "min_data_in_leaf": 500,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.85,
            "lambda_l1": 0.0,
            "lambda_l2": 35.0,
        },
        {"name": "legacy_capacity_64", **common, **legacy},
    )


def _fit_ev_map(score: np.ndarray, target_bps: np.ndarray) -> IsotonicRegression:
    finite = np.isfinite(score) & np.isfinite(target_bps)
    if finite.sum() < 5_000:
        raise ExactH12ResidualError("side-local EV map has insufficient support")
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(score[finite], target_bps[finite])
    return model


def _fit_model(
    matrix: pd.DataFrame,
    target_bps: np.ndarray,
    features: Sequence[str],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> lgb.Booster:
    params = {
        key: value
        for key, value in config.items()
        if key not in {"name", "feature_count", "rounds"}
    }
    params["seed"] = int(seed)
    dataset = lgb.Dataset(
        matrix.loc[:, list(features)].to_numpy(np.float32, copy=False),
        label=np.asarray(target_bps, dtype=np.float32),
        feature_name=list(features),
        free_raw_data=True,
    )
    return lgb.train(params, dataset, num_boost_round=int(config["rounds"]))


def _development_contract(
    history: pd.DataFrame,
    features: Sequence[str],
    *,
    side: str,
    seed: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, Any]],
    pd.DataFrame,
]:
    local = history.loc[history["side_name"].eq(side)].sort_values(
        ["__ts__", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    validation_start = pd.Timestamp("2025-04-01T00:00:00Z")
    validation_end = pd.Timestamp("2025-05-01T00:00:00Z")
    train_mask = (
        local["__ts__"].lt(validation_start)
        & local["label_resolution_utc"].lt(validation_start)
    )
    validation_mask = local["__ts__"].ge(validation_start) & local["__ts__"].lt(
        validation_end
    )
    train = local.loc[train_mask].copy()
    validation = local.loc[validation_mask].copy()
    if len(train) < 20_000 or len(validation) < 20_000:
        raise ExactH12ResidualError(f"{side} development support is insufficient")
    score_train = pd.to_numeric(train["base_oof_score"], errors="raise").to_numpy()
    target_train = pd.to_numeric(train[TARGET], errors="raise").to_numpy() * 1e4
    ev_map = _fit_ev_map(score_train, target_train)
    baseline_train = np.asarray(ev_map.predict(score_train), dtype=float)
    residual_train = target_train - baseline_train
    ordered_features, feature_report = _feature_screen(
        train,
        residual_train,
        features,
        max_count=64,
    )
    full_matrix = local.loc[:, list(features)].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    score_validation = pd.to_numeric(
        validation["base_oof_score"], errors="raise"
    ).to_numpy()
    baseline_validation = np.asarray(ev_map.predict(score_validation), dtype=float)
    train_positions = np.flatnonzero(train_mask.to_numpy())
    validation_positions = np.flatnonzero(validation_mask.to_numpy())
    trials: list[dict[str, Any]] = []
    candidate_contracts: dict[str, dict[str, Any]] = {}
    candidate_predictions: list[pd.DataFrame] = []
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    for config_index, config in enumerate(_configs_for_side(side)):
        count = min(int(config["feature_count"]), len(ordered_features))
        selected = ordered_features[:count]
        model = _fit_model(
            full_matrix.iloc[train_positions],
            residual_train,
            selected,
            config,
            seed=seed + config_index,
        )
        delta = np.asarray(
            model.predict(full_matrix.iloc[validation_positions][selected]),
            dtype=float,
        )
        for alpha in ALPHA_GRID:
            trial_id = f"{config['name']}__alpha_{float(alpha):.2f}"
            corrected = baseline_validation + float(alpha) * delta
            metrics = _weekly_top10_objective(
                validation,
                corrected,
                secondary=score_validation,
            )
            record = {
                "side_name": side,
                "trial_id": trial_id,
                "config_name": config["name"],
                "feature_count": len(selected),
                "alpha": float(alpha),
                **metrics,
            }
            trials.append(record)
            key = (
                float(metrics["objective_bps"]),
                float(metrics["month_top10_net_bps"]),
                float(metrics["rank_ic_exact_net"]),
            )
            candidate = {
                "trial_id": trial_id,
                "side_name": side,
                "config": dict(config),
                "features": selected,
                "alpha": float(alpha),
                "development_train_rows": len(train),
                "development_validation_rows": len(validation),
                "development_train_label_resolution_max": train[
                    "label_resolution_utc"
                ].max(),
                "development_validation_start": validation_start,
                "development_validation_end": validation_end,
                "metrics": metrics,
            }
            candidate["contract_sha256"] = stable_hash(
                {
                    "side_name": side,
                    "config": candidate["config"],
                    "features": candidate["features"],
                    "alpha": candidate["alpha"],
                }
            )
            candidate_contracts[trial_id] = candidate
            trial_prediction = validation.loc[
                :, ["candidate_id", "side_name", "__ts__", TARGET, "base_oof_score"]
            ].copy()
            trial_prediction["trial_id"] = trial_id
            trial_prediction["score_bps"] = corrected
            candidate_predictions.append(trial_prediction)
            if best is None or key > best[0]:
                best = (key, candidate)
        del model
    if best is None:
        raise ExactH12ResidualError("HPO produced no valid contract")
    winner = best[1]
    return (
        winner,
        feature_report,
        pd.DataFrame(trials),
        candidate_contracts,
        pd.concat(candidate_predictions, ignore_index=True),
    )


def _joint_development_contracts(
    candidate_contracts: Mapping[str, Mapping[str, Mapping[str, Any]]],
    candidate_predictions: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Choose side-local contracts on the actual pooled-global objective."""
    records: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float], tuple[str, str]] | None = None
    for long_trial in sorted(candidate_contracts["long"]):
        long_rows = candidate_predictions["long"].loc[
            candidate_predictions["long"]["trial_id"].eq(long_trial)
        ]
        for short_trial in sorted(candidate_contracts["short"]):
            short_rows = candidate_predictions["short"].loc[
                candidate_predictions["short"]["trial_id"].eq(short_trial)
            ]
            pooled = pd.concat([long_rows, short_rows], ignore_index=True)
            metrics = _weekly_top10_objective(
                pooled,
                pooled["score_bps"].to_numpy(float),
                secondary=pooled["base_oof_score"].to_numpy(float),
            )
            record = {
                "long_trial_id": long_trial,
                "short_trial_id": short_trial,
                **metrics,
            }
            records.append(record)
            key = (
                float(metrics["objective_bps"]),
                float(metrics["month_top10_net_bps"]),
                float(metrics["rank_ic_exact_net"]),
            )
            if best is None or key > best[0]:
                best = (key, (long_trial, short_trial))
    if best is None:
        raise ExactH12ResidualError("joint pooled-global HPO produced no winner")
    long_trial, short_trial = best[1]
    selected = {
        "long": dict(candidate_contracts["long"][long_trial]),
        "short": dict(candidate_contracts["short"][short_trial]),
    }
    for side in SIDES:
        selected[side]["selection_basis"] = (
            "joint_pooled_global_april_weekly_top10_objective"
        )
        selected[side]["joint_pair"] = {
            "long_trial_id": long_trial,
            "short_trial_id": short_trial,
        }
    return selected, pd.DataFrame(records).sort_values(
        ["objective_bps", "month_top10_net_bps", "rank_ic_exact_net"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def _folds() -> tuple[Fold, ...]:
    return tuple(
        Fold(name, pd.Timestamp(start), pd.Timestamp(end))
        for name, start, end in FOLDS
    )


def _save_fold(
    root: Path,
    *,
    model: lgb.Booster,
    ev_map: IsotonicRegression,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    model_path = root / "residual_model.txt"
    map_path = root / "base_ev_map.joblib"
    contract_path = root / "contract.json"
    model.save_model(str(model_path))
    joblib.dump(ev_map, map_path, compress=3)
    write_json(contract_path, contract)
    return {
        "model": binding(model_path),
        "base_ev_map": binding(map_path),
        "contract": binding(contract_path),
    }


def _score_oof(
    history: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: Sequence[str],
    contracts: Mapping[str, Mapping[str, Any]],
    stage: Path,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    coverage: list[pd.DataFrame] = []
    for side_index, side in enumerate(SIDES):
        contract = contracts[side]
        selected = list(map(str, contract["features"]))
        side_history = history.loc[history["side_name"].eq(side)].sort_values(
            ["__ts__", "candidate_id"], kind="stable"
        ).reset_index(drop=True)
        matrix = side_history.loc[:, selected].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        for fold_index, fold in enumerate(_folds()):
            train_mask = (
                side_history["__ts__"].lt(fold.start)
                & side_history["label_resolution_utc"].lt(fold.start)
            )
            valid_mask = side_history["__ts__"].ge(
                fold.start
            ) & side_history["__ts__"].lt(fold.end)
            train_positions = np.flatnonzero(train_mask.to_numpy())
            valid_positions = np.flatnonzero(valid_mask.to_numpy())
            train = side_history.iloc[train_positions]
            valid = side_history.iloc[valid_positions]
            expected = evaluation.loc[
                evaluation["side_name"].eq(side)
                & evaluation["__ts__"].ge(fold.start)
                & evaluation["__ts__"].lt(fold.end)
            ]
            if len(valid) != len(expected) or set(valid["candidate_id"]) != set(
                expected["candidate_id"]
            ):
                raise ExactH12ResidualError(
                    f"{side} {fold.name} validation identities changed"
                )
            if len(train) < 30_000 or len(valid) < 1_000:
                raise ExactH12ResidualError(
                    f"{side} {fold.name} support insufficient"
                )
            if not (
                train["__ts__"].lt(fold.start).all()
                and train["label_resolution_utc"].lt(fold.start).all()
            ):
                raise ExactH12ResidualError(f"{side} {fold.name} leaked")
            score_train = pd.to_numeric(
                train["base_oof_score"], errors="raise"
            ).to_numpy(float)
            target_train = (
                pd.to_numeric(train[TARGET], errors="raise").to_numpy(float) * 1e4
            )
            ev_map = _fit_ev_map(score_train, target_train)
            baseline_train = np.asarray(ev_map.predict(score_train), dtype=float)
            model = _fit_model(
                matrix.iloc[train_positions],
                target_train - baseline_train,
                selected,
                contract["config"],
                seed=seed + side_index * 1000 + fold_index,
            )
            score_valid = pd.to_numeric(
                valid["base_oof_score"], errors="raise"
            ).to_numpy(float)
            baseline_valid = np.asarray(ev_map.predict(score_valid), dtype=float)
            delta = np.asarray(
                model.predict(matrix.iloc[valid_positions][selected]), dtype=float
            )
            corrected = baseline_valid + float(contract["alpha"]) * delta
            fold_contract = {
                "schema": SCHEMA,
                "side_name": side,
                "fold": fold.name,
                "validation_start": fold.start,
                "validation_end": fold.end,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "train_signal_max": train["__ts__"].max(),
                "train_label_resolution_max": train["label_resolution_utc"].max(),
                "all_train_labels_resolved_before_fold": True,
                "target": "execution_net_ev_12h * 10000 bps",
                "base_score": "base_oof_score bit-identical to score_base_alpha",
                "features": selected,
                "feature_count": len(selected),
                "native_missing_value_routing": True,
                "imputation": "none",
                "config": contract["config"],
                "alpha": contract["alpha"],
                "development_contract_sha256": contract["contract_sha256"],
                "final_refit_used": False,
            }
            hashes = _save_fold(
                stage / side / "folds" / fold.name,
                model=model,
                ev_map=ev_map,
                contract=fold_contract,
            )
            scored = valid.loc[
                :, [*IDENTITY, "era", "label_resolution_utc", TARGET]
            ].copy()
            scored["residual_oof_fold"] = fold.name
            scored["residual_train_cutoff_utc"] = fold.start
            scored["residual_train_label_resolution_max"] = train[
                "label_resolution_utc"
            ].max()
            scored["score_exact_h12_base_ev_bps"] = baseline_valid
            scored["score_exact_h12_residual_delta_bps"] = delta
            scored["score_exact_h12_residual_bps"] = corrected
            scored["score_available_at_utc"] = scored["__ts__"] + pd.Timedelta(
                hours=1
            )
            scored["is_strict_oof"] = True
            predictions.append(scored)
            audits.append(
                {
                    **fold_contract,
                    "model_sha256": hashes["model"]["sha256"],
                    "base_ev_map_sha256": hashes["base_ev_map"]["sha256"],
                    "contract_file_sha256": hashes["contract"]["sha256"],
                    "train_target_mean_bps": float(target_train.mean()),
                    "validation_target_mean_bps": float(
                        valid[TARGET].mean() * 1e4
                    ),
                    "validation_base_ev_rank_ic": rank_ic(
                        baseline_valid, valid[TARGET]
                    ),
                    "validation_residual_rank_ic": rank_ic(
                        corrected, valid[TARGET]
                    ),
                }
            )
            coverage.append(
                _feature_coverage(
                    train,
                    selected,
                    period=f"{fold.name}_train",
                    side=side,
                )
            )
            coverage.append(
                _feature_coverage(
                    valid,
                    selected,
                    period=f"{fold.name}_validation",
                    side=side,
                )
            )
            del model, ev_map
    result = pd.concat(predictions, ignore_index=True).sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="stable"
    ).reset_index(drop=True)
    if (
        len(result) != len(evaluation)
        or result["candidate_id"].duplicated().any()
        or set(result["candidate_id"]) != set(evaluation["candidate_id"])
    ):
        raise ExactH12ResidualError("OOF prediction identity coverage is incomplete")
    if not (
        result["score_available_at_utc"]
        .le(result["__ts__"] + pd.Timedelta(hours=1))
        .all()
    ):
        raise ExactH12ResidualError("a residual score is late")
    return result, pd.DataFrame(audits), pd.concat(coverage, ignore_index=True)


def _score_registry() -> dict[str, dict[str, str | None]]:
    return {
        "base_alpha_raw": {
            "column": "score_base_alpha",
            "secondary": None,
            "target": "legacy native-24h alpha target",
        },
        "base_ev_exact_h12": {
            "column": "score_exact_h12_base_ev_bps",
            "secondary": "score_base_alpha",
            "target": "side-local exact-H12 net EV map",
        },
        "residual_exact_h12": {
            "column": "score_exact_h12_residual_bps",
            "secondary": "score_base_alpha",
            "target": "side-local exact-H12 net EV residual",
        },
        "direct_q25_exact_h12": {
            "column": "score_direct_q25_challenger_bps",
            "secondary": None,
            "target": "direct exact-H12 q25 challenger",
        },
        "residual_legacy_24h": {
            "column": "score_residual_expected_ev",
            "secondary": None,
            "target": "legacy fixed-cost 24h residual",
        },
    }


def _period_metrics(
    frame: pd.DataFrame,
    registry: Mapping[str, Mapping[str, str | None]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    books: list[pd.DataFrame] = []
    side_rows: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("candidate_month", sort=True):
        for score_name, record in registry.items():
            column = str(record["column"])
            secondary = record["secondary"]
            full = {
                "month": str(month),
                "score_name": score_name,
                "score_column": column,
                "eligible_rows": len(month_rows),
                "rank_ic_native_24h_alpha": rank_ic(
                    month_rows[column], month_rows["__first_touch_target_soft__"]
                ),
                "rank_ic_exact_h12_mfe": rank_ic(
                    month_rows[column], month_rows["execution_mfe_return_12h"]
                ),
                "rank_ic_exact_h12_gross": rank_ic(
                    month_rows[column], month_rows["execution_gross_ev_12h"]
                ),
                "rank_ic_exact_h12_net": rank_ic(
                    month_rows[column], month_rows[TARGET]
                ),
            }
            for fraction in (0.01, 0.05, 0.10, 0.20):
                selected = stable_top(
                    month_rows,
                    column,
                    fraction,
                    secondary_column=str(secondary) if secondary else None,
                )
                net_bps = selected[TARGET].to_numpy(float) * 1e4
                gross_bps = (
                    selected["execution_gross_ev_12h"].to_numpy(float) * 1e4
                )
                cost_bps = (
                    selected["execution_cost_return"].to_numpy(float) * 1e4
                )
                label = f"top{int(fraction * 100):02d}"
                full.update(
                    {
                        f"{label}_rows": len(selected),
                        f"{label}_gross_bps": float(gross_bps.mean()),
                        f"{label}_cost_bps": float(cost_bps.mean()),
                        f"{label}_net_bps": float(net_bps.mean()),
                        f"{label}_sum_net": float(selected[TARGET].sum()),
                        f"{label}_positive_net_rate": float((net_bps > 0).mean()),
                        f"{label}_cvar05_net_bps": float(
                            np.sort(net_bps)[
                                : max(1, int(math.ceil(0.05 * len(net_bps))))
                            ].mean()
                        ),
                    }
                )
                if fraction == 0.10:
                    book = selected.loc[
                        :,
                        [
                            *IDENTITY,
                            "candidate_month",
                            "execution_exit_reason",
                            "execution_gross_ev_12h",
                            "execution_cost_return",
                            TARGET,
                        ],
                    ].copy()
                    book["score_name"] = score_name
                    book["score_value"] = selected[column].to_numpy(float)
                    book["selected_rank"] = np.arange(1, len(book) + 1)
                    books.append(book)
                    for side in SIDES:
                        cohort = selected.loc[selected["side_name"].eq(side)]
                        side_rows.append(
                            {
                                "month": str(month),
                                "score_name": score_name,
                                "side_name": side,
                                "global_book_rows": len(cohort),
                                "global_book_share": float(
                                    len(cohort) / len(selected)
                                ),
                                "conditional_net_bps": (
                                    float(cohort[TARGET].mean() * 1e4)
                                    if len(cohort)
                                    else np.nan
                                ),
                                "contribution_net_bps": (
                                    float(cohort[TARGET].sum() * 1e4 / len(selected))
                                    if len(cohort)
                                    else 0.0
                                ),
                            }
                        )
            metrics.append(full)
    return (
        pd.DataFrame(metrics),
        pd.concat(books, ignore_index=True),
        pd.DataFrame(side_rows),
    )


def _replacement_attribution(books: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("base_alpha_raw", "base_ev_exact_h12"),
        ("base_alpha_raw", "residual_exact_h12"),
        ("base_alpha_raw", "direct_q25_exact_h12"),
        ("direct_q25_exact_h12", "residual_exact_h12"),
        ("residual_legacy_24h", "residual_exact_h12"),
    )
    rows: list[dict[str, Any]] = []
    for month, month_rows in books.groupby("candidate_month", sort=True):
        by_score = {
            score: local.set_index("candidate_id", drop=False)
            for score, local in month_rows.groupby("score_name", sort=True)
        }
        for baseline, challenger in pairs:
            left, right = by_score[baseline], by_score[challenger]
            shared = left.index.intersection(right.index)
            removed = left.loc[left.index.difference(right.index)]
            added = right.loc[right.index.difference(left.index)]
            if len(left) != len(right) or len(removed) != len(added):
                raise ExactH12ResidualError("replacement books do not have fixed size")
            delta = (
                added[TARGET].sum() - removed[TARGET].sum()
            ) * 1e4 / len(left)
            rows.append(
                {
                    "month": str(month),
                    "baseline": baseline,
                    "challenger": challenger,
                    "book_rows": len(left),
                    "shared_rows": len(shared),
                    "jaccard": float(len(shared) / len(left.index.union(right.index))),
                    "removed_rows": len(removed),
                    "removed_mean_net_bps": (
                        float(removed[TARGET].mean() * 1e4)
                        if len(removed)
                        else np.nan
                    ),
                    "added_rows": len(added),
                    "added_mean_net_bps": (
                        float(added[TARGET].mean() * 1e4)
                        if len(added)
                        else np.nan
                    ),
                    "exact_replacement_delta_net_bps": float(delta),
                    "reconciles": True,
                }
            )
            for cohort_name, cohort in (("removed", removed), ("added", added)):
                for side in SIDES:
                    local = cohort.loc[cohort["side_name"].eq(side)]
                    rows.append(
                        {
                            "month": str(month),
                            "baseline": baseline,
                            "challenger": challenger,
                            "book_rows": len(left),
                            "shared_rows": len(shared),
                            "jaccard": float(
                                len(shared) / len(left.index.union(right.index))
                            ),
                            "cohort": cohort_name,
                            "side_name": side,
                            "cohort_rows": len(local),
                            "cohort_mean_net_bps": (
                                float(local[TARGET].mean() * 1e4)
                                if len(local)
                                else np.nan
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def run(
    *,
    dataset_dir: Path,
    waterfall_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    history, evaluation, evidence = _load_inputs(dataset_dir, waterfall_dir)
    stage = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    stage.mkdir(parents=True)
    try:
        local_winners: dict[str, dict[str, Any]] = {}
        candidate_contracts: dict[str, dict[str, dict[str, Any]]] = {}
        candidate_predictions: dict[str, pd.DataFrame] = {}
        feature_reports: list[pd.DataFrame] = []
        hpo_trials: list[pd.DataFrame] = []
        for side_index, side in enumerate(SIDES):
            (
                local_contract,
                feature_report,
                trials,
                side_candidates,
                side_predictions,
            ) = _development_contract(
                history,
                evidence["feature_columns"],
                side=side,
                seed=seed + side_index * 10_000,
            )
            local_winners[side] = local_contract
            candidate_contracts[side] = side_candidates
            candidate_predictions[side] = side_predictions
            feature_report.insert(0, "side_name", side)
            feature_reports.append(feature_report)
            hpo_trials.append(trials)
        contracts, joint_hpo_trials = _joint_development_contracts(
            candidate_contracts, candidate_predictions
        )
        for side in SIDES:
            side_root = stage / side
            side_root.mkdir(parents=True)
            write_json(side_root / "development_contract.json", contracts[side])
        oof, fold_audit, coverage = _score_oof(
            history,
            evaluation,
            evidence["feature_columns"],
            contracts,
            stage,
            seed=seed + 100_000,
        )
        joined = evaluation.merge(
            oof.drop(columns=[TARGET, "era", "label_resolution_utc"]),
            on=list(IDENTITY),
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != len(evaluation):
            raise ExactH12ResidualError("scored evaluation join lost identities")
        registry = _score_registry()
        missing_scores = sorted(
            str(record["column"])
            for record in registry.values()
            if str(record["column"]) not in joined
        )
        if missing_scores:
            raise ExactH12ResidualError(f"comparison scores missing: {missing_scores}")
        period_metrics, books, side_attribution = _period_metrics(joined, registry)
        replacements = _replacement_attribution(books)
        outputs: dict[str, Any] = {}
        tables = (
            ("oof_predictions", oof, ".parquet"),
            ("period_score_metrics", period_metrics, ".csv"),
            ("selection_books", books, ".parquet"),
            ("global_book_side_attribution", side_attribution, ".csv"),
            ("replacement_attribution", replacements, ".csv"),
            (
                "feature_selection",
                pd.concat(feature_reports, ignore_index=True),
                ".csv",
            ),
            ("hpo_trials", pd.concat(hpo_trials, ignore_index=True), ".csv"),
            ("joint_hpo_trials", joint_hpo_trials, ".csv"),
            ("fold_audit", fold_audit, ".csv"),
            ("feature_coverage", coverage, ".csv"),
        )
        for name, table, suffix in tables:
            path = stage / f"{name}{suffix}"
            if suffix == ".parquet":
                table.to_parquet(path, index=False)
            else:
                table.to_csv(path, index=False)
            outputs[name] = {
                "path": str((output_dir / path.name).resolve()),
                "rows": len(table),
                "sha256": sha256(path),
            }
        winners = {
            side: {
                "config_name": contracts[side]["config"]["name"],
                "feature_count": len(contracts[side]["features"]),
                "alpha": contracts[side]["alpha"],
                "development_metrics": contracts[side]["metrics"],
                "contract_sha256": contracts[side]["contract_sha256"],
            }
            for side in SIDES
        }
        top10 = period_metrics.loc[
            :,
            [
                "month",
                "score_name",
                "top10_rows",
                "top10_gross_bps",
                "top10_cost_bps",
                "top10_net_bps",
                "top10_positive_net_rate",
                "rank_ic_native_24h_alpha",
                "rank_ic_exact_h12_net",
            ],
        ].to_dict(orient="records")
        manifest = {
            "schema": SCHEMA,
            "status": "SEALED_DIAGNOSTIC_ONLY_NO_PROMOTION_NO_PORTFOLIO_REPLAY",
            "promotion_eligible": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                key: value
                for key, value in evidence.items()
                if key != "feature_columns"
            },
            "contracts": {
                "architecture": (
                    "frozen base OOF -> side-local exact-H12 EV map + "
                    "side-local exact-H12 residual"
                ),
                "development": (
                    "feature selection and HPO train on resolved February-March "
                    "2025 and validate on April 2025; side-local candidates are "
                    "paired and selected on the pooled-global weekly top10 "
                    "objective, then frozen before May 2026"
                ),
                "oof": (
                    "May, June and July expanding folds; signal and exact-H12 "
                    "label resolution strictly before each fold start"
                ),
                "target": (
                    "execution_net_ev_12h = spread-inclusive gross minus one "
                    "explicit round-trip cost; modeled in bps"
                ),
                "feature_missingness": (
                    "LightGBM native missing-value routing; no imputation; "
                    "coverage persisted by side/fold/feature"
                ),
                "selection": (
                    "one pooled global top-k per calendar month across all "
                    "timestamps and sides; deterministic score/raw-base/candidate "
                    "tie order where declared; no side/timestamp/asset quotas"
                ),
                "actions": (
                    "timing, MAE, wait and target-price layers excluded; frozen "
                    "deployed exit labels only"
                ),
                "mapping": (
                    "only the residual architecture's side-local base-score EV "
                    "map; no recent 21-day admission map in this stage"
                ),
                "final_refit": "none",
            },
            "folds": [
                {"name": fold.name, "start": fold.start, "end": fold.end}
                for fold in _folds()
            ],
            "score_registry": registry,
            "side_winners": winners,
            "side_local_diagnostic_winners_not_used": {
                side: {
                    "trial_id": local_winners[side]["trial_id"],
                    "config_name": local_winners[side]["config"]["name"],
                    "alpha": local_winners[side]["alpha"],
                    "metrics": local_winners[side]["metrics"],
                }
                for side in SIDES
            },
            "top10_summary": top10,
            "outputs": outputs,
            "runner": binding(Path(__file__)),
        }
        write_json(stage / "manifest.json", manifest)
        manifest_sha = sha256(stage / "manifest.json")
        (stage / "manifest.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="utf-8"
        )
        write_json(
            stage / "seal.json",
            {
                "schema": SCHEMA,
                "manifest_sha256": manifest_sha,
                "files_sha256": {
                    path.relative_to(stage).as_posix(): sha256(path)
                    for path in sorted(stage.rglob("*"))
                    if path.is_file() and path.name != "seal.json"
                },
            },
        )
        os.replace(stage, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    value.add_argument("--waterfall-dir", type=Path, default=DEFAULT_WATERFALL)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--seed", type=int, default=20260730)
    return value


if __name__ == "__main__":
    print(json.dumps(safe(run(**vars(parser().parse_args()))), sort_keys=True))
