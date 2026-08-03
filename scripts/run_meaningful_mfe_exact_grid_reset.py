#!/usr/bin/env python3
"""Run the exact-grid meaningful-MFE event/capture reset.

This is a diagnostic runner.  It binds the signed point-in-time capture
feature universe one-to-one to the exact ``h12_u1p5atr`` label grid, selects
model geometry only on a purged May inner split, and freezes that geometry for:

* May -> June;
* June -> July 1--10;
* July 1--10 -> June 1--10 and full June (reverse-time diagnostics); and
* five purged grouped-day OOF folds within July 1--10.

The primary selection remains one pooled global top 10%, with deterministic
candidate-ID tie breaking.  Reverse-time and within-July results are
non-promotable regime-learnability diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_july_exact_preentry_heads import IDENTITY  # noqa: E402
from scripts.run_historical_to_july_meaningful_mfe_gate_challenger import (  # noqa: E402
    classification_metrics,
    fit_model,
    predict_model,
    select_features_nested,
    sha256,
)


SCHEMA = "meaningful_mfe_exact_grid_reset_v1"
GRID_NAME = "h12_u1p5atr"
RAW_PREFIX = "capture_candidate__"
SIDES = ("long", "short")
TASKS: Mapping[str, Mapping[str, Any]] = {
    "any_touch": {
        "fit_target": "any_touch",
        "metric_target": "any_touch",
        "condition": None,
        "soft": False,
    },
    "clean_first": {
        "fit_target": "clean_first",
        "metric_target": "clean_first",
        "condition": None,
        "soft": False,
    },
    "capture_given_touch": {
        "fit_target": "positive_net",
        "metric_target": "positive_net",
        "condition": "any_touch",
        "soft": False,
    },
    "capture_given_clean": {
        "fit_target": "positive_net",
        "metric_target": "positive_net",
        "condition": "clean_first",
        "soft": False,
    },
    "soft_triple_barrier": {
        "fit_target": "soft_label",
        "metric_target": "clean_first",
        "condition": None,
        "soft": True,
    },
}
MODEL_GRIDS: Mapping[str, tuple[Mapping[str, Any], ...]] = {
    "logistic": ({"C": 0.10}, {"C": 1.0}),
    "lightgbm": (
        {
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 250,
            "reg_lambda": 8.0,
        },
        {
            "num_leaves": 31,
            "max_depth": 7,
            "min_child_samples": 150,
            "reg_lambda": 12.0,
        },
    ),
    "catboost": (
        {"depth": 5, "l2_leaf_reg": 8.0},
        {"depth": 7, "l2_leaf_reg": 12.0},
    ),
}

MAY_START = pd.Timestamp("2026-05-01T00:00:00Z")
JUNE_START = pd.Timestamp("2026-06-01T00:00:00Z")
JULY_START = pd.Timestamp("2026-07-01T00:00:00Z")
JULY_DIAGNOSTIC_END = pd.Timestamp("2026-07-11T00:00:00Z")
JULY_RESOLUTION_CUTOFF = pd.Timestamp("2026-07-11T12:00:00Z")
MAY_HPO_START = pd.Timestamp("2026-05-25T00:00:00Z")


@dataclass(frozen=True)
class TransferSpec:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    train_resolution_cutoff: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    promotable: bool
    note: str


TRANSFER_SPECS = (
    TransferSpec(
        "may_to_june",
        MAY_START,
        JUNE_START,
        JUNE_START,
        JUNE_START,
        JULY_START,
        True,
        "chronological transfer",
    ),
    TransferSpec(
        "june_to_july",
        JUNE_START,
        JULY_START,
        JULY_START,
        JULY_START,
        JULY_DIAGNOSTIC_END,
        True,
        "chronological transfer to canonical July 1-10 panel",
    ),
    TransferSpec(
        "july_to_june_matched",
        JULY_START,
        JULY_DIAGNOSTIC_END,
        JULY_RESOLUTION_CUTOFF,
        JUNE_START,
        pd.Timestamp("2026-06-11T00:00:00Z"),
        False,
        "REVERSE_TIME_DIAGNOSTIC_NONPROMOTABLE duration-matched",
    ),
    TransferSpec(
        "july_to_june_full",
        JULY_START,
        JULY_DIAGNOSTIC_END,
        JULY_RESOLUTION_CUTOFF,
        JUNE_START,
        JULY_START,
        False,
        "REVERSE_TIME_DIAGNOSTIC_NONPROMOTABLE duration-mismatched",
    ),
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_unique(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} identity columns missing: {missing}")
    work = frame.copy()
    work["__ts__"] = pd.to_datetime(work["__ts__"], utc=True, errors="raise")
    if work.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{name} contains duplicate identities")
    return work


def derive_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive distinct exact-grid opportunity and capture targets."""

    work = frame.copy()
    peak_return = (
        pd.to_numeric(work["peak_mfe_atr"], errors="raise").to_numpy(float)
        * pd.to_numeric(
            work["oof_entry_atr_fraction"], errors="raise"
        ).to_numpy(float)
    )
    upper = pd.to_numeric(work["upper_return"], errors="raise").to_numpy(float)
    work["any_touch"] = (peak_return + 1e-10 >= upper).astype(np.int8)
    work["clean_first"] = pd.to_numeric(
        work["favorable_first"], errors="raise"
    ).astype(np.int8)
    work["positive_net"] = (
        pd.to_numeric(work["execution_net_ev_12h"], errors="raise") > 0.0
    ).astype(np.int8)
    work["soft_label"] = np.clip(
        pd.to_numeric(work["soft_label"], errors="raise"), 0.0, 1.0
    )
    if bool((work["clean_first"] > work["any_touch"]).any()):
        raise ValueError("clean-first event cannot occur without an upper touch")
    return work


def load_panel(
    feature_path: Path,
    feature_manifest_path: Path,
    grid_path: Path,
    grid_manifest_path: Path,
    *,
    grid_name: str = GRID_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("schema") != (
        "exact_policy_capture_causal_feature_universe_v1"
    ):
        raise ValueError("unexpected exact feature-universe schema")
    if sha256(feature_path) != feature_manifest["outputs"]["universe"]["sha256"]:
        raise ValueError("exact feature-universe hash mismatch")
    if feature_manifest.get("contract", {}).get("point_in_time") != (
        "immutable causal feature-store value at candidate signal __ts__"
    ):
        raise ValueError("feature universe is not proven point-in-time")
    universe_manifest_path = Path(
        feature_manifest["outputs"]["feature_manifest"]["path"]
    )
    universe_manifest = json.loads(
        universe_manifest_path.read_text(encoding="utf-8")
    )
    eligible = list(
        map(str, universe_manifest["eligible_full_period_feature_columns"])
    )
    if (
        len(eligible) != 249
        or len(set(eligible)) != len(eligible)
        or any(not value.startswith(RAW_PREFIX) for value in eligible)
    ):
        raise ValueError("eligible point-in-time feature contract changed")

    grid_manifest = json.loads(grid_manifest_path.read_text(encoding="utf-8"))
    if grid_manifest.get("schema") != "materialize_meaningful_mfe_label_grid_v1":
        raise ValueError("unexpected exact label-grid schema")
    if sha256(grid_path) != grid_manifest["outputs"]["labels"]["sha256"]:
        raise ValueError("exact label-grid hash mismatch")

    features = _require_unique(pd.read_parquet(feature_path), "exact features")
    grid = pd.read_parquet(grid_path)
    grid = grid.loc[
        grid["grid_name"].astype(str).eq(grid_name)
        & grid["label_valid"].astype(bool)
    ].copy()
    grid = _require_unique(grid, f"{grid_name} labels")
    feature_ids = set(map(tuple, features[list(IDENTITY)].itertuples(False, None)))
    grid_ids = set(map(tuple, grid[list(IDENTITY)].itertuples(False, None)))
    if feature_ids != grid_ids:
        raise ValueError(
            "feature/grid identity mismatch: "
            f"feature_only={len(feature_ids-grid_ids)}, grid_only={len(grid_ids-feature_ids)}"
        )
    label_columns = [
        *IDENTITY,
        "execution_decision_utc",
        "oof_entry_atr_fraction",
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "label_resolution_utc",
        "soft_label",
        "favorable_first",
        "adverse_first",
        "timeout",
        "upper_return",
        "peak_mfe_atr",
        "time_to_80pct_mfe_hours",
        "future_close_slope_atr_per_hour",
    ]
    anchor_columns = (
        "execution_decision_utc",
        "oof_entry_atr_fraction",
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
    )
    duplicate_outcomes = [
        column
        for column in label_columns
        if column not in IDENTITY
        and column not in anchor_columns
        and column in features.columns
    ]
    if duplicate_outcomes:
        features = features.drop(columns=duplicate_outcomes)
    grid_labels = grid[label_columns].rename(
        columns={column: f"grid__{column}" for column in anchor_columns}
    )
    panel = features.merge(
        grid_labels,
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    decision = pd.to_datetime(
        panel["execution_decision_utc"], utc=True, errors="raise"
    )
    grid_decision = pd.to_datetime(
        panel["grid__execution_decision_utc"], utc=True, errors="raise"
    )
    if not bool(decision.eq(grid_decision).all()):
        raise ValueError("grid/feature decision timestamps differ")
    for column in (
        "oof_entry_atr_fraction",
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
    ):
        left = pd.to_numeric(panel[column], errors="raise").to_numpy(float)
        right = pd.to_numeric(panel[f"grid__{column}"], errors="raise").to_numpy(
            float
        )
        if not np.allclose(left, right, atol=1e-10, rtol=1e-10):
            raise ValueError(f"grid/feature exact anchor differs: {column}")
    panel = panel.drop(
        columns=[f"grid__{column}" for column in anchor_columns]
    )
    panel["label_resolution_utc"] = pd.to_datetime(
        panel["label_resolution_utc"], utc=True, errors="raise"
    )
    panel["execution_label_end_utc"] = pd.to_datetime(
        panel["execution_label_end_utc"], utc=True, errors="raise"
    )
    if not bool(
        panel["label_resolution_utc"].eq(panel["execution_label_end_utc"]).all()
    ):
        raise ValueError("grid and execution-policy label resolution differ")
    expected_resolution = decision + pd.Timedelta(hours=12)
    if not bool(panel["label_resolution_utc"].eq(expected_resolution).all()):
        raise ValueError("exact label resolution is not decision + 12h")
    gross = pd.to_numeric(
        panel["execution_gross_ev_12h"], errors="raise"
    ).to_numpy(float)
    cost = pd.to_numeric(
        panel["execution_cost_return"], errors="raise"
    ).to_numpy(float)
    net = pd.to_numeric(
        panel["execution_net_ev_12h"], errors="raise"
    ).to_numpy(float)
    if not np.allclose(gross - cost, net, atol=1e-7, rtol=0.0):
        raise ValueError("full-panel exact gross-cost-net identity failed")
    panel = derive_targets(panel)
    matrix = panel[eligible].copy()
    matrix.columns = [value.removeprefix(RAW_PREFIX) for value in eligible]
    matrix = matrix.astype(np.float32)
    if set(panel["side_name"].astype(str)) != set(SIDES):
        raise ValueError("both sides are required")
    lineage = {
        "features": {
            "path": feature_path,
            "sha256": sha256(feature_path),
            "manifest": feature_manifest_path,
            "manifest_sha256": sha256(feature_manifest_path),
            "universe_manifest": universe_manifest_path,
            "universe_manifest_sha256": sha256(universe_manifest_path),
        },
        "labels": {
            "path": grid_path,
            "sha256": sha256(grid_path),
            "manifest": grid_manifest_path,
            "manifest_sha256": sha256(grid_manifest_path),
            "grid_name": grid_name,
        },
        "rows": len(panel),
        "raw_feature_count": len(matrix.columns),
        "signal_start": panel["__ts__"].min(),
        "signal_end": panel["__ts__"].max(),
        "label_resolution_max": panel["label_resolution_utc"].max(),
    }
    return panel.reset_index(drop=True), matrix.reset_index(drop=True), list(
        matrix.columns
    ), lineage


def _task_rows(
    panel: pd.DataFrame, positions: np.ndarray, task: Mapping[str, Any]
) -> np.ndarray:
    if task["condition"] is None:
        return positions
    condition = panel[task["condition"]].to_numpy(int).astype(bool)
    return positions[condition[positions]]


def _classification_with_tail(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    result = classification_metrics(target, prediction)
    count = max(1, int(math.ceil(len(target) * 0.10)))
    order = np.argsort(-np.asarray(prediction), kind="stable")[:count]
    positives = int(np.asarray(target, dtype=int).sum())
    result.update(
        {
            "top10_rows": count,
            "top10_precision": float(np.asarray(target)[order].mean()),
            "top10_recall": (
                float(np.asarray(target)[order].sum() / positives)
                if positives
                else float("nan")
            ),
        }
    )
    return result


def stable_top(
    frame: pd.DataFrame, score_column: str, fraction: float = 0.10
) -> pd.DataFrame:
    local = frame.loc[np.isfinite(frame[score_column])].copy()
    count = max(1, int(math.ceil(len(local) * fraction)))
    return local.sort_values(
        [
            score_column,
            "candidate_id",
            "__ts__",
            "__symbol__",
            "side_name",
        ],
        ascending=[False, True, True, True, True],
        kind="stable",
    ).iloc[:count]


def economic_metrics(
    frame: pd.DataFrame,
    score_column: str,
    *,
    scope: str,
    side: str,
) -> dict[str, Any]:
    local = (
        frame
        if side == "pooled"
        else frame.loc[frame["side_name"].astype(str).eq(side)]
    )
    selected = stable_top(local, score_column)
    net = pd.to_numeric(selected["execution_net_ev_12h"], errors="raise")
    gross = pd.to_numeric(selected["execution_gross_ev_12h"], errors="raise")
    cost = pd.to_numeric(selected["execution_cost_return"], errors="raise")
    if not np.allclose(
        gross.to_numpy(float) - cost.to_numpy(float),
        net.to_numpy(float),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("exact cost identity failed")
    return {
        "evaluation": scope,
        "side": side,
        "score": score_column,
        "population_rows": len(local),
        "selected_rows": len(selected),
        "selected_fraction": len(selected) / len(local),
        "net_ev_bps": float(net.mean() * 1e4),
        "gross_ev_bps": float(gross.mean() * 1e4),
        "cost_bps": float(cost.mean() * 1e4),
        "positive_net_rate": float((net > 0.0).mean()),
        "cvar5_bps": float(net.nsmallest(max(1, math.ceil(len(net) * 0.05))).mean() * 1e4),
        "any_touch_rate": float(selected["any_touch"].mean()),
        "clean_first_rate": float(selected["clean_first"].mean()),
        "timeout_rate": float(selected["timeout"].mean()),
        "asset_count": int(selected["__symbol__"].nunique()),
        "largest_asset_share": float(
            selected["__symbol__"].value_counts(normalize=True).max()
        ),
    }


def _base_masks(
    panel: pd.DataFrame, spec: TransferSpec
) -> tuple[np.ndarray, np.ndarray]:
    signal = panel["__ts__"]
    resolved = panel["label_resolution_utc"]
    decision = pd.to_datetime(
        panel["execution_decision_utc"], utc=True, errors="raise"
    )
    train = np.flatnonzero(
        signal.ge(spec.train_start).to_numpy()
        & signal.lt(spec.train_end).to_numpy()
        & resolved.lt(spec.train_resolution_cutoff).to_numpy()
        & decision.lt(
            spec.train_resolution_cutoff - pd.Timedelta(hours=12)
        ).to_numpy()
    )
    evaluation = np.flatnonzero(
        signal.ge(spec.evaluation_start).to_numpy()
        & signal.lt(spec.evaluation_end).to_numpy()
    )
    if not len(train) or not len(evaluation):
        raise ValueError(f"{spec.name} has empty train/evaluation support")
    if spec.promotable and not bool(
        (resolved.iloc[train] < spec.evaluation_start).all()
    ):
        raise ValueError(f"{spec.name} contains unresolved training labels")
    if spec.promotable and not bool(
        (
            decision.iloc[train]
            < spec.evaluation_start - pd.Timedelta(hours=12)
        ).all()
    ):
        raise ValueError(f"{spec.name} violates the decision-time 12h purge")
    return train, evaluation


def july_grouped_day_folds(
    panel: pd.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray, list[str]]]:
    signal = pd.to_datetime(panel["__ts__"], utc=True)
    july = signal.ge(JULY_START) & signal.lt(JULY_DIAGNOSTIC_END)
    days = signal.dt.floor("D")
    unique_days = sorted(days.loc[july].unique())
    if len(unique_days) != 10:
        raise ValueError(f"expected 10 complete July diagnostic days, got {len(unique_days)}")
    folds: list[tuple[str, np.ndarray, np.ndarray, list[str]]] = []
    for fold in range(5):
        validation_days = [
            pd.Timestamp(unique_days[2 * fold]),
            pd.Timestamp(unique_days[2 * fold + 1]),
        ]
        validation_mask = july & days.isin(validation_days)
        exclusion = np.zeros(len(panel), dtype=bool)
        for day in validation_days:
            exclusion |= (
                signal.ge(day - pd.Timedelta(hours=12)).to_numpy()
                & signal.lt(day + pd.Timedelta(days=1, hours=12)).to_numpy()
            )
        train_mask = july.to_numpy() & ~exclusion
        train = np.flatnonzero(train_mask)
        validation = np.flatnonzero(validation_mask.to_numpy())
        if not len(train) or not len(validation):
            raise ValueError(f"July fold {fold} has empty support")
        if set(days.iloc[train]).intersection(set(days.iloc[validation])):
            raise ValueError(f"July fold {fold} shares UTC days")
        for day in validation_days:
            if bool(
                (
                    signal.iloc[train].ge(day - pd.Timedelta(hours=12))
                    & signal.iloc[train].lt(day + pd.Timedelta(days=1, hours=12))
                ).any()
            ):
                raise ValueError(f"July fold {fold} violates 12h purge/embargo")
        folds.append(
            (
                f"july_group_oof_fold_{fold}",
                train,
                validation,
                [day.date().isoformat() for day in validation_days],
            )
        )
    covered = np.concatenate([validation for _, _, validation, _ in folds])
    expected = np.flatnonzero(july.to_numpy())
    if len(covered) != len(expected) or set(covered) != set(expected):
        raise ValueError("July grouped-day validation is not an exact partition")
    return folds


def aggregate_july_head_metrics(
    scored: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in MODEL_GRIDS:
        for side in SIDES:
            local = scored.loc[scored["side_name"].astype(str).eq(side)]
            for task_name, task in TASKS.items():
                metric_local = (
                    local
                    if task["condition"] is None
                    else local.loc[local[task["condition"]].astype(bool)]
                )
                prediction = metric_local[f"p_{family}_{task_name}"].to_numpy(float)
                metrics = _classification_with_tail(
                    metric_local[task["metric_target"]].to_numpy(int),
                    prediction,
                )
                rows.append(
                    {
                        "evaluation": "july_grouped_oof",
                        "family": family,
                        "side": side,
                        "task": task_name,
                        "train_rows": np.nan,
                        "evaluation_rows": len(metric_local),
                        "training_label_resolution_max": pd.NaT,
                        "validation_days": "five_contiguous_two_day_blocks",
                        **metrics,
                    }
                )
    return rows


def _inner_may_masks(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    signal = panel["__ts__"]
    resolved = panel["label_resolution_utc"]
    decision = pd.to_datetime(
        panel["execution_decision_utc"], utc=True, errors="raise"
    )
    train = np.flatnonzero(
        signal.ge(MAY_START).to_numpy()
        & signal.lt(MAY_HPO_START).to_numpy()
        & resolved.lt(MAY_HPO_START).to_numpy()
        & decision.lt(MAY_HPO_START - pd.Timedelta(hours=12)).to_numpy()
    )
    validation = np.flatnonzero(
        signal.ge(MAY_HPO_START).to_numpy()
        & signal.lt(JUNE_START).to_numpy()
    )
    return train, validation


def _trial_score(
    metrics: Mapping[str, float], net_bps: float | None
) -> float:
    auc = float(metrics["auc"]) if math.isfinite(float(metrics["auc"])) else 0.5
    ap = (
        float(metrics["pr_auc"])
        if math.isfinite(float(metrics["pr_auc"]))
        else float(metrics["prevalence"])
    )
    clipped_net = (
        float(np.clip(net_bps, -500.0, 500.0))
        if net_bps is not None and math.isfinite(net_bps)
        else 0.0
    )
    return float(
        metrics["log_loss"]
        + metrics["brier"]
        - 0.10 * auc
        - 0.10 * ap
        - (0.0001 * clipped_net if net_bps is not None else 0.0)
    )


def select_frozen_geometries(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    feature_count: int,
    seed: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], pd.DataFrame]:
    inner_train, inner_validation = _inner_may_masks(panel)
    winners: dict[str, dict[str, dict[str, Any]]] = {
        family: {} for family in MODEL_GRIDS
    }
    rows: list[dict[str, Any]] = []
    selection_cache: dict[tuple[str, str], tuple[list[str], pd.DataFrame]] = {}
    for family_index, (family, geometries) in enumerate(MODEL_GRIDS.items()):
        for side_index, side in enumerate(SIDES):
            side_mask = panel["side_name"].astype(str).eq(side).to_numpy()
            for task_index, (task_name, task) in enumerate(TASKS.items()):
                train = _task_rows(
                    panel, inner_train[side_mask[inner_train]], task
                )
                metric_validation = _task_rows(
                    panel,
                    inner_validation[side_mask[inner_validation]],
                    task,
                )
                if min(len(train), len(metric_validation)) < 500:
                    raise ValueError(
                        f"insufficient May HPO support for {family}/{side}/{task_name}"
                    )
                cache_key = (side, task_name)
                cached = selection_cache.get(cache_key)
                if cached is None:
                    cached = select_features_nested(
                        matrix,
                        panel[task["fit_target"]].to_numpy(float),
                        train,
                        feature_count,
                    )
                    selection_cache[cache_key] = cached
                selected, _ = cached
                candidates: list[dict[str, Any]] = []
                for geometry_index, geometry in enumerate(geometries):
                    model = fit_model(
                        family,
                        geometry,
                        matrix.iloc[train][selected],
                        panel.iloc[train][task["fit_target"]].to_numpy(float),
                        soft=bool(task["soft"]),
                        seed=(
                            seed
                            + family_index * 100_000
                            + side_index * 10_000
                            + task_index * 100
                            + geometry_index
                        ),
                    )
                    all_validation = inner_validation[
                        side_mask[inner_validation]
                    ]
                    prediction = predict_model(
                        model, family, matrix.iloc[all_validation][selected]
                    )
                    metric_lookup = pd.Series(
                        np.arange(len(all_validation)), index=all_validation
                    )
                    metric_positions = metric_lookup.loc[metric_validation].to_numpy(int)
                    metrics = _classification_with_tail(
                        panel.iloc[metric_validation][task["metric_target"]].to_numpy(int),
                        prediction[metric_positions],
                    )
                    net_bps: float | None = None
                    economic_scope = "excluded_for_uncomposed_conditional_head"
                    if task["condition"] is None:
                        trial_frame = panel.iloc[all_validation].copy()
                        trial_frame["_score"] = prediction
                        net_bps = economic_metrics(
                            trial_frame,
                            "_score",
                            scope="may_hpo_side_local",
                            side="pooled",
                        )["net_ev_bps"]
                        economic_scope = "side_local_top10_secondary"
                    objective = _trial_score(metrics, net_bps)
                    record = {
                        "family": family,
                        "side": side,
                        "task": task_name,
                        "geometry_index": geometry_index,
                        "params": dict(geometry),
                        "train_rows": len(train),
                        "validation_rows": len(metric_validation),
                        "selected_feature_count": len(selected),
                        "selected_features": selected,
                        "net_ev_bps": net_bps,
                        "economic_scope": economic_scope,
                        "objective": objective,
                        **metrics,
                    }
                    rows.append(record)
                    candidates.append(record)
                winner = min(
                    candidates,
                    key=lambda row: (
                        row["objective"],
                        -(
                            row["net_ev_bps"]
                            if row["net_ev_bps"] is not None
                            else -math.inf
                        ),
                        row["geometry_index"],
                    ),
                )
                winners[family].setdefault(side, {})[task_name] = {
                    "params": winner["params"],
                    "selected_features": winner["selected_features"],
                    "hpo_objective": winner["objective"],
                    "hpo_net_ev_bps": winner["net_ev_bps"],
                }
    return winners, pd.DataFrame(rows)


def _fit_predict_split(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    train: np.ndarray,
    evaluation: np.ndarray,
    winners: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    split_name: str,
    seed: int,
    validation_days: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    keep_columns = [
        *IDENTITY,
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_exit_reason",
        "execution_mfe_return_12h",
        "execution_mae_return_12h",
        "any_touch",
        "clean_first",
        "positive_net",
        "adverse_first",
        "timeout",
        "soft_label",
    ]
    scored = panel.iloc[evaluation][keep_columns].copy().reset_index(drop=True)
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selection_cache: dict[
        tuple[str, str], tuple[list[str], pd.DataFrame]
    ] = {}
    for family_index, family in enumerate(MODEL_GRIDS):
        for side_index, side in enumerate(SIDES):
            side_train = train[
                panel.iloc[train]["side_name"].astype(str).eq(side).to_numpy()
            ]
            side_evaluation = evaluation[
                panel.iloc[evaluation]["side_name"].astype(str).eq(side).to_numpy()
            ]
            output_positions = np.flatnonzero(
                scored["side_name"].astype(str).eq(side).to_numpy()
            )
            if len(side_evaluation) != len(output_positions):
                raise AssertionError("side output position mismatch")
            for task_index, (task_name, task) in enumerate(TASKS.items()):
                task_train = _task_rows(panel, side_train, task)
                task_evaluation = _task_rows(panel, side_evaluation, task)
                if len(task_train) < 500 or len(task_evaluation) < 100:
                    raise ValueError(
                        f"insufficient support {split_name}/{family}/{side}/{task_name}: "
                        f"train={len(task_train)}, eval={len(task_evaluation)}"
                    )
                frozen = winners[family][side][task_name]
                cache_key = (side, task_name)
                cached = selection_cache.get(cache_key)
                if cached is None:
                    cached = select_features_nested(
                        matrix,
                        panel[task["fit_target"]].to_numpy(float),
                        task_train,
                        len(frozen["selected_features"]),
                    )
                    selection_cache[cache_key] = cached
                selected, screen = cached
                model = fit_model(
                    family,
                    frozen["params"],
                    matrix.iloc[task_train][selected],
                    panel.iloc[task_train][task["fit_target"]].to_numpy(float),
                    soft=bool(task["soft"]),
                    seed=(
                        seed
                        + family_index * 100_000
                        + side_index * 10_000
                        + task_index * 100
                    ),
                )
                prediction = predict_model(
                    model, family, matrix.iloc[side_evaluation][selected]
                )
                column = f"p_{family}_{task_name}"
                if column not in scored:
                    scored[column] = np.nan
                scored.loc[output_positions, column] = prediction
                lookup = pd.Series(
                    np.arange(len(side_evaluation)), index=side_evaluation
                )
                metric_positions = lookup.loc[task_evaluation].to_numpy(int)
                metrics = _classification_with_tail(
                    panel.iloc[task_evaluation][task["metric_target"]].to_numpy(int),
                    prediction[metric_positions],
                )
                metric_rows.append(
                    {
                        "evaluation": split_name,
                        "family": family,
                        "side": side,
                        "task": task_name,
                        "train_rows": len(task_train),
                        "evaluation_rows": len(task_evaluation),
                        "training_label_resolution_max": panel.iloc[task_train][
                            "label_resolution_utc"
                        ].max(),
                        "validation_days": (
                            "|".join(validation_days) if validation_days else ""
                        ),
                        **metrics,
                    }
                )
                selection_rows.append(
                    {
                        "evaluation": split_name,
                        "family": family,
                        "side": side,
                        "task": task_name,
                        "selected_feature_count": len(selected),
                        "selected_features": json.dumps(selected),
                        "screen_top20": json.dumps(
                            _safe(screen.head(20).to_dict("records")),
                            sort_keys=True,
                        ),
                    }
                )
    for family in MODEL_GRIDS:
        scored[f"score_{family}_touch_capture"] = (
            scored[f"p_{family}_any_touch"]
            * scored[f"p_{family}_capture_given_touch"]
        )
        scored[f"score_{family}_clean_capture"] = (
            scored[f"p_{family}_clean_first"]
            * scored[f"p_{family}_capture_given_clean"]
        )
    scored["evaluation"] = split_name
    return scored, metric_rows, selection_rows


def _economic_table(
    scored: pd.DataFrame, evaluation: str
) -> list[dict[str, Any]]:
    score_columns = [
        column
        for column in scored.columns
        if column.startswith("p_") or column.startswith("score_")
    ]
    return [
        economic_metrics(
            scored,
            score,
            scope=evaluation,
            side=side,
        )
        for score in score_columns
        for side in ("pooled", *SIDES)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, matrix, raw_features, lineage = load_panel(
        args.features,
        args.feature_manifest,
        args.grid,
        args.grid_manifest,
        grid_name=args.grid_name,
    )
    winners, hpo = select_frozen_geometries(
        panel,
        matrix,
        feature_count=args.feature_count,
        seed=args.seed,
    )
    all_scored: list[pd.DataFrame] = []
    all_head_metrics: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    all_economics: list[dict[str, Any]] = []
    split_report: list[dict[str, Any]] = []

    for index, spec in enumerate(TRANSFER_SPECS):
        train, evaluation = _base_masks(panel, spec)
        scored, head_metrics, selections = _fit_predict_split(
            panel,
            matrix,
            train,
            evaluation,
            winners,
            split_name=spec.name,
            seed=args.seed + index * 1_000_000,
        )
        all_scored.append(scored)
        all_head_metrics.extend(head_metrics)
        all_selections.extend(selections)
        all_economics.extend(_economic_table(scored, spec.name))
        split_report.append(
            {
                "name": spec.name,
                "train_rows": len(train),
                "evaluation_rows": len(evaluation),
                "train_start": spec.train_start,
                "train_end": spec.train_end,
                "training_label_resolution_max": panel.iloc[train][
                    "label_resolution_utc"
                ].max(),
                "evaluation_start": spec.evaluation_start,
                "evaluation_end": spec.evaluation_end,
                "promotion_eligible": spec.promotable,
                "note": spec.note,
            }
        )

    july_scored: list[pd.DataFrame] = []
    for fold_index, (name, train, evaluation, validation_days) in enumerate(
        july_grouped_day_folds(panel)
    ):
        scored, head_metrics, selections = _fit_predict_split(
            panel,
            matrix,
            train,
            evaluation,
            winners,
            split_name=name,
            seed=args.seed + 10_000_000 + fold_index * 1_000_000,
            validation_days=validation_days,
        )
        july_scored.append(scored)
        all_head_metrics.extend(head_metrics)
        all_selections.extend(selections)
        split_report.append(
            {
                "name": name,
                "train_rows": len(train),
                "evaluation_rows": len(evaluation),
                "validation_days": validation_days,
                "promotion_eligible": False,
                "note": "GROUPED_DAY_JULY_OOF_LEARNABILITY_NONPROMOTABLE",
            }
        )
    july_oof = pd.concat(july_scored, ignore_index=True)
    july_oof["evaluation"] = "july_grouped_oof"
    if july_oof.duplicated(list(IDENTITY)).any():
        raise ValueError("July grouped OOF produced duplicate identities")
    all_scored.append(july_oof)
    all_head_metrics.extend(aggregate_july_head_metrics(july_oof))
    all_economics.extend(_economic_table(july_oof, "july_grouped_oof"))

    predictions = pd.concat(all_scored, ignore_index=True)
    head_metrics = pd.DataFrame(all_head_metrics)
    feature_selections = pd.DataFrame(all_selections)
    economics = pd.DataFrame(all_economics)
    outputs: dict[str, Any] = {}
    for name, frame in (
        ("predictions", predictions),
        ("head_metrics", head_metrics),
        ("economics", economics),
        ("hpo_trials", hpo),
        ("feature_selections", feature_selections),
    ):
        path = args.output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        outputs[name] = {
            "path": path,
            "rows": len(frame),
            "sha256": sha256(path),
        }
    report = {
        "schema": SCHEMA,
        "status": "COMPLETED_DIAGNOSTIC_EXACT_GRID_NO_PROMOTION",
        "promotion_eligible": False,
        "runner": {
            "path": Path(__file__).resolve(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "lineage": lineage,
        "contracts": {
            "any_touch": (
                "peak_mfe_atr * decision_atr_fraction >= exact grid upper_return"
            ),
            "clean_first": "favorable upper barrier first under competing risk",
            "conditional_capture": (
                "exact-policy net > 0, trained only on the declared touch/clean event"
            ),
            "soft_baseline": (
                "ATR-normalized h12_u1p5atr soft triple-barrier label"
            ),
            "cost": (
                "execution gross already embeds executable spread; explicit "
                "row cost is subtracted exactly once in execution net"
            ),
            "hpo": (
                "model geometry selected only on purged May 1-24 -> May 25-31; "
                "frozen for every transfer and grouped-July evaluation; "
                "per-side tail economics is secondary for unconditional heads "
                "and excluded for uncomposed conditional heads"
            ),
            "selection": (
                "one pooled global top 10% per score with score descending and "
                "candidate-ID ascending tiebreak; side tails diagnostic only"
            ),
            "reverse_time": (
                "July->June is permanently diagnostic/nonpromotable"
            ),
            "july_oof": (
                "five grouped UTC-day folds with +/-12h exclusion; learnability "
                "diagnostic, not deployment validation"
            ),
        },
        "raw_feature_count": len(raw_features),
        "frozen_winners": winners,
        "splits": split_report,
        "outputs": outputs,
    }
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": SCHEMA,
        "status": report["status"],
        "promotion_eligible": False,
        "report": {"path": report_path, "sha256": sha256(report_path)},
        "outputs": outputs,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/"
            "capture_feature_universe.parquet"
        ),
    )
    value.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/"
            "manifest.json"
        ),
    )
    value.add_argument(
        "--grid",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
            "meaningful_mfe_label_grid.parquet"
        ),
    )
    value.add_argument(
        "--grid-manifest",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/"
            "manifest.json"
        ),
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data_perp/artifacts/meaningful_mfe_exact_grid_reset_20260730_v2"
        ),
    )
    value.add_argument(
        "--grid-name",
        choices=("h12_u1p5atr", "h12_u2p0atr"),
        default=GRID_NAME,
    )
    value.add_argument("--feature-count", type=int, default=64)
    value.add_argument("--seed", type=int, default=20260730)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
