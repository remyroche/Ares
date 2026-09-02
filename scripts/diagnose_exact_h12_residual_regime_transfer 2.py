#!/usr/bin/env python3
"""Attach frozen causal context to the exact-H12 residual OOF experiment.

The diagnostic separates market-period recognition from economic routing.
It tests whether regime, transition, and trajectory probabilities can identify
July and whether the same fields can predict when exact-H12 residual or direct
q25 global-book contributions beat their comparator.  All classifiers are
fixed grouped-OOF diagnostics; none is a trading gate or promotion candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "exact_h12_residual_regime_transfer_diagnostic_v1"
INPUT_SCHEMA = "exact_h12_side_local_residual_oof_v2"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
NET = "execution_net_ev_12h"
SIDES = ("long", "short")
DEFAULT_RESIDUAL = (
    ROOT
    / "data_perp/artifacts/exact_h12_side_local_residual_oof_20260730_v2"
)
DEFAULT_WATERFALL = (
    ROOT
    / "data_perp/artifacts/mayjul2026_exact_allscore_ic_ev_waterfall_20260730_v1"
)
DEFAULT_CONTEXT = (
    ROOT
    / "data_perp/artifacts/authoritative_soft_regime_transition_sidecars_20260730_v1"
)
DEFAULT_TRAJECTORY = (
    ROOT
    / "data_perp/artifacts/hourly_trajectory_transition_soft_sidecar_20260730_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data_perp/artifacts/exact_h12_residual_regime_transfer_diagnostic_20260730_v1"
)

REGIME_FIELDS = (
    "bocpd__change_probability_mean",
    "bocpd__change_probability_max",
    "bocpd__run_length_mean",
    "bocpd__run_length_q05",
    "bocpd__run_length_entropy",
    "bocpd__signal_count",
    "bocpd__state_age_hours",
    "bocpd__is_persistent_24h",
    "bocpd__is_persistent_72h",
)
TRANSITION_FIELDS = (
    "lgbm_transition_probability",
    "lgbm_entropy",
    "lgbm_margin",
    "bocpd_stable_vs_transition_probability",
    "bocpd_onset_h1_probability",
    "bocpd_onset_h3_probability",
    "bocpd_onset_h6_probability",
    "bocpd_onset_h12_probability",
)
TRAJECTORY_FIELDS = (
    "trajectory_transition_probability",
    "trajectory_probability_entropy",
    "trajectory_top2_margin",
    "trajectory_available_numeric",
)
ARMS: Mapping[str, tuple[str, ...]] = {
    "regime9": REGIME_FIELDS,
    "transition8": TRANSITION_FIELDS,
    "trajectory4": TRAJECTORY_FIELDS,
    "combined21": (*REGIME_FIELDS, *TRANSITION_FIELDS, *TRAJECTORY_FIELDS),
}
SCORE_NAMES = (
    "base_ev_exact_h12",
    "residual_exact_h12",
    "direct_q25_exact_h12",
)


class RegimeTransferError(RuntimeError):
    """Raised when a causal-context or diagnostic contract fails."""


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


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _verify_sealed_residual(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest_hash = root / "manifest.sha256"
    if not manifest_path.is_file() or not manifest_hash.is_file():
        raise RegimeTransferError("sealed exact-H12 residual input required")
    if sha256(manifest_path) != manifest_hash.read_text().split()[0]:
        raise RegimeTransferError("exact-H12 residual manifest seal changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != INPUT_SCHEMA
        or manifest.get("promotion_eligible") is not False
    ):
        raise RegimeTransferError("unexpected exact-H12 residual manifest")
    for record in manifest.get("outputs", {}).values():
        path = Path(str(record["path"]))
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RegimeTransferError(f"exact-H12 output changed: {path.name}")
    return manifest


def _verify_sidecar(root: Path, schema: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema or manifest.get("promotion_eligible") is not False:
        raise RegimeTransferError(f"unexpected sidecar schema under {root}")
    for name, digest in manifest.get("outputs_sha256", {}).items():
        path = root / name
        if path.is_file() and sha256(path) != digest:
            raise RegimeTransferError(f"sidecar output changed: {name}")
    return manifest


def _normalise_identity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["side_name"] = result["side_name"].astype(str)
    result["__symbol__"] = (
        result["__symbol__"].astype(str).str.replace("/", "_", regex=False)
    )
    result["__ts__"] = pd.to_datetime(result["__ts__"], utc=True, errors="raise")
    return result


def _trajectory_neutral_fill(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the preregistered neutral fill while preserving availability."""
    result = frame.copy()
    available = (
        result["trajectory_available"].fillna(False).astype(bool)
        & result["trajectory_transition_probability"].notna()
        & result["probability_entropy"].notna()
        & result["top2_margin"].notna()
    )
    result["trajectory_available_numeric"] = available.astype(float)
    result["trajectory_transition_probability"] = (
        pd.to_numeric(
            result["trajectory_transition_probability"], errors="coerce"
        )
        .where(available, 0.5)
        .astype(float)
    )
    result["trajectory_probability_entropy"] = (
        pd.to_numeric(result["probability_entropy"], errors="coerce")
        .where(available, math.log(2.0))
        .astype(float)
    )
    result["trajectory_top2_margin"] = (
        pd.to_numeric(result["top2_margin"], errors="coerce")
        .where(available, 0.0)
        .astype(float)
    )
    return result


def _load_panel(
    residual_root: Path,
    waterfall_root: Path,
    context_root: Path,
    trajectory_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    residual_manifest = _verify_sealed_residual(residual_root)
    context_manifest = _verify_sidecar(
        context_root, "authoritative_soft_regime_transition_sidecars_v1"
    )
    trajectory_manifest = _verify_sidecar(
        trajectory_root, "hourly_trajectory_transition_soft_sidecar_v1"
    )
    oof_path = residual_root / "oof_predictions.parquet"
    books_path = residual_root / "selection_books.parquet"
    waterfall_path = waterfall_root / "allscore_waterfall.parquet"
    waterfall_manifest_path = waterfall_root / "manifest.json"
    waterfall_manifest = json.loads(
        waterfall_manifest_path.read_text(encoding="utf-8")
    )
    if (
        waterfall_manifest.get("outputs", {})
        .get("allscore_waterfall", {})
        .get("sha256")
        != sha256(waterfall_path)
    ):
        raise RegimeTransferError("waterfall hash changed")
    oof = _normalise_identity(pd.read_parquet(oof_path))
    waterfall = _normalise_identity(pd.read_parquet(waterfall_path))
    candidate = waterfall.merge(
        oof.loc[
            :,
            [
                *IDENTITY,
                "score_exact_h12_base_ev_bps",
                "score_exact_h12_residual_delta_bps",
                "score_exact_h12_residual_bps",
                "residual_oof_fold",
                "residual_train_cutoff_utc",
                "residual_train_label_resolution_max",
                "is_strict_oof",
            ],
        ],
        on=list(IDENTITY),
        how="inner",
        validate="one_to_one",
    )
    if len(candidate) != 127_777 or not candidate["is_strict_oof"].all():
        raise RegimeTransferError("exact-H12 candidate panel coverage changed")
    regime_path = context_root / "soft_regime_hourly.parquet"
    transition_path = context_root / "soft_transition_hourly.parquet"
    trajectory_path = (
        trajectory_root / "hourly_trajectory_transition_soft_sidecar.parquet"
    )
    regime = pd.read_parquet(
        regime_path,
        columns=[
            "source_utc",
            *REGIME_FIELDS,
            "provenance_partition_bocpd",
            "train_end_exclusive_utc_bocpd",
        ],
    )
    transition = pd.read_parquet(
        transition_path,
        columns=[
            "source_utc",
            *TRANSITION_FIELDS,
            "provenance_partition_lgbm",
            "train_end_exclusive_utc_lgbm",
        ],
    )
    trajectory = pd.read_parquet(
        trajectory_path,
        columns=[
            "source_utc",
            "trajectory_available",
            "trajectory_transition_probability",
            "probability_entropy",
            "top2_margin",
            "provenance_partition",
            "fit_train_eras",
        ],
    )
    for frame in (regime, transition, trajectory):
        frame["source_utc"] = pd.to_datetime(
            frame["source_utc"], utc=True, errors="raise"
        )
        if frame["source_utc"].duplicated().any():
            raise RegimeTransferError("hourly context has duplicate timestamps")
    context = regime.merge(
        transition, on="source_utc", how="inner", validate="one_to_one"
    ).merge(trajectory, on="source_utc", how="left", validate="one_to_one")
    start, end = candidate["__ts__"].min(), candidate["__ts__"].max()
    context = context.loc[
        context["source_utc"].ge(start) & context["source_utc"].le(end)
    ].copy()
    if (
        len(context) != 1_704
        or context["source_utc"].min() != pd.Timestamp("2026-05-01T00:00:00Z")
        or context["source_utc"].max() != pd.Timestamp("2026-07-10T23:00:00Z")
    ):
        raise RegimeTransferError("context does not exactly cover evaluation hours")
    if set(context["provenance_partition_bocpd"].astype(str)) != {
        "untouched_2026_forward"
    }:
        raise RegimeTransferError("BOCPD context is not frozen forward")
    if set(context["provenance_partition_lgbm"].astype(str)) != {
        "untouched_2026_forward"
    }:
        raise RegimeTransferError("transition context is not frozen forward")
    for column in (
        "train_end_exclusive_utc_bocpd",
        "train_end_exclusive_utc_lgbm",
    ):
        cutoff = pd.to_datetime(context[column], utc=True, errors="raise")
        if not cutoff.le(pd.Timestamp("2026-01-01T00:00:00Z")).all():
            raise RegimeTransferError(f"context fit cutoff is not pre-2026: {column}")
    available_trajectory = context["trajectory_available"].fillna(False)
    available_partition = context.loc[
        available_trajectory, "provenance_partition"
    ].astype(str)
    if set(available_partition) != {"untouched_2026_frozen_fit"}:
        raise RegimeTransferError("available trajectory rows are not frozen forward")
    if set(
        context.loc[available_trajectory, "fit_train_eras"].astype(str)
    ) != {"2022,2023,2024,2025"}:
        raise RegimeTransferError("trajectory fit eras changed")
    context = _trajectory_neutral_fill(context)
    context_fields = list(ARMS["combined21"])
    if not np.isfinite(context[context_fields].to_numpy(float)).all():
        raise RegimeTransferError("causal context is not finite after neutral fill")
    panel = candidate.merge(
        context.loc[:, ["source_utc", *context_fields]],
        left_on="__ts__",
        right_on="source_utc",
        how="inner",
        validate="many_to_one",
    ).drop(columns="source_utc")
    if len(panel) != len(candidate):
        raise RegimeTransferError("candidate-context join lost rows")
    books = _normalise_identity(pd.read_parquet(books_path))
    books = books.loc[books["score_name"].isin(SCORE_NAMES)].copy()
    expected_books = sum(
        int(math.ceil(0.10 * rows))
        for rows in (63_351, 49_259, 15_167)
    ) * len(SCORE_NAMES)
    if len(books) != expected_books:
        raise RegimeTransferError(
            f"selected-book support changed: {len(books)} != {expected_books}"
        )
    evidence = {
        "residual_manifest": binding(residual_root / "manifest.json"),
        "residual_manifest_schema": residual_manifest["schema"],
        "oof_predictions": binding(oof_path),
        "selection_books": binding(books_path),
        "waterfall": binding(waterfall_path),
        "waterfall_manifest": binding(waterfall_manifest_path),
        "context_manifest": binding(context_root / "manifest.json"),
        "regime_sidecar": binding(regime_path),
        "transition_sidecar": binding(transition_path),
        "trajectory_manifest": binding(trajectory_root / "manifest.json"),
        "trajectory_sidecar": binding(trajectory_path),
        "context_manifest_schema": context_manifest["schema"],
        "trajectory_manifest_schema": trajectory_manifest["schema"],
        "candidate_rows": len(panel),
        "hourly_context_rows": len(context),
        "trajectory_available_hours": int(
            context["trajectory_available_numeric"].sum()
        ),
        "trajectory_neutral_fill_hours": int(
            context["trajectory_available_numeric"].eq(0.0).sum()
        ),
    }
    return panel, books, evidence


def _day_group(timestamp: pd.Series) -> np.ndarray:
    values = pd.to_datetime(timestamp, utc=True, errors="raise").astype("int64")
    return (values // int(pd.Timedelta(days=1).value)).to_numpy(np.int64)


def _week_group(timestamp: pd.Series) -> np.ndarray:
    values = pd.to_datetime(timestamp, utc=True, errors="raise").astype("int64")
    return (values // int(pd.Timedelta(days=7).value)).to_numpy(np.int64)


def _build_hour_side_panel(
    candidate: pd.DataFrame,
    books: pd.DataFrame,
) -> pd.DataFrame:
    context_fields = list(ARMS["combined21"])
    grouped = candidate.groupby("__ts__", observed=True, sort=True)
    for field in context_fields:
        if grouped[field].nunique(dropna=False).max() != 1:
            raise RegimeTransferError(f"context is not hourly invariant: {field}")
    hourly_context = grouped[context_fields].first().reset_index()
    grid = hourly_context.assign(_key=1).merge(
        pd.DataFrame({"side_name": list(SIDES), "_key": 1}),
        on="_key",
        validate="many_to_many",
    ).drop(columns="_key")
    grid["candidate_month"] = grid["__ts__"].dt.strftime("%Y-%m")
    book_sizes = (
        books.groupby(["candidate_month", "score_name"], observed=True)
        .size()
        .rename("monthly_book_rows")
        .reset_index()
    )
    contributions = (
        books.groupby(
            ["candidate_month", "score_name", "__ts__", "side_name"],
            observed=True,
            sort=True,
        )
        .agg(selected_rows=("candidate_id", "size"), selected_net_sum=(NET, "sum"))
        .reset_index()
        .merge(
            book_sizes,
            on=["candidate_month", "score_name"],
            how="left",
            validate="many_to_one",
        )
    )
    contributions["contribution_bps"] = (
        contributions["selected_net_sum"]
        / contributions["monthly_book_rows"]
        * 1e4
    )
    for score in SCORE_NAMES:
        part = contributions.loc[
            contributions["score_name"].eq(score),
            [
                "candidate_month",
                "__ts__",
                "side_name",
                "selected_rows",
                "contribution_bps",
            ],
        ].rename(
            columns={
                "selected_rows": f"{score}__selected_rows",
                "contribution_bps": f"{score}__contribution_bps",
            }
        )
        grid = grid.merge(
            part,
            on=["candidate_month", "__ts__", "side_name"],
            how="left",
            validate="one_to_one",
        )
        grid[
            [f"{score}__selected_rows", f"{score}__contribution_bps"]
        ] = grid[
            [f"{score}__selected_rows", f"{score}__contribution_bps"]
        ].fillna(
            0.0
        )
    grid["residual_minus_base_bps"] = (
        grid["residual_exact_h12__contribution_bps"]
        - grid["base_ev_exact_h12__contribution_bps"]
    )
    grid["direct_minus_residual_bps"] = (
        grid["direct_q25_exact_h12__contribution_bps"]
        - grid["residual_exact_h12__contribution_bps"]
    )
    grid["residual_or_base_selected"] = (
        grid["residual_exact_h12__selected_rows"].gt(0)
        | grid["base_ev_exact_h12__selected_rows"].gt(0)
    )
    grid["direct_or_residual_selected"] = (
        grid["direct_q25_exact_h12__selected_rows"].gt(0)
        | grid["residual_exact_h12__selected_rows"].gt(0)
    )
    grid["residual_advantage_positive"] = grid[
        "residual_minus_base_bps"
    ].gt(0).astype(np.int8)
    grid["direct_advantage_positive"] = grid[
        "direct_minus_residual_bps"
    ].gt(0).astype(np.int8)
    grid["is_july"] = grid["candidate_month"].eq("2026-07").astype(np.int8)
    return grid


def _auc(target: np.ndarray, probability: np.ndarray) -> float:
    return (
        float(roc_auc_score(target, probability))
        if np.unique(target).size == 2
        else np.nan
    )


def _grouped_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    group: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = frame.loc[:, list(features)].to_numpy(float)
    target = pd.to_numeric(frame[target_column], errors="raise").to_numpy(int)
    if not np.isfinite(matrix).all() or np.unique(target).size != 2:
        raise RegimeTransferError("classifier matrix invalid or target constant")
    unique_groups = np.unique(group)
    splits = min(5, len(unique_groups))
    if splits < 3:
        raise RegimeTransferError("insufficient grouped-CV support")
    splitter = StratifiedGroupKFold(
        n_splits=splits, shuffle=True, random_state=seed
    )
    probability = np.full(len(frame), np.nan)
    fold_id = np.full(len(frame), -1, dtype=int)
    folds: list[dict[str, Any]] = []
    for fold, (train, valid) in enumerate(
        splitter.split(matrix, target, groups=group)
    ):
        if set(group[train]).intersection(group[valid]):
            raise RegimeTransferError("grouped OOF overlaps groups")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                solver="lbfgs",
                random_state=seed + fold,
            ),
        )
        model.fit(matrix[train], target[train])
        probability[valid] = model.predict_proba(matrix[valid])[:, 1]
        fold_id[valid] = fold
        folds.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(valid),
                "train_groups": len(np.unique(group[train])),
                "validation_groups": len(np.unique(group[valid])),
                "validation_positive_rate": float(target[valid].mean()),
                "validation_auc": _auc(target[valid], probability[valid]),
            }
        )
    if np.isnan(probability).any() or (fold_id < 0).any():
        raise RegimeTransferError("grouped OOF coverage incomplete")
    predictions = frame.loc[
        :,
        [
            "__ts__",
            "side_name",
            "candidate_month",
            target_column,
            "residual_minus_base_bps",
            "direct_minus_residual_bps",
        ],
    ].copy()
    predictions["fold"] = fold_id
    predictions["probability"] = probability
    full_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2_000,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    full_model.fit(matrix, target)
    coefficients = pd.DataFrame(
        {
            "feature": list(features),
            "standardized_coefficient": full_model[-1].coef_[0],
        }
    )
    coefficients["absolute_coefficient"] = coefficients[
        "standardized_coefficient"
    ].abs()
    return predictions, pd.DataFrame(folds), coefficients


def _prediction_metrics(
    predictions: pd.DataFrame,
    *,
    task: str,
    arm: str,
    target_column: str,
    advantage_column: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target = predictions[target_column].to_numpy(int)
    probability = predictions["probability"].to_numpy(float)
    aggregate: dict[str, Any] = {
        "task": task,
        "arm": arm,
        "rows": len(predictions),
        "positive_rate": float(target.mean()),
        "grouped_oof_auc": _auc(target, probability),
        "grouped_oof_average_precision": float(
            average_precision_score(target, probability)
        ),
        "grouped_oof_brier": float(brier_score_loss(target, probability)),
        "grouped_oof_balanced_accuracy_at_0_5": float(
            balanced_accuracy_score(target, probability >= 0.5)
        ),
    }
    periods: list[dict[str, Any]] = []
    for month, local in predictions.groupby("candidate_month", sort=True):
        local_target = local[target_column].to_numpy(int)
        local_probability = local["probability"].to_numpy(float)
        row: dict[str, Any] = {
            "task": task,
            "arm": arm,
            "month": str(month),
            "rows": len(local),
            "positive_rate": float(local_target.mean()),
            "auc": _auc(local_target, local_probability),
        }
        if advantage_column is not None:
            threshold = float(np.quantile(local_probability, 0.90))
            selected = local.loc[local_probability >= threshold]
            row["mean_advantage_bps"] = float(local[advantage_column].mean())
            row["top_probability_decile_advantage_bps"] = float(
                selected[advantage_column].mean()
            )
            row["probability_rank_ic_advantage"] = float(
                local["probability"].corr(local[advantage_column], method="spearman")
            )
        periods.append(row)
    if advantage_column is not None:
        threshold = float(np.quantile(probability, 0.90))
        selected = predictions.loc[probability >= threshold]
        aggregate["mean_advantage_bps"] = float(
            predictions[advantage_column].mean()
        )
        aggregate["top_probability_decile_advantage_bps"] = float(
            selected[advantage_column].mean()
        )
        aggregate["probability_rank_ic_advantage"] = float(
            predictions["probability"].corr(
                predictions[advantage_column], method="spearman"
            )
        )
    return aggregate, pd.DataFrame(periods)


def _run_learnability(
    hourly: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_hour = hourly.loc[hourly["side_name"].eq("long")].copy()
    tasks: dict[str, tuple[pd.DataFrame, str, np.ndarray, str | None]] = {
        "july_recognition": (
            unique_hour,
            "is_july",
            _day_group(unique_hour["__ts__"]),
            None,
        )
    }
    for side in SIDES:
        residual = hourly.loc[
            hourly["side_name"].eq(side) & hourly["residual_or_base_selected"]
        ].copy()
        direct = hourly.loc[
            hourly["side_name"].eq(side)
            & hourly["direct_or_residual_selected"]
        ].copy()
        tasks[f"residual_over_base_trust_{side}"] = (
            residual,
            "residual_advantage_positive",
            _week_group(residual["__ts__"]),
            "residual_minus_base_bps",
        )
        tasks[f"direct_over_residual_trust_{side}"] = (
            direct,
            "direct_advantage_positive",
            _week_group(direct["__ts__"]),
            "direct_minus_residual_bps",
        )
    predictions: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    periods: list[pd.DataFrame] = []
    for task_index, (task, (rows, target, groups, advantage)) in enumerate(
        tasks.items()
    ):
        for arm_index, (arm, features) in enumerate(ARMS.items()):
            pred, fold, coefficient = _grouped_oof(
                rows,
                features,
                target,
                groups,
                seed=seed + task_index * 100 + arm_index,
            )
            pred["task"] = task
            pred["arm"] = arm
            fold["task"] = task
            fold["arm"] = arm
            coefficient["task"] = task
            coefficient["arm"] = arm
            aggregate, period = _prediction_metrics(
                pred,
                task=task,
                arm=arm,
                target_column=target,
                advantage_column=advantage,
            )
            predictions.append(pred)
            folds.append(fold)
            coefficients.append(coefficient)
            metrics.append(aggregate)
            periods.append(period)
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(folds, ignore_index=True),
        pd.concat(coefficients, ignore_index=True),
        pd.DataFrame(metrics),
        pd.concat(periods, ignore_index=True),
    )


def _context_shift(candidate: pd.DataFrame) -> pd.DataFrame:
    fields = list(ARMS["combined21"])
    hourly = candidate.drop_duplicates("__ts__").copy()
    reference = hourly.loc[hourly["candidate_month"].eq("2026-05")]
    rows: list[dict[str, Any]] = []
    for field in fields:
        ref_mean = float(reference[field].mean())
        ref_std = float(reference[field].std(ddof=0))
        for month, local in hourly.groupby("candidate_month", sort=True):
            mean = float(local[field].mean())
            rows.append(
                {
                    "field": field,
                    "month": str(month),
                    "hours": len(local),
                    "mean": mean,
                    "std": float(local[field].std(ddof=0)),
                    "may_reference_mean": ref_mean,
                    "may_reference_std": ref_std,
                    "smd_vs_may": (
                        (mean - ref_mean) / ref_std if ref_std > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _selected_context(
    candidate: pd.DataFrame,
    books: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = list(ARMS["combined21"])
    context = candidate.loc[:, ["candidate_id", *fields]]
    joined = books.merge(
        context, on="candidate_id", how="left", validate="many_to_one"
    )
    if joined[fields].isna().any().any():
        raise RegimeTransferError("selected books lost context")
    summary: list[dict[str, Any]] = []
    for (month, score), local in joined.groupby(
        ["candidate_month", "score_name"], sort=True
    ):
        for field in fields:
            summary.append(
                {
                    "month": str(month),
                    "score_name": score,
                    "field": field,
                    "selected_rows": len(local),
                    "selected_mean": float(local[field].mean()),
                    "selected_std": float(local[field].std(ddof=0)),
                    "selected_net_bps": float(local[NET].mean() * 1e4),
                }
            )
    pairs = (
        ("base_ev_exact_h12", "residual_exact_h12"),
        ("residual_exact_h12", "direct_q25_exact_h12"),
    )
    replacements: list[dict[str, Any]] = []
    for month, month_rows in joined.groupby("candidate_month", sort=True):
        by_score = {
            score: local.set_index("candidate_id", drop=False)
            for score, local in month_rows.groupby("score_name", sort=True)
        }
        for baseline, challenger in pairs:
            left, right = by_score[baseline], by_score[challenger]
            removed = left.loc[left.index.difference(right.index)]
            added = right.loc[right.index.difference(left.index)]
            for cohort_name, cohort in (("removed", removed), ("added", added)):
                for field in fields:
                    replacements.append(
                        {
                            "month": str(month),
                            "baseline": baseline,
                            "challenger": challenger,
                            "cohort": cohort_name,
                            "field": field,
                            "rows": len(cohort),
                            "context_mean": float(cohort[field].mean()),
                            "net_bps": float(cohort[NET].mean() * 1e4),
                        }
                    )
    return pd.DataFrame(summary), pd.DataFrame(replacements)


def run(
    *,
    residual_root: Path,
    waterfall_root: Path,
    context_root: Path,
    trajectory_root: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    candidate, books, evidence = _load_panel(
        residual_root, waterfall_root, context_root, trajectory_root
    )
    candidate["candidate_month"] = candidate["__ts__"].dt.strftime("%Y-%m")
    hourly = _build_hour_side_panel(candidate, books)
    predictions, folds, coefficients, metrics, period_metrics = _run_learnability(
        hourly, seed=seed
    )
    shifts = _context_shift(candidate)
    selected_context, replacement_context = _selected_context(candidate, books)
    stage = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    stage.mkdir(parents=True)
    try:
        tables = {
            "candidate_context_panel.parquet": candidate,
            "hour_side_panel.parquet": hourly,
            "oof_predictions.parquet": predictions,
            "fold_metrics.parquet": folds,
            "coefficients.parquet": coefficients,
            "aggregate_metrics.parquet": metrics,
            "period_metrics.parquet": period_metrics,
            "context_shift.parquet": shifts,
            "selected_book_context.parquet": selected_context,
            "replacement_context.parquet": replacement_context,
        }
        outputs: dict[str, Any] = {}
        for name, frame in tables.items():
            path = stage / name
            frame.to_parquet(path, index=False, compression="zstd")
            outputs[name] = {
                "path": str((output_dir / name).resolve()),
                "rows": len(frame),
                "sha256": sha256(path),
            }
        manifest = {
            "schema": SCHEMA,
            "status": (
                "SEALED_REUSED_PERIOD_GROUPED_OOF_CONTEXT_DIAGNOSTIC_"
                "NO_GATE_NO_PROMOTION_NO_REPLAY"
            ),
            "promotion_eligible": False,
            "portfolio_replay_authorized": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": evidence,
            "contract": {
                "candidate_cadence": "1h decisions; 1m only in frozen exact-H12 labels",
                "context": (
                    "frozen pre-2026 regime/transition and trajectory probabilities "
                    "joined at signal __ts__; no state, destination, cluster, OOD, "
                    "outcome, path or action fields"
                ),
                "trajectory_missingness": (
                    "pre-registered probability=0.5, entropy=ln(2), margin=0 "
                    "neutral fill plus explicit availability"
                ),
                "tasks": {
                    "july_recognition": (
                        "July versus May-June market-period recognition from "
                        "context only"
                    ),
                    "residual_over_base_trust": (
                        "positive side-local hourly pooled-global-book contribution "
                        "of exact-H12 residual minus exact-H12 base EV"
                    ),
                    "direct_over_residual_trust": (
                        "positive side-local hourly pooled-global-book contribution "
                        "of frozen direct q25 minus exact-H12 residual"
                    ),
                },
                "validation": (
                    "fixed C=1 balanced logistic, shuffled stratified UTC-day "
                    "groups for July recognition and seven-day groups for trust; "
                    "no HPO, feature selection or threshold search"
                ),
                "selection": (
                    "reuses already frozen one-pooled-global monthly top10 books; "
                    "side is attribution only and never a quota"
                ),
                "interpretation": (
                    "reused-period learnability/attribution only; even a strong "
                    "classifier cannot authorize a gate"
                ),
            },
            "feature_arms": {name: list(fields) for name, fields in ARMS.items()},
            "outputs": outputs,
            "runner": binding(Path(__file__)),
        }
        write_json(stage / "manifest.json", manifest)
        manifest_digest = sha256(stage / "manifest.json")
        (stage / "manifest.sha256").write_text(
            f"{manifest_digest}  manifest.json\n", encoding="utf-8"
        )
        write_json(
            stage / "seal.json",
            {
                "schema": SCHEMA,
                "manifest_sha256": manifest_digest,
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
    value.add_argument("--residual-root", type=Path, default=DEFAULT_RESIDUAL)
    value.add_argument("--waterfall-root", type=Path, default=DEFAULT_WATERFALL)
    value.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT)
    value.add_argument("--trajectory-root", type=Path, default=DEFAULT_TRAJECTORY)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--seed", type=int, default=20260730)
    return value


if __name__ == "__main__":
    print(json.dumps(safe(run(**vars(parser().parse_args()))), sort_keys=True))
