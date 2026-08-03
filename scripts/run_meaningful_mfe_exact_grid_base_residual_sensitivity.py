#!/usr/bin/env python3
"""Diagnostic exact-grid config-routed CatBoost base -> residual sensitivity.

This runner deliberately does *not* tune on any target evaluation period.  It
uses the signed h12_u1p5atr grid and the existing config base/meta routing to
compare, per side and target:

* a base-only CatBoost classifier;
* config base -> residual CatBoost regressor; and
* the same residual architecture after removing every base feature from the
  meta pool (a disjoint-meta sensitivity, not a new production architecture).

Base scores used to fit either residual head are generated only by chronological
cross-fitting inside the relevant outer training window.  The transfer and
grouped-day July splits are diagnostic only; no result from this script is
promotion eligible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_historical_to_july_meaningful_mfe_gate_challenger import (  # noqa: E402
    classification_metrics,
    sha256,
)
from scripts.run_meaningful_mfe_catboost_v2_ablation import (  # noqa: E402
    _fit_catboost,
    _predict,
)
from scripts.run_meaningful_mfe_exact_grid_reset import (  # noqa: E402
    IDENTITY,
    TRANSFER_SPECS,
    _base_masks,
    economic_metrics,
    july_grouped_day_folds,
    load_panel,
)


SCHEMA = "meaningful_mfe_exact_grid_base_residual_sensitivity_v1"
SIDES = ("long", "short")
TASKS = ("any_touch", "clean_first")
ARMS = (
    "base_only",
    "monolithic_union",
    "configured_base_residual",
    "disjoint_meta_sensitivity",
)
TRANSFER_NAMES = ("may_to_june", "june_to_july", "july_to_june_matched")

# These are intentionally conservative, fixed geometries.  They are not HPO
# winners and no target-period result can change them.
BASE_PARAMS: Mapping[str, Any] = {
    "iterations": 320,
    "learning_rate": 0.035,
    "depth": 5,
    "l2_leaf_reg": 10.0,
}
RESIDUAL_PARAMS: Mapping[str, Any] = {
    "iterations": 260,
    "learning_rate": 0.030,
    "depth": 5,
    "l2_leaf_reg": 12.0,
}
RESIDUAL_SHRINKAGE: Mapping[str, float] = {
    # Frozen from the preceding config-routed architecture study.  These are
    # not retuned on any exact-grid transfer or July diagnostic period.
    "long": 0.50,
    "short": 0.25,
}


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


def config_routed_feature_pools(
    matrix_columns: Sequence[str],
    *,
    configured_base_by_side: Mapping[str, Sequence[str]] | None = None,
    configured_meta: Sequence[str] | None = None,
) -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    """Resolve existing config routing against the signed exact matrix.

    Imports the legacy routing helper only when this runner is actually used.
    This keeps pure split/contract tests independent of the heavyweight config
    imports while preserving a single source of truth for the runtime routing.
    """

    if configured_base_by_side is None or configured_meta is None:
        from scripts.run_meaningful_mfe_base_residual_catboost_ablation import (
            _configured_features_by_side,
        )

        configured_base_by_side, configured_meta = _configured_features_by_side()
    available = set(map(str, matrix_columns))

    def match(values: Sequence[str]) -> list[str]:
        # The exact universe strips ``capture_candidate__`` after validating
        # its signed full-period universe; accept either spelling defensively.
        return list(
            dict.fromkeys(
                value
                for item in values
                for value in (str(item).removeprefix("capture_candidate__"),)
                if value in available
            )
        )

    base = {side: match(configured_base_by_side.get(side, ())) for side in SIDES}
    meta = match(configured_meta)
    if min(map(len, base.values())) < 8 or len(meta) < 8:
        raise ValueError(
            "exact config-routed feature support is insufficient: "
            f"base={{side: len(values) for side, values in base.items()}}, "
            f"meta={len(meta)}"
        )
    disjoint = {side: [column for column in meta if column not in set(base[side])] for side in SIDES}
    if min(map(len, disjoint.values())) < 8:
        raise ValueError(
            "disjoint-meta sensitivity has insufficient support: "
            f"{{side: len(values) for side, values in disjoint.items()}}"
        )
    return base, meta, {
        "requested_base_by_side": {
            side: list(map(str, configured_base_by_side.get(side, ())))
            for side in SIDES
        },
        "requested_meta": list(map(str, configured_meta)),
        "available_base_by_side": base,
        "available_meta": meta,
        "disjoint_meta_by_side": disjoint,
        "base_meta_overlap_by_side": {
            side: sorted(set(base[side]).intersection(meta)) for side in SIDES
        },
    }


def _crossfit_plan(
    frame: pd.DataFrame,
    *,
    folds: int = 4,
    min_train_rows: int = 2_000,
    min_validation_rows: int = 500,
) -> list[dict[str, Any]]:
    """Return strict chronological base-OOF folds for one outer train pool."""

    if folds < 2:
        raise ValueError("cross-fit requires at least two folds")
    signal = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    resolved = pd.to_datetime(frame["label_resolution_utc"], utc=True, errors="raise")
    decision = pd.to_datetime(frame["execution_decision_utc"], utc=True, errors="raise")
    unique = np.asarray(sorted(signal.unique()))
    if len(unique) < folds:
        return []
    boundaries = np.unique(np.linspace(0, len(unique), folds + 1, dtype=int))
    plan: list[dict[str, Any]] = []
    for fold in range(1, len(boundaries) - 1):
        start = pd.Timestamp(unique[boundaries[fold]])
        stop_index = int(boundaries[fold + 1])
        stop = (
            pd.Timestamp(unique[stop_index])
            if stop_index < len(unique)
            else pd.Timestamp(unique[-1]) + pd.Timedelta(nanoseconds=1)
        )
        train = np.flatnonzero(
            signal.lt(start).to_numpy()
            & resolved.lt(start).to_numpy()
            & decision.lt(start - pd.Timedelta(hours=12)).to_numpy()
        )
        validation = np.flatnonzero(
            signal.ge(start).to_numpy() & signal.lt(stop).to_numpy()
        )
        if len(train) < min_train_rows or len(validation) < min_validation_rows:
            continue
        if set(train).intersection(validation):
            raise AssertionError("cross-fit train/validation overlap")
        if not bool((resolved.iloc[train] < start).all()):
            raise AssertionError("cross-fit uses unresolved base labels")
        if not bool((decision.iloc[train] < start - pd.Timedelta(hours=12)).all()):
            raise AssertionError("cross-fit violates 12h decision purge")
        plan.append(
            {
                "fold": int(fold - 1),
                "train": train,
                "validation": validation,
                "validation_start": start,
                "validation_end": stop,
                "training_label_resolution_max": resolved.iloc[train].max(),
                "training_decision_max": decision.iloc[train].max(),
            }
        )
    return plan


def _crossfit_base(
    X: pd.DataFrame,
    target: np.ndarray,
    frame: pd.DataFrame,
    *,
    seed: int,
    min_train_rows: int,
    min_validation_rows: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prediction = np.full(len(X), np.nan, dtype=np.float64)
    reports: list[dict[str, Any]] = []
    for item in _crossfit_plan(
        frame,
        min_train_rows=min_train_rows,
        min_validation_rows=min_validation_rows,
    ):
        train = np.asarray(item["train"], dtype=np.int64)
        validation = np.asarray(item["validation"], dtype=np.int64)
        model = _fit_catboost(
            "binary", X.iloc[train], target[train], BASE_PARAMS, seed=seed + int(item["fold"])
        )
        prediction[validation] = _predict(model, "binary", X.iloc[validation])
        reports.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"train", "validation"}
            }
            | {"train_rows": int(len(train)), "validation_rows": int(len(validation))}
        )
    return prediction, reports


def _fit_arm(
    base_X: pd.DataFrame,
    meta_X: pd.DataFrame | None,
    target: np.ndarray,
    frame: pd.DataFrame,
    train: np.ndarray,
    evaluation: np.ndarray,
    *,
    arm: str,
    shrinkage: float,
    seed: int,
    min_crossfit_train_rows: int,
    min_crossfit_validation_rows: int,
    min_residual_train_rows: int,
    base_bundle: tuple[
        np.ndarray, np.ndarray, list[dict[str, Any]]
    ]
    | None = None,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    tuple[np.ndarray, np.ndarray, list[dict[str, Any]]],
]:
    """Fit one arm; residual fitting never sees in-sample base scores."""

    if base_bundle is None:
        base_model = _fit_catboost(
            "binary", base_X.iloc[train], target[train], BASE_PARAMS, seed=seed
        )
        base_prediction = _predict(
            base_model, "binary", base_X.iloc[evaluation]
        )
        outer_frame = frame.iloc[train].reset_index(drop=True)
        outer_target = np.asarray(target[train], dtype=np.float64)
        base_oof, crossfit = _crossfit_base(
            base_X.iloc[train].reset_index(drop=True),
            outer_target,
            outer_frame,
            seed=seed + 10_000,
            min_train_rows=min_crossfit_train_rows,
            min_validation_rows=min_crossfit_validation_rows,
        )
        base_bundle = (base_prediction, base_oof, crossfit)
    else:
        base_prediction, base_oof, crossfit = base_bundle
    residual_rows = np.flatnonzero(np.isfinite(base_oof))
    if arm == "base_only":
        return (
            base_prediction,
            {
                "arm": arm,
                "base_pool_rows": int(len(train)),
                "base_feature_count": int(base_X.shape[1]),
                "meta_feature_count": 0,
                "base_crossfit_rows": int(len(residual_rows)),
                "base_crossfit_missing_rows": int(
                    len(train) - len(residual_rows)
                ),
                "crossfit": crossfit,
                "residual_train_rows": 0,
                "residual_shrinkage": 0.0,
            },
            base_bundle,
        )
    if len(residual_rows) < min_residual_train_rows:
        raise ValueError(
            f"{arm} has insufficient strictly cross-fitted residual rows: {len(residual_rows)}"
        )
    architecture: dict[str, Any] = {
        "arm": arm,
        "base_pool_rows": int(len(train)),
        "base_feature_count": int(base_X.shape[1]),
        "meta_feature_count": int(0 if meta_X is None else meta_X.shape[1]),
        "base_crossfit_rows": int(len(residual_rows)),
        "base_crossfit_missing_rows": int(len(train) - len(residual_rows)),
        "crossfit": crossfit,
        "residual_train_rows": int(0),
    }
    if meta_X is None:
        raise ValueError(f"{arm} requires a meta matrix")
    residual_train = meta_X.iloc[train].iloc[residual_rows].copy()
    residual_train["__base_oof_probability__"] = base_oof[residual_rows]
    residual_target = target[train][residual_rows] - base_oof[residual_rows]
    residual_model = _fit_catboost(
        "quality", residual_train, residual_target, RESIDUAL_PARAMS, seed=seed + 20_000
    )
    residual_evaluation = meta_X.iloc[evaluation].copy()
    residual_evaluation["__base_oof_probability__"] = base_prediction
    residual_prediction = np.asarray(residual_model.predict(residual_evaluation), dtype=np.float64)
    architecture.update(
        {
            "residual_train_rows": int(len(residual_rows)),
            "residual_target_mean": float(np.mean(residual_target)),
            "residual_target_std": float(np.std(residual_target)),
            "residual_prediction_mean": float(np.mean(residual_prediction)),
            "residual_shrinkage": float(shrinkage),
        }
    )
    return (
        np.clip(
            base_prediction + float(shrinkage) * residual_prediction,
            1e-6,
            1.0 - 1e-6,
        ),
        architecture,
        base_bundle,
    )


def _head_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(target, prediction)
    count = max(1, int(math.ceil(len(target) * 0.10)))
    order = np.argsort(-np.asarray(prediction), kind="stable")[:count]
    positives = int(np.asarray(target, dtype=int).sum())
    return {
        **metrics,
        "top10_rows": count,
        "top10_precision": float(np.asarray(target)[order].mean()),
        "top10_recall": (
            float(np.asarray(target)[order].sum() / positives)
            if positives
            else float("nan")
        ),
    }


def _score_split(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    base_by_side: Mapping[str, Sequence[str]],
    meta: Sequence[str],
    *,
    train: np.ndarray,
    evaluation: np.ndarray,
    name: str,
    seed: int,
    validation_days: Sequence[str] = (),
    min_crossfit_train_rows: int,
    min_crossfit_validation_rows: int,
    min_residual_train_rows: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    keep = [
        *IDENTITY,
        "execution_net_ev_12h",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "any_touch",
        "clean_first",
        "timeout",
    ]
    scored = panel.iloc[evaluation][keep].copy().reset_index(drop=True)
    metrics_rows: list[dict[str, Any]] = []
    architecture_rows: list[dict[str, Any]] = []
    for side_index, side in enumerate(SIDES):
        side_train = train[panel.iloc[train]["side_name"].astype(str).eq(side).to_numpy()]
        side_evaluation = evaluation[panel.iloc[evaluation]["side_name"].astype(str).eq(side).to_numpy()]
        output_positions = np.flatnonzero(scored["side_name"].astype(str).eq(side).to_numpy())
        if not len(side_train) or not len(side_evaluation):
            raise ValueError(f"{name}/{side} has empty side-local support")
        base_features = list(base_by_side[side])
        configured_meta = list(meta)
        disjoint_meta = [value for value in meta if value not in set(base_features)]
        union_features = list(dict.fromkeys([*base_features, *configured_meta]))
        for task_index, task in enumerate(TASKS):
            target = panel[task].to_numpy(float)
            base_bundle: tuple[
                np.ndarray, np.ndarray, list[dict[str, Any]]
            ] | None = None
            for arm_index, arm in enumerate(ARMS):
                if arm == "monolithic_union":
                    used_base_features = union_features
                    use_meta: list[str] | None = None
                    model = _fit_catboost(
                        "binary",
                        matrix.iloc[side_train][union_features],
                        target[side_train],
                        BASE_PARAMS,
                        seed=(
                            seed
                            + 100_000 * side_index
                            + 10_000 * task_index
                            + arm_index
                        ),
                    )
                    prediction = _predict(
                        model,
                        "binary",
                        matrix.iloc[side_evaluation][union_features],
                    )
                    architecture = {
                        "arm": arm,
                        "base_pool_rows": int(len(side_train)),
                        "base_feature_count": int(len(union_features)),
                        "meta_feature_count": 0,
                        "base_crossfit_rows": 0,
                        "base_crossfit_missing_rows": 0,
                        "crossfit": [],
                        "residual_train_rows": 0,
                        "residual_shrinkage": 0.0,
                    }
                else:
                    used_base_features = base_features
                    if arm == "base_only":
                        use_meta = None
                    elif arm == "configured_base_residual":
                        use_meta = configured_meta
                    else:
                        use_meta = disjoint_meta
                    prediction, architecture, base_bundle = _fit_arm(
                        matrix[base_features],
                        None if use_meta is None else matrix[use_meta],
                        target,
                        panel,
                        side_train,
                        side_evaluation,
                        arm=arm,
                        shrinkage=RESIDUAL_SHRINKAGE[side],
                        seed=(
                            seed
                            + 100_000 * side_index
                            + 10_000 * task_index
                            + arm_index
                        ),
                        min_crossfit_train_rows=min_crossfit_train_rows,
                        min_crossfit_validation_rows=min_crossfit_validation_rows,
                        min_residual_train_rows=min_residual_train_rows,
                        base_bundle=base_bundle,
                    )
                column = f"score_{task}_{arm}"
                scored.loc[output_positions, column] = prediction
                metrics_rows.append(
                    {
                        "evaluation": name,
                        "side": side,
                        "task": task,
                        "arm": arm,
                        "train_rows": int(len(side_train)),
                        "evaluation_rows": int(len(side_evaluation)),
                        "validation_days": "|".join(validation_days),
                        **_head_metrics(target[side_evaluation].astype(int), prediction),
                    }
                )
                architecture_rows.append(
                    {
                        "evaluation": name,
                        "side": side,
                        "task": task,
                        "arm": arm,
                        "base_features": json.dumps(used_base_features),
                        "meta_features": json.dumps(use_meta or []),
                        "base_meta_overlap_features": json.dumps(
                            sorted(set(base_features).intersection(use_meta or []))
                        ),
                        "base_meta_overlap_feature_count": int(
                            len(set(base_features).intersection(use_meta or []))
                        ),
                        "validation_days": "|".join(validation_days),
                        **architecture,
                    }
                )
    scored["evaluation"] = name
    return scored, metrics_rows, architecture_rows


def _economic_rows(scored: pd.DataFrame, evaluation: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for task in TASKS:
        for arm in ARMS:
            score = f"score_{task}_{arm}"
            for side in ("pooled", *SIDES):
                results.append(economic_metrics(scored, score, scope=evaluation, side=side))
                results[-1].update({"task": task, "arm": arm})
    return results


def _aggregate_july_metrics(scored: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in SIDES:
        local = scored.loc[scored["side_name"].astype(str).eq(side)]
        for task in TASKS:
            target = local[task].to_numpy(int)
            for arm in ARMS:
                prediction = local[f"score_{task}_{arm}"].to_numpy(float)
                rows.append(
                    {
                        "evaluation": "july_grouped_oof",
                        "side": side,
                        "task": task,
                        "arm": arm,
                        "train_rows": np.nan,
                        "evaluation_rows": len(local),
                        "validation_days": "five_contiguous_two_day_blocks",
                        **_head_metrics(target, prediction),
                    }
                )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, matrix, _, lineage = load_panel(
        args.features, args.feature_manifest, args.grid, args.grid_manifest
    )
    base_by_side, meta, feature_contract = config_routed_feature_pools(matrix.columns)
    all_scored: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    all_architecture: list[dict[str, Any]] = []
    all_economics: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    specs = [spec for spec in TRANSFER_SPECS if spec.name in TRANSFER_NAMES]
    if tuple(spec.name for spec in specs) != TRANSFER_NAMES:
        raise AssertionError("required transfer specifications changed")
    for index, spec in enumerate(specs):
        train, evaluation = _base_masks(panel, spec)
        scored, metrics, architecture = _score_split(
            panel, matrix, base_by_side, meta, train=train, evaluation=evaluation,
            name=spec.name, seed=args.seed + index * 1_000_000,
            min_crossfit_train_rows=args.min_crossfit_train_rows,
            min_crossfit_validation_rows=args.min_crossfit_validation_rows,
            min_residual_train_rows=args.min_residual_train_rows,
        )
        all_scored.append(scored)
        all_metrics.extend(metrics)
        all_architecture.extend(architecture)
        all_economics.extend(_economic_rows(scored, spec.name))
        split_rows.append({
            "name": spec.name, "train_rows": len(train), "evaluation_rows": len(evaluation),
            "promotion_eligible": False, "note": spec.note,
        })
    july_parts: list[pd.DataFrame] = []
    for fold_index, (name, train, evaluation, days) in enumerate(july_grouped_day_folds(panel)):
        scored, metrics, architecture = _score_split(
            panel, matrix, base_by_side, meta, train=train, evaluation=evaluation,
            name=name, seed=args.seed + 10_000_000 + fold_index * 1_000_000,
            validation_days=days, min_crossfit_train_rows=args.min_crossfit_train_rows,
            min_crossfit_validation_rows=args.min_crossfit_validation_rows,
            min_residual_train_rows=args.min_residual_train_rows,
        )
        july_parts.append(scored)
        all_metrics.extend(metrics)
        all_architecture.extend(architecture)
        split_rows.append({
            "name": name, "train_rows": len(train), "evaluation_rows": len(evaluation),
            "promotion_eligible": False,
            "note": "GROUPED_DAY_JULY_OOF_LEARNABILITY_NONPROMOTABLE",
            "validation_days": days,
        })
    july_oof = pd.concat(july_parts, ignore_index=True)
    if july_oof.duplicated(list(IDENTITY)).any():
        raise ValueError("July grouped-day base-residual OOF has duplicate identities")
    july_oof["evaluation"] = "july_grouped_oof"
    all_scored.append(july_oof)
    all_metrics.extend(_aggregate_july_metrics(july_oof))
    all_economics.extend(_economic_rows(july_oof, "july_grouped_oof"))
    outputs: dict[str, Any] = {}
    for name, value in (
        ("predictions", pd.concat(all_scored, ignore_index=True)),
        ("head_metrics", pd.DataFrame(all_metrics)),
        ("architecture", pd.DataFrame(all_architecture)),
        ("economics", pd.DataFrame(all_economics)),
    ):
        path = args.output_dir / f"{name}.parquet"
        value.to_parquet(path, index=False)
        outputs[name] = {"path": path, "rows": len(value), "sha256": sha256(path)}
    report = {
        "schema": SCHEMA,
        "status": "COMPLETED_DIAGNOSTIC_EXACT_GRID_NO_PROMOTION",
        "promotion_eligible": False,
        "lineage": lineage,
        "runner": {
            "path": Path(__file__).resolve(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "feature_contract": feature_contract,
        "geometry": {"base": dict(BASE_PARAMS), "residual": dict(RESIDUAL_PARAMS)},
        "residual_shrinkage_by_side": dict(RESIDUAL_SHRINKAGE),
        "minimum_support": {
            "crossfit_train_rows": args.min_crossfit_train_rows,
            "crossfit_validation_rows": args.min_crossfit_validation_rows,
            "residual_train_rows": args.min_residual_train_rows,
        },
        "contracts": {
            "targets": "side-local exact-grid any_touch and clean_first",
            "residual": "base OOF probabilities only; no residual row has an in-sample base score",
            "transfer": "May->June, June->July and July->June matched are diagnostic; reverse time is permanently nonpromotable",
            "july_oof": "five grouped UTC-day folds with +/-12h exclusion; diagnostic only",
            "hpo": (
                "no target-period geometry, feature, calibration, or shrinkage "
                "HPO; shrinkage is frozen from the preceding config-routed study"
            ),
            "global_selection": "economics uses a pooled global deterministic top 10% per score; side economics are diagnostic",
        },
        "splits": split_rows,
        "outputs": outputs,
    }
    report_path = args.output_dir / "report.json"
    _write_json(report_path, report)
    _write_json(args.output_dir / "manifest.json", {
        "schema": SCHEMA, "status": report["status"], "promotion_eligible": False,
        "report": {"path": report_path, "sha256": sha256(report_path)}, "outputs": outputs,
    })
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--features", type=Path, default=Path(
        "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet"))
    value.add_argument("--feature-manifest", type=Path, default=Path(
        "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/manifest.json"))
    value.add_argument("--grid", type=Path, default=Path(
        "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/meaningful_mfe_label_grid.parquet"))
    value.add_argument("--grid-manifest", type=Path, default=Path(
        "data_perp/artifacts/meaningful_mfe_exact_policy_label_grid_20260727_v1/manifest.json"))
    value.add_argument("--output-dir", type=Path, default=Path(
        "data_perp/artifacts/meaningful_mfe_exact_grid_base_residual_sensitivity_20260730_v2"))
    value.add_argument("--seed", type=int, default=20260730)
    value.add_argument("--min-crossfit-train-rows", type=int, default=2_000)
    value.add_argument("--min-crossfit-validation-rows", type=int, default=500)
    value.add_argument("--min-residual-train-rows", type=int, default=1_500)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
