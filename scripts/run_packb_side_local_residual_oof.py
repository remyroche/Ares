#!/usr/bin/env python3
"""Train strict side-local residual-alpha models on canonical Pack-B top-40 OOF.

April is development-only because the canonical 31/8 base OOF stream begins
there.  May, June, and July are expanding, prior-resolved OOF folds.  Every
selector, HPO study, baseline EV calibrator, residual model, and correction
strength is fitted independently by side.  Final refits are stored separately
and are never used in OOF metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
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

from extreme_price_movements.packb_side_local_fs_hpo_stage import (  # noqa: E402
    _bounded_beginning_middle_end_sample,
)
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.run_packb_pre_march_side_ae import (  # noqa: E402
    DEFAULT_DECISIONS,
    DEFAULT_FEATURE_INVENTORY,
    DEFAULT_FEATURE_STORE,
    DEFAULT_POPULATION_ROOT,
    _source_contracts,
)
from scripts.run_packb_pre_march_side_fs_hpo import (  # noqa: E402
    ECONOMIC_COLUMN,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    ExactLabelLoader,
    SideRepresentationFeatureLoader,
    _active_ae_gmm_columns,
    _canonical_label_files,
    _economic_objective,
    _git_revision,
    _load_loader_contract,
    _load_side_ae_state,
    make_fs_hpo_raw_feature_loader,
)

SCHEMA = "packb_side_local_residual_oof_v1"
SIDES = ("long", "short")
ANCHORS = (
    "base_prediction",
    "base_rank_pct_timestamp_side",
    "base_rank_timestamp_side",
    "base_group_rows",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)
DEFAULT_POPULATION = (
    ROOT / "data_perp/artifacts/packb_side_local_top40_20260724_v1_31_8/"
    "base_candidate_population.parquet"
)
DEFAULT_POPULATION_MANIFEST = DEFAULT_POPULATION.with_name("manifest.json")
DEFAULT_OUTER_ROOT = (
    ROOT / "data_perp/artifacts/packb_side_local_outer_oof_20260724_v1_31_8"
)
DEFAULT_AE_ROOT = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1"
DEFAULT_INNER_POPULATION = DEFAULT_POPULATION_ROOT
DEFAULT_LABELS = (
    ROOT / "data_perp/artifacts/"
    "20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/packb_side_local_residual_oof_20260724_v1_31_8"
)
FOLDS = (
    ("residual_1_20260501", "2026-05-01T00:00:00Z", "2026-06-01T00:00:00Z"),
    ("residual_2_20260601", "2026-06-01T00:00:00Z", "2026-07-01T00:00:00Z"),
    ("residual_3_20260701", "2026-07-01T00:00:00Z", "2026-07-11T00:00:00Z"),
)


class ResidualOOFError(RuntimeError):
    """Raised when the strict residual-alpha contract cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(
            json.dumps(_jsonable(dict(value)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_population(
    population_path: Path,
    manifest_path: Path,
    outer_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "base_candidate_population_v2"
        or manifest.get("selected_column") != "selected_top40"
        or manifest.get("fold_provenance", {})
        .get("strict_execution_ev_handoff", {})
        .get("status")
        != "ready"
    ):
        raise ResidualOOFError("strict canonical top-40 manifest is required")
    if manifest.get("output", {}).get("sha256") != _sha256(population_path):
        raise ResidualOOFError("canonical top-40 parquet hash changed")
    source = outer_root / "oof_predictions.parquet"
    if (
        not source.is_file()
        or manifest.get("source", {}).get("sha256") != _sha256(source)
        or manifest.get("source_rows") != 744251
    ):
        raise ResidualOOFError("canonical base OOF source binding changed")
    frame = pd.read_parquet(population_path)
    required = {
        "candidate_id",
        "side_name",
        "__ts__",
        "__symbol__",
        "prediction",
        TARGET_COLUMN,
        WEIGHT_COLUMN,
        ECONOMIC_COLUMN,
        "selected_top40",
        "prediction_source",
        "base_candidate_rank_timestamp_side",
        "base_candidate_rank_pct_timestamp_side",
        "base_candidate_group_rows",
    }
    missing = sorted(required.difference(frame))
    if missing:
        raise ResidualOOFError(f"canonical top-40 columns missing: {missing}")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["__decision_ts__"] = frame["__ts__"] + pd.Timedelta(hours=1)
    frame["__label_resolution_ts__"] = frame["__decision_ts__"] + pd.Timedelta(hours=24)
    if (
        len(frame) != int(manifest.get("selected_rows", -1))
        or frame["candidate_id"].astype(str).duplicated().any()
        or not frame["selected_top40"].astype(bool).all()
        or set(frame["side_name"].astype(str)) != set(SIDES)
        or set(frame["prediction_source"].astype(str)) != {"outer_oof_fold_model"}
    ):
        raise ResidualOOFError("canonical top-40 identity or OOF source changed")
    return frame, manifest


def _verify_labels(
    frame: pd.DataFrame,
    *,
    labels: ExactLabelLoader,
) -> None:
    observed = labels.load(
        frame.loc[
            :,
            [
                "candidate_id",
                "side_name",
                "__ts__",
                "__decision_ts__",
                "__label_resolution_ts__",
                "__symbol__",
            ],
        ]
    )
    for column in (TARGET_COLUMN, WEIGHT_COLUMN, ECONOMIC_COLUMN):
        left = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
        right = pd.to_numeric(observed[column], errors="coerce").to_numpy(np.float64)
        if not np.allclose(left, right, rtol=0.0, atol=1e-7, equal_nan=True):
            raise ResidualOOFError(f"canonical top-40 {column} differs from labels")


def _add_anchors(frame: pd.DataFrame, representation: pd.DataFrame) -> pd.DataFrame:
    output = representation.reset_index(drop=True).copy()
    timestamp = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    output["base_prediction"] = pd.to_numeric(
        frame["prediction"], errors="coerce"
    ).to_numpy(np.float32)
    output["base_rank_pct_timestamp_side"] = pd.to_numeric(
        frame["base_candidate_rank_pct_timestamp_side"], errors="coerce"
    ).to_numpy(np.float32)
    output["base_rank_timestamp_side"] = pd.to_numeric(
        frame["base_candidate_rank_timestamp_side"], errors="coerce"
    ).to_numpy(np.float32)
    output["base_group_rows"] = pd.to_numeric(
        frame["base_candidate_group_rows"], errors="coerce"
    ).to_numpy(np.float32)
    output["hour_sin"] = np.sin(2.0 * np.pi * timestamp.dt.hour / 24.0).astype(
        np.float32
    )
    output["hour_cos"] = np.cos(2.0 * np.pi * timestamp.dt.hour / 24.0).astype(
        np.float32
    )
    output["dow_sin"] = np.sin(2.0 * np.pi * timestamp.dt.dayofweek / 7.0).astype(
        np.float32
    )
    output["dow_cos"] = np.cos(2.0 * np.pi * timestamp.dt.dayofweek / 7.0).astype(
        np.float32
    )
    return output


def _active_features(
    matrix: pd.DataFrame,
    candidates: Sequence[str],
    *,
    minimum_finite_fraction: float,
) -> list[str]:
    selected: list[str] = []
    for feature in candidates:
        values = pd.to_numeric(matrix[feature], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        if finite.mean() < minimum_finite_fraction:
            continue
        if np.unique(values[finite]).size < 8:
            continue
        selected.append(str(feature))
    return selected


def _fit_ev_map(
    score: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> IsotonicRegression:
    finite = (
        np.isfinite(score) & np.isfinite(target) & np.isfinite(weight) & (weight > 0)
    )
    if finite.sum() < 1000:
        raise ResidualOOFError("side-local EV map has insufficient resolved support")
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(score[finite], target[finite], sample_weight=weight[finite])
    return model


def _predict_ev_map(model: IsotonicRegression, score: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(score), dtype=np.float64)


def _base_params(seed: int) -> dict[str, Any]:
    return {
        "objective": "regression_l2",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 5,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 8.0,
        "verbosity": -1,
        "num_threads": max(1, min(6, int(os.cpu_count() or 1))),
        "seed": int(seed),
    }


def _fit_residual_model(
    matrix: pd.DataFrame,
    residual: np.ndarray,
    weight: np.ndarray,
    features: Sequence[str],
    params: Mapping[str, Any],
    *,
    rounds: int,
) -> lgb.Booster:
    dataset = lgb.Dataset(
        matrix.loc[:, list(features)].to_numpy(np.float32, copy=False),
        label=np.asarray(residual, dtype=np.float32),
        weight=np.asarray(weight, dtype=np.float32),
        feature_name=list(features),
        free_raw_data=True,
    )
    return lgb.train(dict(params), dataset, num_boost_round=int(rounds))


def _development_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    validation_start = pd.Timestamp("2026-04-22T00:00:00Z")
    train = (
        frame["__ts__"].lt(validation_start)
        & frame["__label_resolution_ts__"].lt(validation_start)
    ).to_numpy()
    validation = frame["__ts__"].ge(validation_start).to_numpy()
    if train.sum() < 5000 or validation.sum() < 2000:
        raise ResidualOOFError("April development split has insufficient support")
    return train, validation


def _select_features(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    candidates: Sequence[str],
    *,
    seed: int,
    max_features: int,
) -> tuple[list[str], pd.DataFrame]:
    train_mask, validation_mask = _development_split(frame)
    score = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(np.float64)
    target = pd.to_numeric(frame[ECONOMIC_COLUMN], errors="coerce").to_numpy(np.float64)
    weight = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce").to_numpy(np.float64)
    ev_map = _fit_ev_map(score[train_mask], target[train_mask], weight[train_mask])
    residual = target[train_mask] - _predict_ev_map(ev_map, score[train_mask])
    model = _fit_residual_model(
        matrix.loc[train_mask],
        residual,
        weight[train_mask],
        candidates,
        _base_params(seed),
        rounds=220,
    )
    gain = np.asarray(model.feature_importance("gain"), dtype=np.float64)
    names = np.asarray(model.feature_name(), dtype=object)
    order = np.argsort(-gain, kind="stable")
    positive = [str(names[index]) for index in order if gain[index] > 0]
    protected = [feature for feature in ANCHORS if feature in candidates]
    context = [feature for feature in positive if feature not in protected]
    if gain.sum() > 0 and context:
        gain_by_name = dict(zip(names.astype(str), gain, strict=True))
        cumulative = 0.0
        automatic: list[str] = []
        for feature in context:
            automatic.append(feature)
            cumulative += float(gain_by_name[feature])
            if cumulative / float(gain.sum()) >= 0.95 and len(automatic) >= 8:
                break
        context = automatic
    selected = list(dict.fromkeys([*protected, *context[:max_features]]))
    if len(selected) < 4:
        raise ResidualOOFError("automatic residual feature selection collapsed")
    report = pd.DataFrame(
        {
            "feature": names.astype(str),
            "gain": gain,
            "gain_share": gain / max(float(gain.sum()), 1e-12),
            "selected": [str(name) in selected for name in names],
            "development_validation_rows": int(validation_mask.sum()),
        }
    ).sort_values(["selected", "gain"], ascending=[False, False])
    return selected, report.reset_index(drop=True)


def _trial_params(rng: np.random.Generator, seed: int) -> tuple[dict[str, Any], int]:
    max_depth = int(rng.integers(3, 8))
    params = _base_params(seed)
    params.update(
        {
            "learning_rate": float(np.exp(rng.uniform(np.log(0.008), np.log(0.08)))),
            "num_leaves": int(rng.integers(6, min(2**max_depth, 48) + 1)),
            "max_depth": max_depth,
            "min_data_in_leaf": int(rng.integers(150, 900)),
            "feature_fraction": float(rng.uniform(0.55, 1.0)),
            "bagging_fraction": float(rng.uniform(0.6, 1.0)),
            "lambda_l1": float(np.exp(rng.uniform(np.log(0.02), np.log(8.0)))),
            "lambda_l2": float(np.exp(rng.uniform(np.log(0.2), np.log(20.0)))),
        }
    )
    return params, int(rng.integers(120, 700))


def _hpo(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    features: Sequence[str],
    *,
    trials: int,
    patience: int,
    seed: int,
    guard: TrainingResourceGuard | None = None,
) -> tuple[dict[str, Any], int, float, pd.DataFrame]:
    train_mask, validation_mask = _development_split(frame)
    score = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(np.float64)
    target = pd.to_numeric(frame[ECONOMIC_COLUMN], errors="coerce").to_numpy(np.float64)
    weight = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce").to_numpy(np.float64)
    ev_map = _fit_ev_map(score[train_mask], target[train_mask], weight[train_mask])
    train_ev = _predict_ev_map(ev_map, score[train_mask])
    validation_ev = _predict_ev_map(ev_map, score[validation_mask])
    residual = target[train_mask] - train_ev
    rng = np.random.default_rng(seed)
    alpha_grid = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0, 1.25), dtype=float)
    records: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], int, float] | None = None
    without_improvement = 0
    for trial in range(int(trials)):
        params, rounds = _trial_params(rng, seed + trial)
        model = _fit_residual_model(
            matrix.loc[train_mask],
            residual,
            weight[train_mask],
            features,
            params,
            rounds=rounds,
        )
        delta = np.asarray(
            model.predict(matrix.loc[validation_mask, list(features)]), dtype=np.float64
        )
        alpha_scores = [
            _economic_objective(
                validation_ev + alpha * delta,
                target[validation_mask],
                weight[validation_mask],
                target[validation_mask],
                timestamps=frame.loc[validation_mask, "__ts__"],
                symbols=frame.loc[validation_mask, "__symbol__"],
            )["objective"]
            for alpha in alpha_grid
        ]
        alpha_index = int(np.nanargmax(alpha_scores))
        objective = float(alpha_scores[alpha_index])
        alpha = float(alpha_grid[alpha_index])
        records.append(
            {
                "trial": trial,
                "objective": objective,
                "alpha": alpha,
                "rounds": rounds,
                "params": json.dumps(_jsonable(params), sort_keys=True),
            }
        )
        if best is None or objective > best[0]:
            best = (objective, params, rounds, alpha)
            without_improvement = 0
        else:
            without_improvement += 1
        del model, delta
        gc.collect()
        if guard is not None:
            guard.checkpoint(f"packb_residual:hpo:trial_{trial:03d}")
        if without_improvement >= int(patience):
            break
    if best is None:
        raise ResidualOOFError("residual HPO returned no valid trial")
    return best[1], best[2], best[3], pd.DataFrame(records)


def _metrics(
    prediction: np.ndarray,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    target = pd.to_numeric(frame[ECONOMIC_COLUMN], errors="coerce").to_numpy(np.float64)
    weight = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce").to_numpy(np.float64)
    return _economic_objective(
        prediction,
        target,
        weight,
        target,
        timestamps=frame["__ts__"],
        symbols=frame["__symbol__"],
    )


def _promotion_gate(
    *,
    base: Mapping[str, Any],
    residual: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Require aggregate uplift without material quality or fold instability."""

    checks = {
        "aggregate_objective_higher": (
            float(residual["objective"]) > float(base["objective"])
        ),
        "rank_ic_not_materially_lower": (
            float(residual["weighted_rank_ic"])
            >= float(base["weighted_rank_ic"]) - 0.01
        ),
        "top10_lift_not_materially_lower": (
            float(residual["top10_net_return_lift"])
            >= float(base["top10_net_return_lift"]) - 0.0005
        ),
        "relative_rmse_gain_not_materially_lower": (
            float(residual["relative_rmse_gain"])
            >= float(base["relative_rmse_gain"]) - 0.01
        ),
        "no_fold_objective_collapse": all(
            float(fold["residual_metrics"]["objective"])
            >= float(fold["base_metrics"]["objective"]) - 0.03
            for fold in folds
        ),
        "majority_of_folds_improve": (
            sum(
                float(fold["residual_metrics"]["objective"])
                > float(fold["base_metrics"]["objective"])
                for fold in folds
            )
            >= math.ceil(len(folds) / 2)
        ),
    }
    return all(checks.values()), checks


def _load_representation(
    loader: SideRepresentationFeatureLoader,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(index=pd.RangeIndex(len(frame)))
    return loader(frame, features)


def _bounded_position_indices(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    name: str,
) -> np.ndarray:
    sampled = _bounded_beginning_middle_end_sample(
        frame,
        max_rows=int(max_rows),
        name=name,
    )
    position_by_id = pd.Series(
        np.arange(len(frame), dtype=int),
        index=frame["candidate_id"].astype(str),
    )
    positions = position_by_id.reindex(sampled["candidate_id"].astype(str)).to_numpy()
    if pd.isna(positions).any():
        raise ResidualOOFError(f"{name} sampling lost row identity")
    result = positions.astype(int, copy=False)
    if len(np.unique(result)) != len(result):
        raise ResidualOOFError(f"{name} sampling duplicated row identity")
    return result


def _side_loader(
    *,
    side: str,
    ae_root: Path,
    feature_store: Path,
    guard: TrainingResourceGuard,
) -> tuple[SideRepresentationFeatureLoader, list[str], dict[str, Any]]:
    ae_summary = json.loads((ae_root / "summary.json").read_text(encoding="utf-8"))
    ae_revision = str(ae_summary.get("source_revision") or "")
    contract, bundle, loader_hashes = _load_loader_contract(
        ae_root / side / "loader_evidence",
        source_revision=ae_revision,
    )
    raw_loader = make_fs_hpo_raw_feature_loader(
        feature_store_dir=feature_store,
        feature_contract=contract,
        evidence_bundle=bundle,
        resource_guard=guard,
    )
    ae_manifest_path = ae_root / side / "ae_gmm" / "side_stage_manifest.json"
    ae_manifest = json.loads(ae_manifest_path.read_text(encoding="utf-8"))
    state_path = ae_root / side / "ae_gmm" / str(ae_manifest["artifact"]["path"])
    state = _load_side_ae_state(
        state_path,
        expected_side=side,
        expected_sha256=str(ae_manifest["artifact"]["sha256"]),
        raw_features=contract["feature_columns"],
    )
    generated = list(_active_ae_gmm_columns(state))
    loader = SideRepresentationFeatureLoader(
        raw_loader=raw_loader,
        raw_features=contract["feature_columns"],
        state=state,
        generated_features=generated,
    )
    candidates = [*map(str, contract["feature_columns"]), *generated]
    evidence = {
        **loader_hashes,
        "ae_state_sha256": str(ae_manifest["artifact"]["sha256"]),
        "ae_manifest_sha256": _sha256(ae_manifest_path),
        "raw_candidate_features": len(contract["feature_columns"]),
        "generated_candidate_features": len(generated),
    }
    return loader, candidates, evidence


def _save_fold(
    root: Path,
    *,
    ev_map: IsotonicRegression,
    model: lgb.Booster,
    features: Sequence[str],
    params: Mapping[str, Any],
    rounds: int,
    alpha: float,
) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "residual_model.txt"
    model.save_model(str(model_path))
    calibrator_path = root / "baseline_ev_map.joblib"
    joblib.dump(ev_map, calibrator_path, compress=3)
    contract = {
        "features": list(features),
        "params": dict(params),
        "rounds": int(rounds),
        "alpha": float(alpha),
    }
    contract_path = root / "contract.json"
    _atomic_json(contract_path, contract)
    return {
        "model_sha256": _sha256(model_path),
        "baseline_ev_map_sha256": _sha256(calibrator_path),
        "contract_sha256": _sha256(contract_path),
    }


def run(
    *,
    population_path: Path,
    population_manifest_path: Path,
    outer_root: Path,
    inner_population_root: Path,
    ae_root: Path,
    labels_dir: Path,
    feature_store: Path,
    decisions_path: Path,
    feature_inventory_path: Path,
    destination: Path,
    selection_rows: int,
    hpo_rows: int,
    hpo_trials: int,
    hpo_patience: int,
    max_selected_features: int,
    outer_train_rows: int,
    seed: int,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite residual artifact: {destination}")
    revision = _git_revision()
    inner_manifest, source_hashes, fixed_calendar_sha256, feature_binding = (
        _source_contracts(
            population_root=Path(inner_population_root),
            feature_inventory_path=Path(feature_inventory_path),
            decisions_path=Path(decisions_path),
        )
    )
    label_files = _canonical_label_files(Path(labels_dir), inner_manifest)
    frame, population_manifest = _validate_population(
        population_path, population_manifest_path, outer_root
    )
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=destination.parent,
        telemetry_path=stage / "training_resource_telemetry.jsonl",
    )
    guard.preflight("packb_residual:preflight")
    label_loader = ExactLabelLoader(label_files, resource_guard=guard)
    side_reports: dict[str, Any] = {}
    all_predictions: list[pd.DataFrame] = []
    try:
        for side_index, side in enumerate(SIDES):
            guard.checkpoint(f"packb_residual:{side}:start")
            side_frame = frame.loc[frame["side_name"].astype(str).eq(side)].copy()
            _verify_labels(side_frame, labels=label_loader)
            representation_loader, representation_candidates, loader_evidence = (
                _side_loader(
                    side=side,
                    ae_root=ae_root,
                    feature_store=feature_store,
                    guard=guard,
                )
            )
            april = side_frame.loc[
                side_frame["__ts__"].ge("2026-04-01")
                & side_frame["__ts__"].lt("2026-05-01")
            ].copy()
            april_selection = _bounded_beginning_middle_end_sample(
                april,
                max_rows=int(selection_rows),
                name=f"{side}_residual_april_selection",
            )
            guard.checkpoint(f"packb_residual:{side}:load_selection_features")
            selection_representation = _load_representation(
                representation_loader, april_selection, representation_candidates
            )
            selection_matrix = _add_anchors(april_selection, selection_representation)
            active = _active_features(
                selection_matrix,
                [*representation_candidates, *ANCHORS],
                minimum_finite_fraction=0.95,
            )
            features, feature_report = _select_features(
                april_selection,
                selection_matrix,
                active,
                seed=seed + side_index * 1000,
                max_features=max_selected_features,
            )
            hpo_frame = _bounded_beginning_middle_end_sample(
                april,
                max_rows=int(hpo_rows),
                name=f"{side}_residual_april_hpo",
            )
            hpo_representation_features = [
                feature for feature in features if feature not in ANCHORS
            ]
            hpo_matrix = _add_anchors(
                hpo_frame,
                _load_representation(
                    representation_loader, hpo_frame, hpo_representation_features
                ),
            )
            params, rounds, alpha, trials = _hpo(
                hpo_frame,
                hpo_matrix,
                features,
                trials=hpo_trials,
                patience=hpo_patience,
                seed=seed + side_index * 1000 + 100,
                guard=guard,
            )
            side_root = stage / side
            side_root.mkdir(parents=True)
            feature_report.to_csv(side_root / "feature_selection.csv", index=False)
            trials.to_csv(side_root / "hpo_trials.csv", index=False)
            _atomic_json(
                side_root / "feature_contract.json",
                {
                    "side": side,
                    "features": features,
                    "sha256": _stable_hash(features),
                    "development_period": "2026-04-01/2026-05-01",
                    "april_oof_claim": False,
                },
            )
            _atomic_json(
                side_root / "hpo_contract.json",
                {
                    "side": side,
                    "params": params,
                    "rounds": rounds,
                    "alpha": alpha,
                    "trials_completed": len(trials),
                    "sha256": _stable_hash(
                        {"params": params, "rounds": rounds, "alpha": alpha}
                    ),
                },
            )
            guard.checkpoint(f"packb_residual:{side}:load_full_selected_features")
            full_representation_features = [
                feature for feature in features if feature not in ANCHORS
            ]
            full_matrix = _add_anchors(
                side_frame,
                _load_representation(
                    representation_loader, side_frame, full_representation_features
                ),
            )
            fold_reports: list[dict[str, Any]] = []
            for fold_index, (fold_id, start_text, end_text) in enumerate(FOLDS):
                start = pd.Timestamp(start_text)
                end = pd.Timestamp(end_text)
                train_mask = side_frame["__ts__"].lt(start) & side_frame[
                    "__label_resolution_ts__"
                ].lt(start)
                test_mask = side_frame["__ts__"].ge(start) & side_frame["__ts__"].lt(
                    end
                )
                train_indices = np.flatnonzero(train_mask.to_numpy())
                if len(train_indices) > int(outer_train_rows):
                    local_positions = _bounded_position_indices(
                        side_frame.iloc[train_indices],
                        max_rows=int(outer_train_rows),
                        name=f"{side}_{fold_id}_train",
                    )
                    train_indices = train_indices[local_positions]
                test_indices = np.flatnonzero(test_mask.to_numpy())
                if len(train_indices) < 5000 or len(test_indices) < 1000:
                    raise ResidualOOFError(f"{side} {fold_id} has insufficient support")
                train = side_frame.iloc[train_indices]
                test = side_frame.iloc[test_indices]
                score_train = pd.to_numeric(
                    train["prediction"], errors="coerce"
                ).to_numpy(np.float64)
                target_train = pd.to_numeric(
                    train[ECONOMIC_COLUMN], errors="coerce"
                ).to_numpy(np.float64)
                weight_train = pd.to_numeric(
                    train[WEIGHT_COLUMN], errors="coerce"
                ).to_numpy(np.float64)
                ev_map = _fit_ev_map(score_train, target_train, weight_train)
                baseline_train = _predict_ev_map(ev_map, score_train)
                model = _fit_residual_model(
                    full_matrix.iloc[train_indices],
                    target_train - baseline_train,
                    weight_train,
                    features,
                    {**params, "seed": seed + side_index * 1000 + fold_index + 500},
                    rounds=rounds,
                )
                score_test = pd.to_numeric(
                    test["prediction"], errors="coerce"
                ).to_numpy(np.float64)
                baseline_test = _predict_ev_map(ev_map, score_test)
                delta = np.asarray(
                    model.predict(full_matrix.iloc[test_indices][features]),
                    dtype=np.float64,
                )
                corrected = baseline_test + float(alpha) * delta
                train_cutoff = pd.to_datetime(
                    train["__label_resolution_ts__"], utc=True
                ).max()
                if not train_cutoff < start:
                    raise ResidualOOFError(f"{side} {fold_id} cutoff leaked")
                hashes = _save_fold(
                    side_root / "folds" / fold_id,
                    ev_map=ev_map,
                    model=model,
                    features=features,
                    params=params,
                    rounds=rounds,
                    alpha=alpha,
                )
                scored = test.loc[
                    :,
                    [
                        "candidate_id",
                        "side_name",
                        "__ts__",
                        "__symbol__",
                        "__label_resolution_ts__",
                        "prediction",
                        TARGET_COLUMN,
                        WEIGHT_COLUMN,
                        ECONOMIC_COLUMN,
                    ],
                ].copy()
                scored["residual_oof_fold"] = fold_id
                scored["residual_validation_start"] = start
                scored["residual_train_decision_cutoff"] = train_cutoff
                scored["base_expected_ev"] = baseline_test
                scored["residual_delta_ev"] = delta
                scored["residual_expected_ev"] = corrected
                scored["residual_prediction_available_at"] = pd.to_datetime(
                    scored["__ts__"], utc=True
                ) + pd.Timedelta(hours=1)
                scored["residual_is_oof"] = True
                all_predictions.append(scored)
                fold_reports.append(
                    {
                        "fold": fold_id,
                        "validation_start": start,
                        "validation_end": end,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_decision_cutoff": train_cutoff,
                        "base_metrics": _metrics(baseline_test, test),
                        "residual_metrics": _metrics(corrected, test),
                        "hashes": hashes,
                        "final_refit_prediction_used": False,
                    }
                )
                guard.checkpoint(f"packb_residual:{side}:{fold_id}:complete")
                del model, ev_map, delta, corrected
                gc.collect()
            side_predictions = pd.concat(
                [part for part in all_predictions if set(part["side_name"]) == {side}],
                ignore_index=True,
            )
            base_metrics = _metrics(
                side_predictions["base_expected_ev"].to_numpy(np.float64),
                side_predictions,
            )
            residual_metrics = _metrics(
                side_predictions["residual_expected_ev"].to_numpy(np.float64),
                side_predictions,
            )
            final_resolution_cutoff = pd.Timestamp.now(tz="UTC")
            final_indices = np.flatnonzero(
                side_frame["__label_resolution_ts__"]
                .lt(final_resolution_cutoff)
                .to_numpy()
            )
            if len(final_indices) != len(side_frame):
                raise ResidualOOFError(f"{side} final refit contains unresolved labels")
            if len(final_indices) > int(outer_train_rows):
                local_positions = _bounded_position_indices(
                    side_frame.iloc[final_indices],
                    max_rows=int(outer_train_rows),
                    name=f"{side}_final_refit",
                )
                final_indices = final_indices[local_positions]
            final_score = pd.to_numeric(
                side_frame.iloc[final_indices]["prediction"], errors="coerce"
            ).to_numpy(np.float64)
            final_target = pd.to_numeric(
                side_frame.iloc[final_indices][ECONOMIC_COLUMN], errors="coerce"
            ).to_numpy(np.float64)
            final_weight = pd.to_numeric(
                side_frame.iloc[final_indices][WEIGHT_COLUMN], errors="coerce"
            ).to_numpy(np.float64)
            final_ev_map = _fit_ev_map(final_score, final_target, final_weight)
            final_baseline = _predict_ev_map(final_ev_map, final_score)
            final_model = _fit_residual_model(
                full_matrix.iloc[final_indices],
                final_target - final_baseline,
                final_weight,
                features,
                {**params, "seed": seed + side_index * 1000 + 900},
                rounds=rounds,
            )
            final_hashes = _save_fold(
                side_root / "final_refit",
                ev_map=final_ev_map,
                model=final_model,
                features=features,
                params=params,
                rounds=rounds,
                alpha=alpha,
            )
            gate_passed, gate_checks = _promotion_gate(
                base=base_metrics,
                residual=residual_metrics,
                folds=fold_reports,
            )
            side_reports[side] = {
                "side": side,
                "model_side_scope": "per_side",
                "selected_feature_count": len(features),
                "feature_contract_sha256": _sha256(side_root / "feature_contract.json"),
                "hpo_contract_sha256": _sha256(side_root / "hpo_contract.json"),
                "loader_evidence": loader_evidence,
                "folds": fold_reports,
                "aggregate_base_metrics": base_metrics,
                "aggregate_residual_metrics": residual_metrics,
                "promotion_gate_passed": gate_passed,
                "promotion_gate_checks": gate_checks,
                "final_refit": {
                    "rows": len(final_indices),
                    "label_resolution_cutoff": final_resolution_cutoff,
                    "hashes": final_hashes,
                    "excluded_from_oof_metrics": True,
                    "predictions_persisted": False,
                },
            }
            _atomic_json(side_root / "manifest.json", side_reports[side])
            del (
                selection_representation,
                selection_matrix,
                hpo_matrix,
                full_matrix,
                representation_loader,
            )
            guard.checkpoint(f"packb_residual:{side}:released")
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions = predictions.sort_values(
            ["__ts__", "__symbol__", "side_name"], kind="mergesort"
        ).reset_index(drop=True)
        if predictions["candidate_id"].astype(str).duplicated().any():
            raise ResidualOOFError("residual OOF output has duplicate candidate IDs")
        prediction_path = stage / "oof_predictions.parquet"
        predictions.to_parquet(
            prediction_path, index=False, compression="zstd", compression_level=5
        )
        manifest = {
            "schema": SCHEMA,
            "status": (
                "PROMOTED_BOTH_SIDES"
                if all(side_reports[side]["promotion_gate_passed"] for side in SIDES)
                else "SIDE_GATE_REJECTION_BASE_PASSTHROUGH_REQUIRED"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": revision,
            "model_side_scope": "per_side",
            "shared_fitted_state": False,
            "population": {
                "path": str(population_path),
                "sha256": _sha256(population_path),
                "manifest_sha256": _sha256(population_manifest_path),
                "rows": len(frame),
                "candidate_identity_sha256": population_manifest.get(
                    "candidate_identity_sha256"
                ),
            },
            "source_hashes": source_hashes,
            "fixed_calendar_sha256": fixed_calendar_sha256,
            "feature_store_binding": feature_binding,
            "calendar": {
                "april": "development_only_base_passthrough_warmup",
                "oof_folds": [
                    {"fold": fold, "start": start, "end": end}
                    for fold, start, end in FOLDS
                ],
            },
            "cost_contract": (
                "__first_touch_capture_net__ already includes fixed 1% round-trip "
                "cost exactly once"
            ),
            "final_refit_predictions_used_in_oof": False,
            "oof_rows": len(predictions),
            "oof_predictions_sha256": _sha256(prediction_path),
            "sides": side_reports,
        }
        _atomic_json(stage / "manifest.json", manifest)
        guard.checkpoint("packb_residual:publish")
        os.replace(stage, destination)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument(
        "--population-manifest", type=Path, default=DEFAULT_POPULATION_MANIFEST
    )
    parser.add_argument("--outer-root", type=Path, default=DEFAULT_OUTER_ROOT)
    parser.add_argument(
        "--inner-population-root", type=Path, default=DEFAULT_INNER_POPULATION
    )
    parser.add_argument("--ae-root", type=Path, default=DEFAULT_AE_ROOT)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--feature-inventory", type=Path, default=DEFAULT_FEATURE_INVENTORY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-rows", type=int, default=45_000)
    parser.add_argument("--hpo-rows", type=int, default=45_000)
    parser.add_argument("--hpo-trials", type=int, default=75)
    parser.add_argument("--hpo-patience", type=int, default=20)
    parser.add_argument("--max-selected-features", type=int, default=64)
    parser.add_argument("--outer-train-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    manifest = run(
        population_path=args.population,
        population_manifest_path=args.population_manifest,
        outer_root=args.outer_root,
        inner_population_root=args.inner_population_root,
        ae_root=args.ae_root,
        labels_dir=args.labels_dir,
        feature_store=args.feature_store,
        decisions_path=args.decisions,
        feature_inventory_path=args.feature_inventory,
        destination=args.output_dir,
        selection_rows=args.selection_rows,
        hpo_rows=args.hpo_rows,
        hpo_trials=args.hpo_trials,
        hpo_patience=args.hpo_patience,
        max_selected_features=args.max_selected_features,
        outer_train_rows=args.outer_train_rows,
        seed=args.seed,
    )
    print(json.dumps(_jsonable(manifest), sort_keys=True))


if __name__ == "__main__":
    main()
