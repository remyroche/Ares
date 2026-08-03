#!/usr/bin/env python3
"""Strict, resumable February-warm-up auxiliary-head OOF runner.

The runner deliberately consumes the partitioned historical PIT context rather
than a future-enriched training table.  For every requested role and each
March/April outer month it redoes feature selection *and* HPO independently
per side using only decisions and 12-hour labels resolved before that month's
start.  February is therefore a legal warm-up source for March, but never a
shortcut around the label-resolution boundary.

The first bounded run is intentionally only the meaningful-MFE event and the
conditional peak magnitude.  ``--roles`` can extend it to the remaining fixed
role roadmap after these two artifacts have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_auxiliary_model_families import (  # noqa: E402
    ROLE_SPECS,
    ROLE_SPECS_BY_NAME,
    build_role_targets,
)
from extreme_price_movements.path_auxiliary_role_training import (  # noqa: E402
    _role_metrics,
    fit_auxiliary_role_model,
    select_auxiliary_role_features,
)
from extreme_price_movements.path_auxiliary_lgbm import (  # noqa: E402
    auxiliary_hpo_sample_indices,
    configured_auxiliary_feature_universe,
)


SCHEMA = "febapr2025_historical_auxiliary_role_oof_v1"
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
LABEL_IDENTITY = ("candidate_id", "side_name", "__ts__")
SIDES = ("long", "short")
OUTER_MONTHS = ("2025-03", "2025-04")
STARTER_ROLES = (
    "peak_mfe_12h_atr.p_hit",
    "peak_mfe_12h_atr.conditional_mean",
)
LABEL_COLUMNS = (
    *LABEL_IDENTITY,
    "__label_end_ts__",
    "__path_auxiliary_target_valid__",
    "__meaningful_mfe_reached_12h__",
    "__time_to_first_meaningful_mfe_hours_12h__",
    "__peak_mfe_atr_12h__",
    "__mae_before_meaningful_mfe_atr_12h__",
    "__bars_before_price_stops_decreasing_12h__",
    "__bars_to_confirmed_adverse_trough__",
    "__future_slope_atr_per_hour_12h__",
)
DEFAULT_CONTEXT = ROOT / "data_perp/artifacts/febapr2025_historical_path_head_context_20260727_v1"
DEFAULT_LABEL_DIR = ROOT / "data_perp/artifacts/20260723_s59_h5_path_aux_targets_v11_resolved_supportive_15atr/labels"
DEFAULT_STRICT_RESIDUAL = ROOT / "data_perp/artifacts/febapr2025_canonical_residual_oof_20260727_v1/oof_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/febapr2025_historical_auxiliary_oof_20260729_v2"

# These are the already-reviewed side-local geometries from the production
# slope-diagnostic bundle.  Historical PIT context predates a large part of
# that bundle's selected representation, so copying its *feature list* would
# silently discard most of the 2025 data contract.  The fixed-geometry mode
# below instead uses the frozen historical pre-entry auxiliary universe and
# freezes only the estimator geometry.  This is intentionally a materialising
# run, not a feature-selection or HPO experiment.
FROZEN_FUTURE_SLOPE_GEOMETRY: dict[str, dict[str, Any]] = {
    "long": {
        "learning_rate": 0.026118562667248452, "max_depth": 7,
        "num_leaves": 16, "min_child_samples": 100,
        "min_split_gain": 0.004744150623743692,
        "reg_alpha": 0.09419361797482531, "reg_lambda": 1.754747906814361,
        "subsample": 0.8387026228920781, "colsample_bytree": 0.7059247220158033,
        "max_bin": 63, "objective": "regression", "n_estimators": 126,
        "subsample_freq": 1,
    },
    "short": {
        "learning_rate": 0.02387603920833309, "max_depth": 4,
        "num_leaves": 24, "min_child_samples": 310,
        "min_split_gain": 0.00016522406824125756,
        "reg_alpha": 0.004691558635054462, "reg_lambda": 3.867795741735027,
        "subsample": 0.7045052360249506, "colsample_bytree": 0.6774700094022718,
        "max_bin": 63, "objective": "regression", "n_estimators": 272,
        "subsample_freq": 1,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, default=str, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _load_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or str(record["sha256"]) != _sha256(path):
        raise ValueError(f"checkpoint artifact is missing or changed: {path}")
    return path


def _month_start(month: str) -> pd.Timestamp:
    return pd.Timestamp(pd.Period(month, freq="M").start_time, tz="UTC")


def _utc(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    return result


def _identity_sha(frame: pd.DataFrame) -> str:
    keys = frame.loc[:, ["candidate_id", "side_name", "__ts__"]].copy()
    keys["__ts__"] = pd.to_datetime(keys["__ts__"], utc=True, errors="raise").astype(str)
    keys = keys.astype(str).sort_values(["candidate_id", "side_name", "__ts__"], kind="stable")
    return hashlib.sha256("\n".join("\x1f".join(row) for row in keys.itertuples(index=False, name=None)).encode()).hexdigest()


def _availability_sha(frame: pd.DataFrame, target: np.ndarray, available: np.ndarray) -> str:
    """Hash exact target availability without exposing a realised target as input."""
    values = pd.DataFrame({
        "candidate_id": frame["candidate_id"].astype(str),
        "side_name": frame["side_name"].astype(str),
        "__ts__": pd.to_datetime(frame["__ts__"], utc=True).astype(str),
        "available": np.asarray(available, dtype=bool).astype(np.int8),
        "target": np.where(np.asarray(available, dtype=bool), np.asarray(target, dtype=np.float32), np.nan),
    }).sort_values(["candidate_id", "side_name", "__ts__"], kind="stable")
    return hashlib.sha256(values.to_csv(index=False, float_format="%.8g").encode()).hexdigest()


def _strict_residual_population(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=["candidate_id", "side_name", "__ts__", "residual_is_oof"])
    raw = _utc(raw, ("__ts__",))
    strict = raw.loc[raw["residual_is_oof"].astype(bool), list(LABEL_IDENTITY)].copy()
    if len(strict) != 140_682 or strict.duplicated(list(LABEL_IDENTITY), keep=False).any():
        raise ValueError("strict residual OOF identity contract is not exactly 140,682 unique rows")
    months = strict["__ts__"].dt.strftime("%Y-%m")
    if not months.isin(OUTER_MONTHS).all():
        raise ValueError("strict residual OOF ledger must be signal-month March/April only")
    return strict


def _forbid_outcomes(feature_columns: Sequence[str]) -> None:
    forbidden = [
        name for name in feature_columns
        if name.startswith("__path_auxiliary_")
        or name in {
            "__label_end_ts__", "execution_net_ev_12h", "execution_label_end_utc",
            "native_label_resolution_utc", "__first_touch_target_soft__",
            "__first_touch_capture_net__", "__w__",
        }
    ]
    if forbidden:
        raise ValueError(f"future outcome columns cannot be model inputs: {forbidden[:10]}")


def _load_context(context_root: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    manifest_path = context_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "febapr2025_historical_path_head_context_v2_partitioned":
        raise ValueError("unexpected PIT context schema")
    if "pass:" not in str(manifest.get("context_index", {}).get("forbidden_outcome_scan", "")):
        raise ValueError("PIT context does not prove its forbidden-outcome scan")
    index = _utc(pd.read_parquet(context_root / "context_index.parquet"), ("__ts__",))
    if len(index) != 205_194 or index.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("historical PIT context index is not the frozen 205,194-row population")
    first_shard = Path(str(index["shard_manifest"].iloc[0]))
    first_data = Path(json.loads(first_shard.read_text(encoding="utf-8"))["output_path"])
    available = pd.read_parquet(first_data).columns.tolist()
    feature_columns, universe = configured_auxiliary_feature_universe(available)
    # The frozen alpha score is a pre-entry candidate-context field.  The
    # current generic universe names it ``score``; create that transparent
    # alias below without changing its value or using any realized outcome.
    if "base_oof_score" not in available:
        raise ValueError("PIT context is missing frozen pre-entry alpha score")
    _forbid_outcomes(feature_columns)
    requested = list(dict.fromkeys([*IDENTITY, "__decision_ts__", "base_oof_score", *feature_columns]))
    pieces: list[pd.DataFrame] = []
    for shard_manifest in index["shard_manifest"].drop_duplicates().tolist():
        shard = json.loads(Path(str(shard_manifest)).read_text(encoding="utf-8"))
        data = pd.read_parquet(Path(str(shard["output_path"])), columns=requested)
        pieces.append(data)
    context = _utc(pd.concat(pieces, ignore_index=True), ("__ts__", "__decision_ts__"))
    if len(context) != len(index) or context.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("sharded PIT context coverage changed while loading")
    if not context["__decision_ts__"].eq(context["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("PIT decision timestamp must remain signal timestamp plus one hour")
    context["score"] = pd.to_numeric(context["base_oof_score"], errors="coerce").astype(np.float32)
    features = list(dict.fromkeys([*feature_columns, "score"]))
    _forbid_outcomes(features)
    return context.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), features, universe


def _load_labels(label_dir: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for side in SIDES:
        path = label_dir / f"train_global_{side}_3.parquet"
        source = _utc(pd.read_parquet(path, columns=list(LABEL_COLUMNS)), ("__ts__", "__label_end_ts__"))
        if not source["side_name"].eq(side).all() or source.duplicated(list(LABEL_IDENTITY), keep=False).any():
            raise ValueError(f"invalid exact v6 label source for {side}")
        pieces.append(source)
    return pd.concat(pieces, ignore_index=True)


def load_inputs(context_root: Path, label_dir: Path, strict_residual: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, Any]]:
    context, features, universe = _load_context(context_root)
    labels = _load_labels(label_dir)
    joined = context.merge(labels, on=list(LABEL_IDENTITY), how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("one or more frozen PIT candidates are missing exact v6 labels")
    joined = joined.drop(columns="_merge")
    if not joined["__label_end_ts__"].eq(joined["__decision_ts__"] + pd.Timedelta(hours=12)).all():
        raise ValueError("auxiliary labels must resolve exactly 12h after decision")
    if not joined["side_name"].isin(SIDES).all():
        raise ValueError("unexpected side in joined historical population")
    strict = _strict_residual_population(strict_residual)
    strict_hash = _identity_sha(strict)
    joined["__strict_residual_oof__"] = joined["candidate_id"].isin(set(strict["candidate_id"]))
    actual = joined.loc[joined["__strict_residual_oof__"], list(LABEL_IDENTITY)]
    if len(actual) != len(strict) or _identity_sha(actual) != strict_hash:
        raise ValueError("PIT auxiliary rows do not exactly equal frozen strict residual OOF identities")
    return joined, features, universe, {"path": str(strict_residual.resolve()), "rows": int(len(strict)), "identity_sha256": strict_hash}


def _sample_reference(rows: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    local = auxiliary_hpo_sample_indices(rows["__decision_ts__"].to_numpy(), max_rows=maximum, random_state=seed)
    return rows.iloc[local].reset_index(drop=True)


def _selection_for_fold(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: np.ndarray,
    role_mask: np.ndarray,
    *,
    cutoff: pd.Timestamp,
    role: str,
    selection_rows: int,
    seed: int,
) -> dict[str, Any]:
    reference = frame["__decision_ts__"].lt(cutoff) & frame["__label_end_ts__"].lt(cutoff)
    eligible = reference.to_numpy() & role_mask & np.isfinite(target)
    selected = frame.loc[eligible, [*features, "__decision_ts__", "__symbol__", "side_name"]].reset_index(drop=True)
    selected_target = target[eligible]
    selected = _sample_reference(selected, selection_rows, seed)
    if len(selected) != len(selected_target):
        # sample indices above are deterministic, but preserve target alignment
        local = auxiliary_hpo_sample_indices(
            frame.loc[eligible, "__decision_ts__"].to_numpy(), max_rows=selection_rows, random_state=seed
        )
        source_target = target[eligible]
        selected_target = source_target[local]
    if len(selected) < 400:
        raise ValueError(f"{role}: fewer than 400 resolved role rows before {cutoff.isoformat()}")
    return select_auxiliary_role_features(
        selected.loc[:, list(features)], selected_target,
        task_kind=str(ROLE_SPECS_BY_NAME[role].task),
        timestamps=selected["__decision_ts__"].to_numpy(),
        assets=selected["__symbol__"].to_numpy(), sides=selected["side_name"].to_numpy(),
        # There is no historical learned archetype in the PIT contract. Symbol
        # is a pre-entry locality key only; it cannot leak a path outcome.
        archetypes=selected["__symbol__"].to_numpy(), role_name=role,
        random_state=seed, purge_hours=13.0,
    )


def _economic_proxy(frame: pd.DataFrame, prediction: np.ndarray, role_mask: np.ndarray) -> dict[str, Any]:
    """Target-proximal economic relevance, never a traded-policy backtest."""
    report: dict[str, Any] = {}
    for side in SIDES:
        for month in OUTER_MONTHS:
            mask = (
                frame["side_name"].eq(side).to_numpy()
                & frame["__strict_residual_oof__"].to_numpy()
                & frame["__ts__"].dt.strftime("%Y-%m").eq(month).to_numpy()
                & role_mask & np.isfinite(prediction)
            )
            local = frame.loc[mask]
            values = prediction[mask]
            if not len(local):
                item: dict[str, Any] = {"rows": 0}
            else:
                top_n = max(1, int(np.ceil(0.10 * len(local))))
                top = np.argsort(-values, kind="stable")[:top_n]
                hit = pd.to_numeric(local["__meaningful_mfe_reached_12h__"], errors="coerce").to_numpy(dtype=float)
                peak = pd.to_numeric(local["__peak_mfe_atr_12h__"], errors="coerce").to_numpy(dtype=float)
                item = {
                    "rows": int(len(local)), "top_decile_rows": int(top_n),
                    "all_meaningful_hit_rate": float(np.nanmean(hit)),
                    "top_decile_meaningful_hit_rate": float(np.nanmean(hit[top])),
                    "all_mean_peak_mfe_atr": float(np.nanmean(peak)),
                    "top_decile_mean_peak_mfe_atr": float(np.nanmean(peak[top])),
                    "hit_rate_lift": float(np.nanmean(hit[top]) - np.nanmean(hit)),
                    "peak_mfe_atr_lift": float(np.nanmean(peak[top]) - np.nanmean(peak)),
                }
            report[f"{side}/{month}"] = item
    return report


def _checkpoint(output: Path, fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    path = output / "checkpoint.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != dict(fingerprint):
            raise ValueError("existing auxiliary OOF checkpoint has a different input contract")
        return existing
    output.mkdir(parents=True, exist_ok=True)
    state = {"schema": SCHEMA, "fingerprint": dict(fingerprint), "completed": {}}
    _write_json(path, state)
    return state


def _save_checkpoint(output: Path, state: Mapping[str, Any]) -> None:
    payload = dict(state)
    payload["updated_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    _write_json(output / "checkpoint.json", payload)


def _save_progress(
    output: Path,
    *,
    role: str,
    month: str,
    stage: str,
    side: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> None:
    """Publish an atomic, deliberately non-authoritative live status record.

    ``checkpoint.json`` remains the only resume authority: it is written only
    after a whole role/month artifact is checksummed.  This separate record
    makes long side-local selection and HPO stages visible without pretending
    that an interrupted estimator is resumable mid-fit.
    """
    _write_json(
        output / "progress.json",
        {
            "schema": f"{SCHEMA}_progress_v1",
            "role": role,
            "month": month,
            "side": side,
            "stage": stage,
            "detail": dict(detail or {}),
            "updated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "resume_authority": "checkpoint.json only after a checksummed role/month artifact",
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    requested_roles = tuple(dict.fromkeys(args.roles))
    unknown = sorted(set(requested_roles).difference(ROLE_SPECS_BY_NAME))
    if unknown:
        raise ValueError(f"unknown auxiliary roles: {unknown}")
    if args.fixed_geometry and requested_roles != ("future_slope_atr_per_hour.diagnostic",):
        raise ValueError("--fixed-geometry is bound only to the future-slope diagnostic role")
    frame, features, universe, strict_contract = load_inputs(args.context_root, args.label_dir, args.strict_residual)
    targets = build_role_targets(frame, role_names=requested_roles)
    label_files = [_record(args.label_dir / f"train_global_{side}_3.parquet") for side in SIDES]
    fingerprint = {
        "schema": SCHEMA, "context_index": _record(args.context_root / "context_index.parquet"),
        "label_dir": str(args.label_dir.resolve()), "roles": list(requested_roles),
        "outer_months": list(OUTER_MONTHS), "feature_universe": list(features),
        "n_trials": int(args.n_trials), "hpo_rows": int(args.hpo_rows),
        "selection_rows": int(args.selection_rows), "seed": int(args.seed), "strict_residual": strict_contract,
        "label_files": label_files, "fixed_geometry": bool(args.fixed_geometry),
        "fixed_geometry_source": "packb future-slope side-local bundle 20260725_v1_31_8" if args.fixed_geometry else None,
    }
    state = _checkpoint(args.output_dir, fingerprint)
    predictions: dict[str, np.ndarray] = {role: np.full(len(frame), np.nan, dtype=np.float32) for role in requested_roles}
    folds: dict[str, dict[str, Any]] = {role: {} for role in requested_roles}
    for role_index, role in enumerate(requested_roles):
        role_target = targets[role]
        for month_index, month in enumerate(OUTER_MONTHS):
            cutoff = _month_start(month)
            fold_key = f"{role}/{month}"
            completed = state["completed"].get(fold_key)
            if completed:
                artifact = joblib.load(_load_record(completed))
                row_index = np.asarray(artifact["row_index"], dtype=np.int64)
                predictions[role][row_index] = np.asarray(artifact["prediction"], dtype=np.float32)
                folds[role][month] = artifact["report"]
                continue
            _save_progress(
                args.output_dir, role=role, month=month, stage="feature_selection_start",
                detail={"selection_rows_cap": int(args.selection_rows), "cutoff_utc": cutoff.isoformat()},
            )
            if args.fixed_geometry:
                selection = {
                    "role_name": role,
                    "task_kind": ROLE_SPECS_BY_NAME[role].task,
                    "selected_features_by_side": {side: list(features) for side in SIDES},
                    "selected_features": list(features),
                    "selection_metrics": {"contract": "no_feature_selection; frozen_historical_preentry_auxiliary_universe_v1"},
                    "feature_universe_report": universe,
                    "fixed_geometry_contract": "fixed estimator geometry; feature selection and HPO disabled",
                }
            else:
                selection = _selection_for_fold(
                    frame, features, role_target.target, role_target.train_mask, cutoff=cutoff,
                    role=role, selection_rows=int(args.selection_rows), seed=int(args.seed) + 1009 * role_index + month_index,
                )
            selected_by_side = selection["selected_features_by_side"]
            _save_progress(
                args.output_dir, role=role, month=month, stage="feature_selection_complete",
                detail={"selected_feature_counts": {side: len(selected_by_side[side]) for side in SIDES}},
            )
            valid_rows = np.flatnonzero(frame["__strict_residual_oof__"].to_numpy() & frame["__ts__"].dt.strftime("%Y-%m").eq(month).to_numpy())
            local_prediction = np.full(len(valid_rows), np.nan, dtype=np.float32)
            side_report: dict[str, Any] = {}
            side_models: dict[str, Any] = {}
            for side_index, side in enumerate(SIDES):
                side_rows = np.flatnonzero(frame["side_name"].eq(side).to_numpy())

                def progress_callback(event: str, payload: Mapping[str, Any], *, _side: str = side) -> None:
                    _save_progress(
                        args.output_dir, role=role, month=month, side=_side,
                        stage=f"model_{event}", detail=payload,
                    )

                _save_progress(args.output_dir, role=role, month=month, side=side, stage="model_start")
                fitted = fit_auxiliary_role_model(
                    frame.iloc[side_rows].loc[:, list(features)], role_target.target[side_rows],
                    role_train_mask=role_target.train_mask[side_rows], task_kind=ROLE_SPECS_BY_NAME[role].task,
                    selected_features=selected_by_side[side], timestamps=frame.iloc[side_rows]["__decision_ts__"].to_numpy(),
                    label_resolved_at=frame.iloc[side_rows]["__label_end_ts__"].to_numpy(),
                    selection_hpo_reference_end=cutoff, n_trials=int(args.n_trials),
                    hpo_rows=int(args.hpo_rows), hpo_patience=int(args.hpo_patience),
                    preset_params=FROZEN_FUTURE_SLOPE_GEOMETRY[side] if args.fixed_geometry else None,
                    random_state=int(args.seed) + 100_003 * role_index + 1_009 * month_index + side_index,
                    purge_hours=13.0, oof_months=(month,), role_name=role,
                    oof_validation_timestamps=frame.iloc[side_rows]["__ts__"].to_numpy(),
                    progress_callback=progress_callback,
                )
                chosen = valid_rows[frame.iloc[valid_rows]["side_name"].eq(side).to_numpy()]
                lookup = {int(row): position for position, row in enumerate(valid_rows)}
                local_positions = np.asarray([lookup[int(row)] for row in chosen], dtype=np.int64)
                local_index = np.searchsorted(side_rows, chosen)
                local_prediction[local_positions] = np.asarray(fitted["oof_predictions"], dtype=np.float32)[local_index]
                side_models[side] = fitted
                side_report[side] = {
                    "selected_features": selected_by_side[side],
                    "selected_features_sha256": hashlib.sha256("\n".join(selected_by_side[side]).encode()).hexdigest(),
                    "selection_metrics": (selection["selection_metrics"].get("by_side", {}).get(side, selection["selection_metrics"])),
                    "hpo": fitted["hpo"], "outer_fold": fitted["fold_provenance"],
                    "oof_metrics": fitted["oof_metrics"],
                    "fixed_geometry": FROZEN_FUTURE_SLOPE_GEOMETRY.get(side) if args.fixed_geometry else None,
                }
            predictions[role][valid_rows] = local_prediction
            metric_mask = role_target.train_mask[valid_rows] & np.isfinite(role_target.target[valid_rows])
            metrics = _role_metrics(role_target.target[valid_rows], local_prediction, metric_mask, task_kind=ROLE_SPECS_BY_NAME[role].task, quantile_alpha=0.80)
            report = {
                "month": month, "cutoff_utc": cutoff.isoformat(), "role": role,
                "validation_rows": int(len(valid_rows)), "conditional_metric_rows": int(metric_mask.sum()),
                "selection_contract": selection, "side": side_report, "metrics": metrics,
                "training_rule": "decision < fold start AND 12h label end < fold start; no March label enters March fit",
            }
            artifact = {"row_index": valid_rows, "prediction": local_prediction, "report": report, "models": side_models}
            path = args.output_dir / "folds" / role.replace(".", "__") / f"{month}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            joblib.dump(artifact, temporary)
            os.replace(temporary, path)
            state["completed"][fold_key] = _record(path)
            _save_checkpoint(args.output_dir, state)
            _save_progress(
                args.output_dir, role=role, month=month, stage="role_month_complete",
                detail={"artifact": state["completed"][fold_key]},
            )
            folds[role][month] = report
    output_cols = [*IDENTITY, "__decision_ts__", "__label_end_ts__", "__meaningful_mfe_reached_12h__", "__peak_mfe_atr_12h__"]
    output = frame.loc[frame["__strict_residual_oof__"], output_cols].copy()
    if len(output) != 140_682 or _identity_sha(output) != strict_contract["identity_sha256"]:
        raise AssertionError("emitted auxiliary OOF population differs from frozen strict residual OOF ledger")
    role_reports: dict[str, Any] = {}
    for role in requested_roles:
        column = "pred_" + role.replace(".", "__")
        # Models are fitted/scored against the full Feb--Apr context so
        # February can be a legal warm-up.  The emitted OOF ledger is the
        # smaller frozen strict-residual identity set; scatter by that exact
        # boolean mask rather than assuming the two populations have equal
        # length or matching positional indexes.
        output[column] = predictions[role][
            frame["__strict_residual_oof__"].to_numpy(dtype=bool)
        ]
        target = targets[role]
        mask = target.train_mask & np.isfinite(target.target) & np.isfinite(predictions[role])
        role_reports[role] = {
            "spec": {"task": ROLE_SPECS_BY_NAME[role].task, "source_column": target.source_column, "condition": ROLE_SPECS_BY_NAME[role].target_condition},
            "aggregate_oof_metrics": _role_metrics(target.target, predictions[role], mask, task_kind=ROLE_SPECS_BY_NAME[role].task, quantile_alpha=0.80),
            "folds": folds[role], "target_proximal_economic_diagnostic": _economic_proxy(frame, predictions[role], target.train_mask),
            "label_availability": {
                "target_valid_rows": int((target.train_mask & np.isfinite(target.target)).sum()),
                "target_valid_sha256": _availability_sha(frame, target.target, target.train_mask & np.isfinite(target.target)),
                "resolution_contract": "__label_end_ts__ == __decision_ts__ + 12h; fit labels resolve strictly before each validation start",
            },
        }
    predictions_path = args.output_dir / "oof_predictions.parquet"
    _write_parquet(predictions_path, output)
    roadmap = {spec.name: {"head": spec.head_name, "task": spec.task, "condition": spec.target_condition, "status": "trained_in_this_bounded_run" if spec.name in requested_roles else "pending_same_strict_side_local_fold_local_fs_hpo"} for spec in ROLE_SPECS}
    manifest = {
        "schema": SCHEMA, "status": "STRICT_SIDE_LOCAL_MARCH_APRIL_AUXILIARY_OOF_COMPLETE", "fingerprint": fingerprint,
        "input_contract": {"rows": int(len(frame)), "strict_residual_oof": strict_contract, "emitted_oof_rows": int(len(output)), "rows_by_side": output["side_name"].value_counts().sort_index().astype(int).to_dict(), "context_forbidden_outcome_scan": "pass", "label_resolution": "decision + 12h", "outer_validation_membership": "frozen strict residual candidate identities and signal __ts__ month; training remains decision/label-resolution purged", "february_warmup": "only labels with label_end < 2025-03-01 were eligible for March"},
        "feature_contract": {"configured_preentry_features": len(features), "universe": universe, "score_alias": "score = base_oof_score; frozen pre-entry only", "forbidden_outcome_inputs": "none"},
        "roles": role_reports, "complete_role_roadmap": roadmap,
        "oof_predictions": _record(predictions_path),
        "reproducibility": {
            "runner_source": _record(Path(__file__)),
            "checkpoint_authority": _record(args.output_dir / "checkpoint.json"),
            "context_index": _record(args.context_root / "context_index.parquet"),
        },
        "economic_diagnostic_scope": "target-proximal top-decile uplift only; not a global top-K traded-policy or net-PnL claim",
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-root", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--strict-residual", type=Path, default=DEFAULT_STRICT_RESIDUAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roles", nargs="+", default=list(STARTER_ROLES))
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--hpo-patience", type=int, default=4)
    parser.add_argument("--hpo-rows", type=int, default=20_000)
    parser.add_argument("--selection-rows", type=int, default=45_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-geometry", action="store_true", help="materialise only the slope ledger using frozen side-local geometry; disables FS/HPO")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "oof_predictions": result["oof_predictions"]}, indent=2))
