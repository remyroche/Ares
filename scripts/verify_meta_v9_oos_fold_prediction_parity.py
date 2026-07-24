#!/usr/bin/env python3
"""Strictly replay persisted meta-v9 residual-expert OOS fold bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.supervised_market_state_calibration import (
    expected_ev_rank,
    predict_hierarchical_ev,
)

BUNDLE_SCHEMA = "side_base_residual_expert_oos_fold_bundle_v1"
KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_text_sha256(model: Any) -> str:
    if not hasattr(model, "model_to_string"):
        raise ValueError("residual model is not a serialized LightGBM Booster")
    return hashlib.sha256(model.model_to_string().encode("utf-8")).hexdigest()


def _validate_ev_calibrator(calibrator: Any, *, fold_name: str, name: str) -> None:
    """Reject incomplete train-reference mapping state before replaying it."""

    for attribute in (
        "global_model",
        "local_models",
        "local_weights",
        "rank_reference",
    ):
        if not hasattr(calibrator, attribute):
            raise ValueError(f"fold {fold_name} {name} lacks persisted {attribute}")
    if not callable(getattr(calibrator.global_model, "predict", None)):
        raise ValueError(f"fold {fold_name} {name} has no global prediction model")
    if not isinstance(calibrator.local_models, dict) or not isinstance(
        calibrator.local_weights, dict
    ):
        raise ValueError(f"fold {fold_name} {name} has invalid local calibration state")
    for key, local_model in calibrator.local_models.items():
        if not callable(getattr(local_model, "predict", None)):
            raise ValueError(f"fold {fold_name} {name} has an invalid local model")
        if key not in calibrator.local_weights or not np.isfinite(
            float(calibrator.local_weights[key])
        ):
            raise ValueError(f"fold {fold_name} {name} has an invalid local weight")
    reference = np.asarray(calibrator.rank_reference, dtype=np.float32)
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError(f"fold {fold_name} {name} lacks a valid train rank reference")


def _required(mapping: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _load_and_validate_bundle(fold_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_path = fold_dir / "bundle.joblib"
    manifest_path = fold_dir / "manifest.json"
    if not bundle_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            f"fold {fold_dir.name} must contain bundle.joblib and manifest.json"
        )
    bundle = joblib.load(bundle_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict) or not isinstance(manifest, dict):
        raise ValueError(f"fold {fold_dir.name} bundle and manifest must be mappings")
    for payload, name in ((bundle, "bundle"), (manifest, "manifest")):
        if payload.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(
                f"fold {fold_dir.name} {name} has unsupported bundle schema"
            )
        if payload.get("final_refit_included") is not False:
            raise ValueError(
                f"fold {fold_dir.name} {name} does not explicitly exclude final refit"
            )
        _required(payload, "fold_id", context=f"fold {fold_dir.name} {name}")
        _required(payload, "test_boundary", context=f"fold {fold_dir.name} {name}")
        _required(payload, "feature_contract", context=f"fold {fold_dir.name} {name}")
        _required(payload, "alpha_by_side", context=f"fold {fold_dir.name} {name}")
        _required(
            payload,
            "configured_model_params_by_side",
            context=f"fold {fold_dir.name} {name}",
        )
        _required(
            payload,
            "effective_model_params_by_side",
            context=f"fold {fold_dir.name} {name}",
        )
        _required(payload, "component_hashes", context=f"fold {fold_dir.name} {name}")
    if str(bundle["fold_id"]) != str(manifest["fold_id"]):
        raise ValueError(f"fold {fold_dir.name} bundle/manifest fold_id mismatch")
    for field in (
        "backbone_score",
        "backbone_score_col",
        "test_boundary",
        "feature_contract",
        "alpha_by_side",
        "configured_model_params_by_side",
        "effective_model_params_by_side",
        "component_hashes",
    ):
        if bundle.get(field) != manifest.get(field):
            raise ValueError(f"fold {fold_dir.name} bundle/manifest {field} mismatch")
    hashes = _required(manifest, "hashes", context=f"fold {fold_dir.name} manifest")
    if hashes.get("bundle_sha256") != _sha256_file(bundle_path):
        raise ValueError(f"fold {fold_dir.name} bundle hash does not match manifest")
    feature_contract = bundle["feature_contract"]
    alpha_by_side = bundle["alpha_by_side"]
    if not isinstance(feature_contract, dict) or not isinstance(alpha_by_side, dict):
        raise ValueError(f"fold {fold_dir.name} has invalid side contracts")
    if str(bundle["backbone_score"]) not in {"base", "meta"}:
        raise ValueError(f"fold {fold_dir.name} has an unsupported backbone score")
    for side in ("long", "short"):
        features = feature_contract.get(side)
        if not isinstance(features, list) or any(
            not isinstance(feature, str) for feature in features
        ):
            raise ValueError(
                f"fold {fold_dir.name} has invalid {side} feature contract"
            )
        if len(set(features)) != len(features):
            raise ValueError(f"fold {fold_dir.name} has duplicate {side} model inputs")
        alpha = alpha_by_side.get(side)
        if alpha is None or not np.isfinite(float(alpha)):
            raise ValueError(f"fold {fold_dir.name} has invalid {side} alpha")
    component_hashes = bundle["component_hashes"]
    expected_feature_hash = _stable_json_sha256(feature_contract)
    expected_alpha_hash = _stable_json_sha256(
        {side: float(alpha_by_side.get(side, 0.0)) for side in ("long", "short")}
    )
    if component_hashes.get("feature_contract_sha256") != expected_feature_hash:
        raise ValueError(f"fold {fold_dir.name} feature contract hash is invalid")
    if component_hashes.get("alpha_by_side_sha256") != expected_alpha_hash:
        raise ValueError(f"fold {fold_dir.name} alpha contract hash is invalid")
    if hashes.get("feature_contract_sha256") != expected_feature_hash:
        raise ValueError(
            f"fold {fold_dir.name} manifest feature contract hash is invalid"
        )
    if hashes.get("alpha_by_side_sha256") != expected_alpha_hash:
        raise ValueError(
            f"fold {fold_dir.name} manifest alpha contract hash is invalid"
        )
    models = _required(
        bundle, "residual_models", context=f"fold {fold_dir.name} bundle"
    )
    if not isinstance(models, dict):
        raise ValueError(f"fold {fold_dir.name} residual_models must be a mapping")
    model_hashes = {
        str(side): _model_text_sha256(model) for side, model in sorted(models.items())
    }
    if component_hashes.get("residual_model_text_sha256") != model_hashes:
        raise ValueError(f"fold {fold_dir.name} residual model hashes are invalid")
    if hashes.get("residual_model_text_sha256") != model_hashes:
        raise ValueError(
            f"fold {fold_dir.name} manifest residual model hashes are invalid"
        )
    for field in ("baseline_ev_map", "corrected_ev_map"):
        _required(bundle, field, context=f"fold {fold_dir.name} bundle")
    _validate_ev_calibrator(
        bundle["baseline_ev_map"], fold_name=fold_dir.name, name="baseline EV map"
    )
    _validate_ev_calibrator(
        bundle["corrected_ev_map"], fold_name=fold_dir.name, name="corrected EV map"
    )
    return bundle, manifest


def _test_mask(frame: pd.DataFrame, boundary: Any, fold_name: str) -> np.ndarray:
    if not isinstance(boundary, dict):
        raise ValueError(f"fold {fold_name} has invalid test boundary")
    start = pd.to_datetime(
        boundary.get("signal_timestamp_min"), utc=True, errors="coerce"
    )
    end = pd.to_datetime(
        boundary.get("signal_timestamp_max"), utc=True, errors="coerce"
    )
    rows = boundary.get("rows")
    if pd.isna(start) or pd.isna(end) or start > end or not isinstance(rows, int):
        raise ValueError(f"fold {fold_name} has invalid test boundary contract")
    return frame["__ts__"].ge(start).to_numpy() & frame["__ts__"].le(end).to_numpy()


def _validate_oos_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"oos_predictions.parquet is missing identity columns: {missing}"
        )
    out = frame.copy()
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True, errors="coerce")
    out["__symbol__"] = out["__symbol__"].astype("string")
    out["side_name"] = out["side_name"].astype("string").str.lower()
    if out[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(
            "oos_predictions.parquet has invalid timestamp/symbol/side identities"
        )
    if out.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(
            "oos_predictions.parquet has duplicate timestamp/symbol/side identities"
        )
    return out


def _score_columns(bundle: dict[str, Any]) -> dict[str, str]:
    backbone = str(bundle["backbone_score"])
    prefix = "score_meta" if backbone == "meta" else "score_base"
    return {
        "baseline_ev": f"{prefix}_ev_mapped",
        "corrected_ev": f"{prefix}_ev_residual_expert",
        "hierarchical_ev": f"{prefix}_ev_residual_expert_hier_mapped",
        "residual_delta": "meta_residual_expert_delta_ev",
        "baseline_rank": f"{prefix}_ev_rank_train_reference",
        "hierarchical_rank": f"{prefix}_residual_ev_rank_train_reference",
    }


def _replay_fold(
    frame: pd.DataFrame, bundle: dict[str, Any], fold_name: str
) -> dict[str, np.ndarray]:
    required = {str(bundle["backbone_score_col"]), "side_name", "archetype_policy_key"}
    for side, model in bundle["residual_models"].items():
        if side not in ("long", "short"):
            raise ValueError(
                f"fold {fold_name} has unsupported residual model side {side!r}"
            )
        if model is None:
            raise ValueError(f"fold {fold_name} has null residual model for {side}")
        features = bundle["feature_contract"].get(side)
        if not isinstance(features, list) or not features:
            raise ValueError(
                f"fold {fold_name} lacks an ordered feature contract for {side}"
            )
        required.update(str(feature) for feature in features)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"fold {fold_name} cannot reconstruct exact model inputs; missing {missing}"
        )
    raw = pd.to_numeric(
        frame[str(bundle["backbone_score_col"])], errors="coerce"
    ).to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError(f"fold {fold_name} has invalid backbone score rows")
    baseline_ev = predict_hierarchical_ev(bundle["baseline_ev_map"], frame, raw)
    residual = np.zeros(len(frame), dtype=np.float32)
    sides = frame["side_name"].astype(str).to_numpy()
    for side, model in bundle["residual_models"].items():
        mask = sides == str(side)
        if not mask.any():
            continue
        features = [str(feature) for feature in bundle["feature_contract"][side]]
        matrix = (
            frame.loc[mask, features]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        # LightGBM's native missing-value handling is part of the saved model;
        # preserve NaNs exactly and reject only infinities/non-numeric coercions.
        if np.isinf(matrix).any():
            raise ValueError(
                f"fold {fold_name} has infinite residual-model inputs for {side}"
            )
        residual[mask] = np.asarray(model.predict(matrix), dtype=np.float32)
    alpha = frame["side_name"].astype(str).map(bundle["alpha_by_side"]).fillna(0.0)
    alpha_values = alpha.to_numpy(dtype=np.float32)
    if not np.isfinite(alpha_values).all():
        raise ValueError(f"fold {fold_name} has invalid persisted alpha values")
    corrected_ev = baseline_ev + alpha_values * residual
    hierarchical_ev = predict_hierarchical_ev(
        bundle["corrected_ev_map"], frame, corrected_ev
    )
    return {
        "baseline_ev": baseline_ev.astype(np.float32),
        "corrected_ev": corrected_ev.astype(np.float32),
        "hierarchical_ev": hierarchical_ev.astype(np.float32),
        "residual_delta": (alpha_values * residual).astype(np.float32),
        "baseline_rank": expected_ev_rank(bundle["baseline_ev_map"], baseline_ev, raw),
        "hierarchical_rank": expected_ev_rank(
            bundle["corrected_ev_map"], hierarchical_ev, corrected_ev
        ),
    }


def _drift_summary(details: pd.DataFrame, columns: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in columns:
        abs_col = f"{name}_abs_drift"
        values = details[abs_col].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[name] = {
            "max_abs_drift": float(np.max(finite)) if len(finite) else None,
            "mean_abs_drift": float(np.mean(finite)) if len(finite) else None,
        }
    return summary


def verify_meta_v9_oos_fold_prediction_parity(
    *,
    report_dir: Path,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Replay every persisted OOS fold using only its bundle and stored row inputs."""

    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    report_dir = Path(report_dir)
    oos_path = report_dir / "oos_predictions.parquet"
    if not oos_path.is_file():
        raise ValueError(f"missing {oos_path}")
    fold_root = report_dir / "fold_models"
    fold_dirs = (
        sorted(path for path in fold_root.iterdir() if path.is_dir())
        if fold_root.is_dir()
        else []
    )
    if not fold_dirs:
        raise ValueError(f"no persisted fold bundles found under {fold_root}")
    oos = _validate_oos_frame(pd.read_parquet(oos_path))
    claimed = np.zeros(len(oos), dtype=bool)
    detail_frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    common_columns: dict[str, str] | None = None
    for fold_dir in fold_dirs:
        bundle, manifest = _load_and_validate_bundle(fold_dir)
        fold_id = str(bundle["fold_id"])
        mask = _test_mask(oos, bundle["test_boundary"], fold_id)
        if (claimed & mask).any():
            raise ValueError(f"fold {fold_id} overlaps another persisted fold boundary")
        fold = oos.loc[mask].copy()
        expected_rows = int(bundle["test_boundary"]["rows"])
        if len(fold) != expected_rows:
            raise ValueError(
                f"fold {fold_id} expected {expected_rows} persisted OOS rows but found {len(fold)}"
            )
        claimed |= mask
        columns = _score_columns(bundle)
        if common_columns is None:
            common_columns = columns
        elif columns != common_columns:
            raise ValueError(
                "persisted fold bundles disagree on the score column contract"
            )
        missing_scores = sorted(set(columns.values()) - set(fold.columns))
        if missing_scores:
            raise ValueError(
                f"fold {fold_id} is missing stored prediction columns: {missing_scores}"
            )
        replayed = _replay_fold(fold, bundle, fold_id)
        details = fold.loc[:, list(KEY_COLUMNS)].copy()
        details["fold_id"] = fold_id
        for name, stored_column in columns.items():
            expected = pd.to_numeric(fold[stored_column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            actual = np.asarray(replayed[name], dtype=np.float64)
            valid = np.isfinite(expected) & np.isfinite(actual)
            abs_drift = np.full(len(fold), np.nan, dtype=np.float64)
            abs_drift[valid] = np.abs(actual[valid] - expected[valid])
            allowed = float(absolute_tolerance) + float(relative_tolerance) * np.abs(
                expected
            )
            passed = valid & (abs_drift <= allowed)
            details[f"{name}_stored"] = expected
            details[f"{name}_replayed"] = actual
            details[f"{name}_abs_drift"] = abs_drift
            details[f"{name}_within_tolerance"] = passed
            details[f"{name}_missing_or_invalid"] = ~valid
        component_pass = details[[f"{name}_within_tolerance" for name in columns]].all(
            axis=1
        )
        component_invalid = details[
            [f"{name}_missing_or_invalid" for name in columns]
        ].any(axis=1)
        details["missing_or_invalid"] = component_invalid
        details["drift"] = ~component_pass & ~component_invalid
        details["verification_status"] = np.where(
            component_pass,
            "within_tolerance",
            np.where(component_invalid, "missing_or_invalid", "drift"),
        )
        detail_frames.append(details)
        fold_reports.append(
            {
                "fold_id": fold_id,
                "manifest_path": str(fold_dir / "manifest.json"),
                "bundle_path": str(fold_dir / "bundle.joblib"),
                "rows": int(len(details)),
                "missing_or_invalid_rows": int(component_invalid.sum()),
                "drift_rows": int((~component_pass & ~component_invalid).sum()),
                "drift": _drift_summary(details, columns),
                "pass": bool(component_pass.all()),
                "bundle_sha256": manifest["hashes"]["bundle_sha256"],
            }
        )
    if not claimed.all():
        raise ValueError(
            f"{int((~claimed).sum())} OOS rows are not covered by a persisted fold bundle"
        )
    details = pd.concat(detail_frames, ignore_index=True)
    if common_columns is None:
        raise ValueError("no fold score contracts were available for replay")
    columns = common_columns
    component_pass = details[[f"{name}_within_tolerance" for name in columns]].all(
        axis=1
    )
    side_reports = []
    for side, side_details in details.groupby("side_name", sort=True, dropna=False):
        side_pass = side_details[[f"{name}_within_tolerance" for name in columns]].all(
            axis=1
        )
        side_invalid = side_details["missing_or_invalid"].to_numpy(dtype=bool)
        side_reports.append(
            {
                "side_name": str(side),
                "rows": int(len(side_details)),
                "missing_or_invalid_rows": int(side_invalid.sum()),
                "drift_rows": int((~side_pass & ~side_invalid).sum()),
                "drift": _drift_summary(side_details, columns),
                "pass": bool(side_pass.all()),
            }
        )
    report = {
        "schema": "meta_v9_oos_fold_prediction_parity_v1",
        "report_dir": str(report_dir),
        "oos_predictions": str(oos_path),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "outcome_inputs_used": False,
        "overall": {
            "rows": int(len(details)),
            "missing_or_invalid_rows": int(details["missing_or_invalid"].sum()),
            "drift_rows": int(details["drift"].sum()),
            "drift": _drift_summary(details, columns),
            "pass": bool(component_pass.all()),
        },
        "fold_reports": fold_reports,
        "side_reports": side_reports,
        "pass": bool(component_pass.all() and all(row["pass"] for row in fold_reports)),
        "row_details": details,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    report = verify_meta_v9_oos_fold_prediction_parity(
        report_dir=args.report_dir,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    details = report.pop("row_details")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.details_output is not None:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(args.details_output, index=False)
    print(
        json.dumps(
            {"pass": report["pass"], "rows": report["overall"]["rows"]}, sort_keys=True
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
