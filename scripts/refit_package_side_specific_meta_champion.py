#!/usr/bin/env python3
"""Final-refit and package a strict side-specific S52 META champion.

This script deliberately does not run feature selection, HPO, or an OOS
evaluation.  It consumes a completed ``run_s52_train_meta_regime_handoff_smoke``
manifest, reuses its frozen side-local feature and winning-parameter contracts,
and refits one soft-label regressor per side on every permitted strict-top30
row.  The resulting package is a promotion candidate, not research evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_s52_train_meta_regime_handoff_smoke import (
    ALL_META_POST_SELECTION_OOD_FEATURE_NAMES,
    BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN,
    BASE_TARGET_CONTRACT_HASH_COLUMN,
    HANDOFF_RANK_SCOPE,
    _add_fold_base_prior_features,
    _add_fold_reliability_features,
    _base_soft_label_target,
    _candidate_column,
    _feature_contract_hash,
    _fit_base_soft_label_model,
    _load_and_validate_handoff_contract,
    _load_joined_frame,
    _make_xy,
    _projected_handoff_columns_for_selected,
    _target_strength_spec_from_contract,
)
from extreme_price_movements.inference.s52_meta_ood import (
    append_s52_meta_ood_features,
    fit_s52_meta_ood_reference,
)


SIDES = ("long", "short")
KEY_COLUMNS = ("__ts__", "__symbol__", "side_name")
SCHEMA = "side_specific_meta_champion_final_refit_v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Completed smoke manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Smoke manifest must be a JSON object: {manifest_path}")
    return payload, manifest_path


def _required_side_features(manifest: dict[str, Any]) -> dict[str, list[str]]:
    raw = manifest.get("selected_features_by_side")
    if not isinstance(raw, dict):
        raise ValueError("Smoke manifest is missing selected_features_by_side")
    result: dict[str, list[str]] = {}
    for side in SIDES:
        values = raw.get(side)
        if not isinstance(values, list):
            raise ValueError(f"Smoke manifest is missing {side} selected features")
        features = list(dict.fromkeys(str(value) for value in values if str(value).strip()))
        if not features:
            raise ValueError(f"Smoke manifest has an empty {side} feature contract")
        result[side] = features
    return result


def _winning_params(manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract the fixed soft-label regressor parameters without defaults."""
    candidates = (
        manifest.get("winning_params"),
        manifest.get("classifier_params"),
        (manifest.get("model_params") or {}).get("classifier")
        if isinstance(manifest.get("model_params"), dict)
        else None,
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            # Some experiment wrappers store both heads in winning_params.
            nested = candidate.get("classifier")
            return dict(nested) if isinstance(nested, dict) and nested else dict(candidate)
    raise ValueError("Smoke manifest is missing fixed winning classifier parameters")


def _validate_completed_side_specific_champion(manifest: dict[str, Any]) -> None:
    if str(manifest.get("generated_by", "")) != "run_s52_train_meta_regime_handoff_smoke":
        raise ValueError("Refit requires a completed run_s52_train_meta_regime_handoff_smoke manifest")
    if str(manifest.get("frontier", "")).lower() != "top30":
        raise ValueError("Refit requires the strict top30 META handoff frontier")
    if not bool(manifest.get("strict_handoff_contract")):
        raise ValueError("Refit requires strict_handoff_contract=true in the smoke manifest")
    if str(manifest.get("meta_head_mode", "")).lower() != "single_base_soft_label":
        raise ValueError("Refit requires the single_base_soft_label META winner")
    if not bool(manifest.get("side_specific_feature_contract_enabled")):
        raise ValueError("Refit requires a side-specific META feature contract")
    _required_side_features(manifest)
    _winning_params(manifest)
    inherited = manifest.get("inherited_base_handoff_contract")
    if not isinstance(inherited, dict) or inherited.get("validation_status") != "strict_pass":
        raise ValueError("Smoke manifest does not record a strict-passing inherited base contract")


def _prepare_permitted_rows(
    *,
    handoff_path: Path,
    ledger_path: Path,
    selected_by_side: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    selected_union = list(dict.fromkeys(
        feature for side in SIDES for feature in selected_by_side[side]
    ))
    handoff_columns = _projected_handoff_columns_for_selected(
        handoff_path, selected_union
    )
    data = _load_joined_frame(
        handoff_path, ledger_path, "top30", handoff_columns=handoff_columns
    )
    inherited = _load_and_validate_handoff_contract(
        handoff_path=handoff_path, handoff_rows=data, strict=True
    )
    candidate_col = _candidate_column("top30")
    if candidate_col not in data.columns:
        raise ValueError("Strict top30 handoff is missing selected_top30")
    input_rows = int(len(data))
    data = data.loc[data[candidate_col].fillna(False).astype(bool)].copy()
    if data.empty:
        raise ValueError("Strict top30 handoff contains no selected rows")
    data["__ts__"] = pd.to_datetime(data["__ts__"], utc=True, errors="coerce")
    data["side_name"] = data["side_name"].astype(str).str.lower()
    invalid_side = sorted(set(data.loc[~data["side_name"].isin(SIDES), "side_name"]))
    if invalid_side:
        raise ValueError(f"Strict top30 handoff has unsupported sides: {invalid_side}")
    if data.loc[:, list(KEY_COLUMNS)].duplicated().any():
        raise ValueError("Strict top30 handoff has duplicate decision-time keys")
    target, target_column = _base_soft_label_target(
        data, target_contract=inherited["base_target_contract"]
    )
    valid = data["__ts__"].notna() & target.notna()
    excluded = {
        "input_rows": input_rows,
        "strict_top30_rows": int(len(data)),
        "invalid_timestamp_rows": int(data["__ts__"].isna().sum()),
        "unresolved_target_rows": int(target.isna().sum()),
        "permitted_rows": int(valid.sum()),
    }
    data = data.loc[valid].copy()
    data["__final_refit_target__"] = target.loc[valid].astype(np.float32)
    if data.empty:
        raise ValueError("No permitted strict-top30 rows have resolved soft labels")
    missing_sides = [side for side in SIDES if not bool(data["side_name"].eq(side).any())]
    if missing_sides:
        raise ValueError(f"No permitted strict-top30 rows for side(s): {missing_sides}")
    inherited["resolved_target_column"] = target_column
    return data.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True), inherited, excluded


def _require_matching_inherited_contract(
    smoke_manifest: dict[str, Any], current_contract: dict[str, Any]
) -> None:
    """Reject a stale smoke result before it can refit a different base stream."""
    recorded = smoke_manifest.get("inherited_base_handoff_contract")
    if not isinstance(recorded, dict):
        raise ValueError("Smoke manifest is missing its inherited base handoff contract")
    for key in (
        BASE_TARGET_CONTRACT_HASH_COLUMN,
        BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN,
    ):
        if str(recorded.get(key) or "") != str(current_contract.get(key) or ""):
            raise ValueError(
                "Completed smoke result inherited base contract does not match "
                f"the strict handoff sidecar for {key}"
            )
    for key in ("base_target_contract", "base_sample_weight_spec"):
        if _sha256_json(recorded.get(key) or {}) != _sha256_json(
            current_contract.get(key) or {}
        ):
            raise ValueError(
                "Completed smoke result inherited base contract payload does not "
                f"match the strict handoff sidecar for {key}"
            )


def _load_frozen_preprocessing_contract(
    smoke_manifest: dict[str, Any],
) -> dict[str, list[str]]:
    """Load the exact numeric/categorical source partition from a saved fold.

    Prefix inference is not a valid substitute: a numeric feature can share a
    categorical source prefix (for example ``base_margin_to_cutoff_z``).  The
    completed smoke's ``columns.json`` is the only authoritative encoding
    contract for the final refit.
    """
    saved = smoke_manifest.get("saved_fold_models")
    if not isinstance(saved, list) or not saved:
        raise ValueError(
            "Final refit requires saved_fold_models with a frozen columns.json "
            "preprocessing contract"
        )
    contracts: list[dict[str, list[str]]] = []
    for record in saved:
        if not isinstance(record, dict):
            raise ValueError("Saved-fold metadata is malformed")
        raw_path = str(record.get("columns_path") or "").strip()
        path = Path(raw_path)
        if not raw_path or not path.is_file():
            raise ValueError(
                "Final refit cannot load saved fold columns.json: "
                f"{raw_path or '<missing>'}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        preprocessing = payload.get("preprocessing_state")
        input_contract = payload.get("input_feature_contract")
        if not isinstance(preprocessing, dict) or not isinstance(input_contract, dict):
            raise ValueError(f"Saved fold columns.json lacks preprocessing contract: {path}")
        numeric = [str(value) for value in preprocessing.get("numeric_columns", [])]
        categorical = [
            str(value)
            for value in preprocessing.get("categorical_source_columns", [])
        ]
        entries = input_contract.get("entries", [])
        encoded_sources = sorted(
            {
                str(entry.get("source_column"))
                for entry in entries
                if isinstance(entry, dict)
                and str(entry.get("source_type")) == "categorical_one_hot"
                and str(entry.get("source_column") or "").strip()
            }
        )
        if not numeric or sorted(set(categorical)) != encoded_sources:
            raise ValueError(
                "Saved fold preprocessing/input-feature contracts disagree on "
                f"numeric or categorical sources: {path}"
            )
        contracts.append(
            {
                "numeric_columns": list(dict.fromkeys(numeric)),
                "categorical_source_columns": list(dict.fromkeys(categorical)),
            }
        )
    reference = contracts[0]
    if any(contract != reference for contract in contracts[1:]):
        raise ValueError(
            "Saved fold preprocessing contracts differ; final refit cannot "
            "reproduce one frozen feature encoding"
        )
    return reference


def _final_training_matrix(
    frame: pd.DataFrame,
    selected_by_side: dict[str, list[str]],
    *,
    frozen_preprocessing: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Materialize the shared fold-time matrix before fitting side models.

    The OOS runner fits priors, categorical encodings, and post-selection OOD
    statistics on the complete training population.  Long/short contracts are
    applied only after that shared transform.  Repeating preprocessing inside
    each side would change both the priors and the OOD reference at inference.
    """
    selected = list(
        dict.fromkeys(
            feature
            for side in SIDES
            for feature in selected_by_side[side]
        )
    )
    requested_ood = [
        feature
        for feature in selected
        if feature in ALL_META_POST_SELECTION_OOD_FEATURE_NAMES
    ]
    core_features = [
        feature
        for feature in selected
        if feature not in ALL_META_POST_SELECTION_OOD_FEATURE_NAMES
    ]
    empty_valid = frame.iloc[:0].copy()
    train, _ = _add_fold_base_prior_features(
        frame, empty_valid, selected_col=_candidate_column("top30")
    )
    train, _ = _add_fold_reliability_features(train, empty_valid)
    frozen_numeric = set(frozen_preprocessing["numeric_columns"])
    categorical_sources = list(frozen_preprocessing["categorical_source_columns"])
    numeric = [feature for feature in core_features if feature in frozen_numeric]
    categorical_features = {
        feature
        for source in categorical_sources
        for feature in core_features
        if feature.startswith(f"{source}_")
    }
    unresolved = [
        feature
        for feature in core_features
        if feature not in set(numeric) and feature not in categorical_features
    ]
    if unresolved:
        raise ValueError(
            "Final refit selected core feature(s) are absent from the frozen "
            "numeric/categorical preprocessing contract: " + ", ".join(unresolved)
        )
    preprocessing: dict[str, Any] = {}
    matrix, _, encoded = _make_xy(
        train,
        train.iloc[:0].copy(),
        numeric_cols=numeric,
        categorical_cols=categorical_sources,
        selected_features=core_features,
        preprocessing_state_out=preprocessing,
    )
    missing = [feature for feature in core_features if feature not in matrix.columns]
    if missing:
        raise ValueError(
            "Final refit cannot materialize selected side feature(s): "
            + ", ".join(missing)
        )
    if encoded != core_features:
        matrix = matrix.reindex(columns=core_features)
    matrix = matrix.astype(np.float32, copy=False)
    if requested_ood:
        # A selected OOD aggregate is a derived model input, never a neutral
        # fallback. Recreate it from the final-refit training population using
        # the same selected pre-OOD core; failing here is preferable to
        # serializing a model whose feature contract cannot be reproduced.
        ood_reference = fit_s52_meta_ood_reference(matrix, core_features)
        if not bool(ood_reference.get("enabled", False)):
            raise ValueError(
                "Final refit cannot reproduce selected post-selection OOD "
                "feature(s) from the train-derived core contract: "
                + ", ".join(requested_ood)
            )
        matrix = append_s52_meta_ood_features(
            matrix,
            ood_reference,
            output_features=requested_ood,
        )
        unresolved_ood = [
            feature
            for feature in requested_ood
            if feature not in matrix.columns
            or not np.isfinite(
                pd.to_numeric(matrix[feature], errors="coerce").to_numpy(
                    dtype=np.float64
                )
            ).all()
        ]
        if unresolved_ood:
            raise ValueError(
                "Final refit could not materialize finite selected "
                "post-selection OOD feature(s): " + ", ".join(unresolved_ood)
            )
    else:
        ood_reference = {
            "enabled": False,
            "reason": "no_selected_post_selection_ood_features",
        }
    missing = [feature for feature in selected if feature not in matrix.columns]
    if missing:
        raise ValueError(
            "Final refit cannot materialize selected side feature(s): "
            + ", ".join(missing)
        )
    preprocessing["post_selection_ood_reference"] = _json_safe(ood_reference)
    preprocessing["post_selection_ood_outputs"] = list(requested_ood)
    return (
        matrix.reindex(columns=selected).astype(np.float32, copy=False),
        preprocessing,
        ood_reference,
    )


def _score_reference(model: Any, matrix: pd.DataFrame) -> dict[str, Any]:
    scores = np.asarray(model.predict(matrix), dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise RuntimeError("Final refit produced no finite training scores")
    probabilities = np.linspace(0.0, 1.0, 257)
    return {
        "schema": "meta_final_refit_score_reference_v1",
        "score_domain": "raw_model_prediction",
        "provenance": "final_refit_training_rows_in_sample",
        "oos": False,
        "rows": int(scores.size),
        "quantile_probabilities": probabilities.astype(float).tolist(),
        "quantiles": np.quantile(scores, probabilities).astype(float).tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


def _side_package(
    *,
    root: Path,
    side: str,
    model: Any,
    features: list[str],
    preprocessing: dict[str, Any],
    score_reference: dict[str, Any],
    inherited: dict[str, Any],
    winning_params: dict[str, Any],
    train_rows: int,
    target_hash: str,
    weight_hash: str,
    weight_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    side_dir = root / side
    side_dir.mkdir(parents=True, exist_ok=False)
    model_path = side_dir / "base_soft_label.joblib"
    features_path = side_dir / "features.json"
    metadata_path = side_dir / "metadata.json"
    joblib.dump(model, model_path, compress=3)
    model_sha256 = _sha256_file(model_path)
    feature_payload = {
        "schema": "side_specific_meta_feature_contract_v1",
        "side": side,
        "feature_names": features,
        "feature_count": len(features),
        "feature_contract_hash": _feature_contract_hash(features),
        "preprocessing": preprocessing,
    }
    features_path.write_text(json.dumps(_json_safe(feature_payload), indent=2, sort_keys=True), encoding="utf-8")
    features_sha256 = _sha256_file(features_path)
    metadata = {
        "schema": SCHEMA,
        "side": side,
        "model_path": model_path.name,
        "model_sha256": model_sha256,
        "features_path": features_path.name,
        "features_sha256": features_sha256,
        "model_class": type(model).__name__,
        "train_rows": int(train_rows),
        "winning_params": winning_params,
        "winning_params_hash": _sha256_json(winning_params),
        "inherited_base_contract": {
            "candidate_handoff_rank_scope": HANDOFF_RANK_SCOPE,
            "base_target_contract": inherited["base_target_contract"],
            BASE_TARGET_CONTRACT_HASH_COLUMN: target_hash,
            "base_sample_weight_spec": inherited["base_sample_weight_spec"],
            BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN: weight_hash,
        },
        "train_derived_priors": {
            "base_score_priors": "fit on all permitted final-refit rows",
            "reliability_priors": "fit on all permitted final-refit rows; training matrix uses leave-one-out values",
            "oos": False,
        },
        "score_reference": score_reference,
        "target_strength_weight_diagnostics": weight_diagnostics,
        "leakage_contract": {
            "status": "final_refit_non_oos",
            "excluded_from_oos_metrics": True,
            "feature_selection_and_hpo": "frozen from completed smoke manifest",
            "candidate_population": "strict timestamp_side top30 rows with resolved labels",
            "sample_weights": "recomputed from inherited target-strength contract on final-refit rows",
        },
    }
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "side": side,
        "model": str(model_path.relative_to(root)),
        "features": str(features_path.relative_to(root)),
        "metadata": str(metadata_path.relative_to(root)),
        "feature_contract_hash": feature_payload["feature_contract_hash"],
        "model_sha256": model_sha256,
        "features_sha256": features_sha256,
        "score_reference_hash": _sha256_json(score_reference),
        "train_rows": int(train_rows),
    }


def run_final_refit(
    *,
    smoke_result: Path,
    output_dir: Path,
    handoff_path: Path | None = None,
    ledger_path: Path | None = None,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Refit the frozen long/short champion and write an immutable package."""
    smoke_manifest, smoke_manifest_path = _read_manifest(smoke_result)
    _validate_completed_side_specific_champion(smoke_manifest)
    selected_by_side = _required_side_features(smoke_manifest)
    params = _winning_params(smoke_manifest)
    resolved_handoff = Path(handoff_path or smoke_manifest.get("handoff_path", ""))
    resolved_ledger = Path(ledger_path or smoke_manifest.get("ledger_path", ""))
    if not resolved_handoff.is_file() or not resolved_ledger.is_file():
        raise FileNotFoundError("Completed smoke manifest must resolve existing handoff and ledger parquet files")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite final-refit package: {output_dir}")

    data, inherited, exclusions = _prepare_permitted_rows(
        handoff_path=resolved_handoff,
        ledger_path=resolved_ledger,
        selected_by_side=selected_by_side,
    )
    _require_matching_inherited_contract(smoke_manifest, inherited)
    frozen_preprocessing = _load_frozen_preprocessing_contract(smoke_manifest)
    expected_target_hash = str(inherited.get(BASE_TARGET_CONTRACT_HASH_COLUMN) or "")
    expected_weight_hash = str(inherited.get(BASE_SAMPLE_WEIGHT_SPEC_HASH_COLUMN) or "")
    if not expected_target_hash or not expected_weight_hash:
        raise ValueError("Strict inherited base contract is missing target or sample-weight hash")
    weight_spec = _target_strength_spec_from_contract(inherited)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_", dir=output_dir.parent))
    staging = staging_parent / output_dir.name
    staging.mkdir()
    package_sides: dict[str, dict[str, Any]] = {}
    try:
        full_matrix, preprocessing, ood_reference = _final_training_matrix(
            data,
            selected_by_side,
            frozen_preprocessing=frozen_preprocessing,
        )
        for side_index, side in enumerate(SIDES, start=1):
            side_frame = data.loc[data["side_name"].eq(side)].copy().reset_index(drop=True)
            side_mask = data["side_name"].eq(side).to_numpy()
            matrix = (
                full_matrix.loc[side_mask, selected_by_side[side]]
                .reset_index(drop=True)
                .astype(np.float32, copy=False)
            )
            weights_diagnostics: dict[str, Any] = {}
            model = _fit_base_soft_label_model(
                matrix,
                side_frame["__final_refit_target__"],
                side_frame,
                int(seed) + side_index * 10_000,
                lgbm_params=params,
                target_strength_weight_spec=weight_spec,
                strict_handoff_contract=True,
                weight_diagnostics_out=weights_diagnostics,
            )
            if model is None:
                raise RuntimeError(f"Final refit did not fit a {side} soft-label model")
            # Inference materializes selected OOD aggregates from this frozen
            # model attribute. Keep the exact final-refit train reference with
            # each side model rather than relying on a live neutral fallback.
            if bool(ood_reference.get("enabled", False)):
                setattr(model, "s52_meta_ood_reference_", _json_safe(ood_reference))
            score_reference = _score_reference(model, matrix)
            side_manifest = _side_package(
                root=staging,
                side=side,
                model=model,
                features=selected_by_side[side],
                preprocessing=preprocessing,
                score_reference=score_reference,
                inherited=inherited,
                winning_params=params,
                train_rows=len(side_frame),
                target_hash=expected_target_hash,
                weight_hash=expected_weight_hash,
                weight_diagnostics=_json_safe(weights_diagnostics),
            )
            package_sides[side] = side_manifest

        manifest = {
            "schema": SCHEMA,
            "generated_by": "refit_package_side_specific_meta_champion",
            "source_smoke_manifest": str(smoke_manifest_path),
            "handoff_path": str(resolved_handoff),
            "ledger_path": str(resolved_ledger),
            "status": "final_refit_non_oos_pending_promotion",
            "excluded_from_oos_metrics": True,
            "frontier": "top30",
            "strict_handoff_contract": True,
            "all_permitted_rows_refit": True,
            "permitted_row_accounting": exclusions,
            "inherited_base_handoff_contract": inherited,
            "winning_params": params,
            "selected_features_by_side": selected_by_side,
            "shared_preprocessing": preprocessing,
            "post_selection_ood_reference": _json_safe(ood_reference),
            "sides": package_sides,
            "score_reference_and_side_comparability": {
                "raw_scores_directly_comparable_across_sides": False,
                "comparison_basis": "per-side raw-score quantile references in long/metadata.json and short/metadata.json",
                "reference_provenance": "final-refit in-sample training scores; not OOS calibration",
                "promotion_requirement": "later promotion must validate any cross-side score mapping on an allowed OOS contract",
            },
            "leakage_contract": {
                "final_refit": "all permitted resolved strict-top30 rows",
                "oos_claim": "none; package is explicitly excluded from OOS metrics",
                "features_and_params": "loaded frozen from completed smoke result",
                "base_provenance": "strict timestamp_side handoff with uniform inherited target and sample-weight hashes",
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
        staging.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--handoff-path", type=Path)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = run_final_refit(
        smoke_result=args.smoke_result,
        output_dir=args.output_dir,
        handoff_path=args.handoff_path,
        ledger_path=args.ledger_path,
        seed=args.seed,
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
