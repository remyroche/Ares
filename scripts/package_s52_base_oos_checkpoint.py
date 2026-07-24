#!/usr/bin/env python3
"""Package a saved leakage-safe base OOS checkpoint into an inference bundle.

The meta learner is trained on base OOS scores.  Deploying a separately refit
base learner can change both the score scale and ordering seen by meta.  This
utility keeps the base-to-meta contract exact for a declared OOS validity
window by installing the saved base fold model, feature order, and matching
frozen AE/GMM transform into a copied candidate bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.feature_transform_contract import (
    FLOAT16_CLIPPED_THEN_FLOAT32_V1,
    FeatureSourceContract,
    build_model_input_numeric_contract,
    file_sha256,
    ordered_names_hash,
)
from extreme_price_movements.features_gmm_ae import (
    ae_gmm_learned_transform_hash,
    load_ae_gmm_state_artifact,
)

BASE_INPUT_NUMERIC_CONTRACT = FLOAT16_CLIPPED_THEN_FLOAT32_V1


def _physical_feature_store_run_id(value: str | None) -> str:
    """Normalize descriptive shared-store aliases to their physical run ID."""
    run_id = str(value or "").strip()
    suffix = "_shared_static_feature_store"
    return run_id[: -len(suffix)] if run_id.endswith(suffix) else run_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _normalized_sha256(value: Any) -> str:
    return str(value or "").strip().lower().removeprefix("sha256:")


def _ae_gmm_source_manifest_path(state_path: Path) -> Path:
    """Resolve the manifest emitted beside a serialized AE/GMM state."""
    candidates = []
    if state_path.name.endswith("_state.pkl"):
        candidates.append(
            state_path.with_name(
                state_path.name[: -len("_state.pkl")] + "_manifest.json"
            )
        )
    candidates.extend(
        [
            state_path.with_name("ae_gmm_state_manifest.json"),
            state_path.with_suffix(".manifest.json"),
            # Final base bundles keep the exact fitted-state contract separate
            # from the enclosing model manifest. Prefer it when present.
            state_path.with_name("source_state_manifest.json"),
            state_path.with_name("manifest.json"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"AE/GMM state has no adjacent source manifest: {state_path}"
    )


def _validate_ae_gmm_source_contract(
    state_path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Verify the exact fitted transform before installing it in a bundle."""
    state = load_ae_gmm_state_artifact(state_path)
    if not bool(state.get("enabled", False)):
        raise ValueError(f"AE/GMM state is disabled: {state_path}")
    input_features = [str(value) for value in state.get("feature_columns", []) or []]
    if not input_features:
        raise ValueError(f"AE/GMM state has no ordered input contract: {state_path}")
    actual_order_hash = ordered_names_hash(input_features)
    actual_transform_hash = ae_gmm_learned_transform_hash(state)
    expected_order_hash = manifest.get("input_feature_order_hash")
    expected_transform_hash = manifest.get("learned_transform_hash") or manifest.get(
        "cycle_state_hash"
    )
    if not expected_order_hash or _normalized_sha256(
        expected_order_hash
    ) != _normalized_sha256(actual_order_hash):
        raise ValueError("AE/GMM source input feature order does not match manifest")
    if not expected_transform_hash or _normalized_sha256(
        expected_transform_hash
    ) != _normalized_sha256(actual_transform_hash):
        raise ValueError("AE/GMM source learned transform does not match manifest")
    if not manifest.get("materialized_transform_rules"):
        raise ValueError("AE/GMM source manifest has no materialized transform rules")
    return {
        "input_feature_count": len(input_features),
        "input_feature_order_hash": _normalized_sha256(actual_order_hash),
        "learned_transform_hash": _normalized_sha256(actual_transform_hash),
    }


def _load_sealed_feature_source_contract(
    path: Path, *, feature_names: list[str], run_id: str | None
) -> tuple[FeatureSourceContract, dict[str, Any]]:
    payload = _load_json(path)
    contract = FeatureSourceContract.from_dict(payload)
    contract.validate_seal()
    if run_id and str(contract.run_id) != str(run_id):
        raise ValueError(
            f"Feature source run mismatch: {contract.run_id} != {run_id}"
        )
    if contract.model_feature_names_hash != ordered_names_hash(feature_names):
        raise ValueError("Feature source contract does not match base feature order")
    if str(contract.semantics.get("open_interest_unit") or "") != "quote_notional":
        raise ValueError("Feature source contract must declare quote-notional OI")
    source_root = Path(contract.source_root)
    for relative_path, record in contract.file_records.items():
        source_file = source_root / relative_path
        if not source_file.is_file():
            raise FileNotFoundError(f"Contracted feature source file missing: {source_file}")
        expected_hash = str(record.get("sha256") or "")
        if not expected_hash or file_sha256(source_file) != expected_hash:
            raise ValueError(f"Contracted feature source file changed: {source_file}")
    return contract, payload


def _install_base_checkpoint(
    state: dict[str, Any],
    *,
    model: Any,
    feature_names: list[str],
    input_numeric_contract: str = BASE_INPUT_NUMERIC_CONTRACT,
    input_numeric_contract_payload: dict[str, Any] | None = None,
) -> None:
    numeric_payload = input_numeric_contract_payload or build_model_input_numeric_contract(
        feature_names
    ).asdict()
    bundle = state.get("bundle")
    if not isinstance(bundle, dict):
        raise KeyError("Inference state has no bundle")
    alpha_models = bundle.get("alpha_models")
    if not isinstance(alpha_models, dict) or not alpha_models:
        raise KeyError("Inference state has no alpha_models")
    for side_key, side_state in alpha_models.items():
        if not isinstance(side_state, dict):
            raise TypeError(f"Invalid alpha model state for {side_key}")
        side_state["model"] = model
        side_state["feat_cols"] = list(feature_names)
        side_state["input_numeric_contract"] = str(input_numeric_contract)
        side_state["input_numeric_contract_payload"] = dict(numeric_payload)
        horizon = int(side_state.get("H", 5))
        side_state["models_by_h"] = {
            horizon: {
                "model": model,
                "feat_cols": list(feature_names),
                "input_numeric_contract": str(input_numeric_contract),
                "input_numeric_contract_payload": dict(numeric_payload),
            }
        }
    # The same checkpoint object is installed for both sides. Keep the contract
    # on the estimator as well so direct replay calls cannot bypass it.
    model.epm_input_numeric_contract_ = str(input_numeric_contract)
    model.epm_input_numeric_contract_payload_ = dict(numeric_payload)


def _install_trained_symbol_summary(
    output_bundle: Path,
    *,
    base_oos_ledger: Path | None,
) -> dict[str, Any] | None:
    """Persist the exact base-training universe for replay/live parity."""
    if base_oos_ledger is None or not base_oos_ledger.is_file():
        return None
    schema = set(pq.read_schema(base_oos_ledger).names)
    symbol_col = "__symbol__" if "__symbol__" in schema else "symbol"
    if symbol_col not in schema:
        raise ValueError(
            f"Base OOS ledger has no symbol column: {base_oos_ledger}"
        )
    symbols = (
        pd.read_parquet(base_oos_ledger, columns=[symbol_col])[symbol_col]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(kind="stable")
    )
    if symbols.empty:
        raise ValueError(f"Base OOS ledger has no trained symbols: {base_oos_ledger}")
    target = output_bundle / "features/feature_health_symbol_summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": symbols.to_numpy(copy=False)}).to_csv(
        target, index=False
    )
    return {
        "source": str(base_oos_ledger),
        "source_sha256": _sha256(base_oos_ledger),
        "path": str(target),
        "sha256": _sha256(target),
        "symbol_count": int(len(symbols)),
    }


def _install_meta_checkpoint(
    state: dict[str, Any], *, model: Any, feature_names: list[str]
) -> dict[str, Any]:
    bundle = state.get("bundle")
    if not isinstance(bundle, dict):
        raise KeyError("Inference state has no bundle")
    current = bundle.get("meta_models")
    if not isinstance(current, dict) or not current:
        raise KeyError("Inference state has no meta_models")
    template = next(iter(current.values()))
    for attr in (
        "s52_meta_ood_reference_",
        "s52_meta_ood_enabled_",
        "s52_meta_ood_input_features_",
    ):
        if hasattr(template, attr):
            setattr(model, attr, getattr(template, attr))
    model.selected_features = list(feature_names)
    model.feature_columns = list(feature_names)
    model.input_feature_names = list(feature_names)
    # The saved OOS checkpoint already emits the historical V9 score domain.
    # Carrying the final-refit alignment would distort it a second time.
    if hasattr(model, "s52_meta_score_alignment_"):
        delattr(model, "s52_meta_score_alignment_")
    for key in list(current):
        current[key] = model
    bundle["meta_models"] = current
    return current


def _write_meta_feature_contract(
    output_bundle: Path, *, model_keys: list[str], feature_names: list[str]
) -> None:
    payload = {
        "schema_version": "meta_feature_contract_v1",
        "run_id": output_bundle.name,
        "generated_by": "package_s52_base_oos_checkpoint",
        "meta_models": {
            str(key): {
                "model_key": str(key),
                "feature_columns": list(feature_names),
                "n_features": len(feature_names),
                "feature_contract_hash": hashlib.sha256(
                    json.dumps(
                        list(feature_names), separators=(",", ":"), ensure_ascii=True
                    ).encode("utf-8")
                ).hexdigest(),
                "positional_feature_mapping": {
                    f"f{idx}": name for idx, name in enumerate(feature_names)
                },
                "source": "saved walk-forward OOS meta checkpoint",
            }
            for key in model_keys
        },
    }
    target = output_bundle / "meta_oof/meta_feature_contract.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _install_meta_reliability_priors(
    output_bundle: Path,
    *,
    source: Path,
    valid_start: str | None,
) -> dict[str, Any]:
    """Install the train-fold priors paired with a saved OOS meta checkpoint."""
    payload = _load_json(source)
    rows = int(payload.get("rows", 0) or 0)
    global_prior = payload.get("global_prior", payload.get("global_stats"))
    if rows <= 0 or not isinstance(global_prior, dict):
        raise ValueError(f"Invalid meta reliability prior payload: {source}")

    prior_source = payload.get("source")
    if not isinstance(prior_source, dict):
        raise ValueError("Meta reliability priors must record their training source")
    train_end = str(prior_source.get("train_end_exclusive", "") or "").strip()
    if not train_end:
        raise ValueError(
            "Saved OOS meta checkpoints require reliability priors with "
            "source.train_end_exclusive"
        )
    if valid_start:
        import pandas as pd

        cutoff = pd.Timestamp(train_end)
        oos_start = pd.Timestamp(valid_start)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        if oos_start.tzinfo is None:
            oos_start = oos_start.tz_localize("UTC")
        else:
            oos_start = oos_start.tz_convert("UTC")
        if cutoff > oos_start:
            raise ValueError(
                "Meta reliability priors extend into the OOS checkpoint window: "
                f"train_end_exclusive={cutoff}, valid_start={oos_start}"
            )

    target = output_bundle / "policy_params/meta_reliability_priors.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source": str(source),
        "sha256": _sha256(target),
        "rows": rows,
        "groups": len(payload.get("groups", {})),
        "train_end_exclusive": train_end,
        "exact_groups_only": bool(payload.get("exact_groups_only", False)),
    }


def _install_v9_predecessor(
    output_bundle: Path,
    *,
    source: Path,
) -> dict[str, Any]:
    """Install and identify the exact frozen historical V9 predecessor."""
    if not source.is_file():
        raise FileNotFoundError(f"V9 predecessor bundle not found: {source}")
    predecessor = joblib.load(source)
    required_method = getattr(predecessor, "required_input_features", None)
    predict_method = getattr(predecessor, "predict", None)
    if not callable(required_method) or not callable(predict_method):
        raise TypeError(
            "V9 predecessor must implement required_input_features() and predict()"
        )
    target = output_bundle / "policy_params/v9_tail95_predecessor_bundle.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source": str(source),
        "sha256": _sha256(target),
        "class": (
            f"{predecessor.__class__.__module__}."
            f"{predecessor.__class__.__qualname__}"
        ),
        "required_input_feature_count": len(required_method()),
        "contract": "exact_frozen_historical_policy_predecessor_v1",
    }


def _refresh_training_live_parity_contract(
    output_bundle: Path,
    *,
    state: dict[str, Any],
    feature_source_run_id: str | None = None,
) -> list[str]:
    """Rebind model and policy hashes after all checkpoint substitutions."""
    from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
    from extreme_price_movements.inference.training_live_parity_contract import (
        build_training_live_parity_contract,
        load_training_live_parity_contract,
        persist_training_live_parity_contract,
    )

    previous = load_training_live_parity_contract(
        data_root=str(output_bundle.parent.parent),
        run_id=output_bundle.name,
        require=False,
    )
    feature_source = previous.get("feature_source", {})
    pinned_feature_source_run_id = str(feature_source_run_id or "").strip()
    pinned_feature_source_run_id = _physical_feature_store_run_id(
        pinned_feature_source_run_id
    )
    deployment_path = (
        output_bundle
        / "simple_policy_optimiser/deployment/best_policy_params_perps.json"
    )
    if not deployment_path.exists():
        deployment_path = (
            output_bundle / "simple_policy_optimiser/deployment/best_policy_params.json"
        )
    portfolio_path = output_bundle / "policy_params/optimized_portfolio_policy_config.json"
    deployment = _load_json(deployment_path) if deployment_path.exists() else {}
    portfolio = _load_json(portfolio_path) if portfolio_path.exists() else {}
    strategy_ids = sorted(
        str(key)
        for key in (state.get("bundle", {}).get("alpha_models", {}) or {})
        if str(key)
    )
    if not strategy_ids:
        raise RuntimeError("Checkpoint bundle has no alpha strategy routing keys")
    data_root = str(output_bundle.parent.parent)
    contract = build_training_live_parity_contract(
        data_root=data_root,
        run_id=output_bundle.name,
        market_mode=str(previous.get("market_mode") or "perps"),
        orchestrator=ModelOrchestrator(state, {}),
        model_bundle=state,
        strategy_ids=strategy_ids,
        deployment_payload=deployment,
        portfolio_payload=portfolio,
        feature_source_run_id=(
            pinned_feature_source_run_id
            or _physical_feature_store_run_id(feature_source.get("run_id"))
        ),
        feature_source_data_root=feature_source.get("data_root") or data_root,
    )
    return [
        str(path)
        for path in persist_training_live_parity_contract(
            contract,
            data_root=data_root,
            run_id=output_bundle.name,
        )
    ]


def _rebind_local_policy_artifact_paths(output_bundle: Path) -> list[str]:
    """Point runtime policy JSON at the files packaged in this artifact."""
    key_to_name = {
        "artifact_path": "composite_policy_regime_ev_calibration.json",
        "predecessor_bundle_path": "v9_tail95_predecessor_bundle.joblib",
        "residual_event_state_path": "residual_event_state.joblib",
        "regime_ev_calibration_artifact_path": (
            "composite_policy_regime_ev_calibration.json"
        ),
        "regime_ev_predecessor_bundle_path": "v9_tail95_predecessor_bundle.joblib",
        "regime_ev_residual_event_state_path": "residual_event_state.joblib",
        "threshold_basis_policy_path": (
            "threshold_basis_policy_sidearch_ev70_trim10_21d.json"
        ),
    }
    replacements = {
        key: str((output_bundle / "policy_params" / name).resolve())
        for key, name in key_to_name.items()
    }
    missing = [path for path in replacements.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Packaged policy path rebind targets are missing: {missing}"
        )

    def replace(value: Any, *, key: str = "") -> tuple[Any, int]:
        if isinstance(value, dict):
            changed = 0
            output: dict[str, Any] = {}
            for child_key, child_value in value.items():
                output[child_key], count = replace(
                    child_value,
                    key=str(child_key),
                )
                changed += count
            return output, changed
        if isinstance(value, list):
            output_list: list[Any] = []
            changed = 0
            for child in value:
                replaced, count = replace(child, key=key)
                output_list.append(replaced)
                changed += count
            return output_list, changed
        if key in replacements and isinstance(value, str):
            return replacements[key], int(value != replacements[key])
        return value, 0

    updated: list[str] = []
    for path in output_bundle.rglob("*.json"):
        try:
            payload = _load_json(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        rebound, changed = replace(payload)
        if changed:
            path.write_text(
                json.dumps(rebound, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            updated.append(str(path))
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--base-columns", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument(
        "--base-oos-ledger",
        type=Path,
        help=(
            "Base OOS ledger used to persist the exact trained-symbol universe. "
            "Defaults to best_oos_scored_ledger.parquet beside the base run."
        ),
    )
    parser.add_argument("--ae-gmm-state", type=Path, required=True)
    parser.add_argument("--meta-model", type=Path)
    parser.add_argument("--meta-columns", type=Path)
    parser.add_argument("--meta-manifest", type=Path)
    parser.add_argument("--meta-reliability-priors", type=Path)
    parser.add_argument(
        "--feature-source-run-id",
        default=None,
        help=(
            "Immutable feature-store snapshot used to construct the saved OOS "
            "checkpoint. It is persisted ahead of runtime defaults."
        ),
    )
    parser.add_argument(
        "--feature-source-contract",
        type=Path,
        required=True,
        help=(
            "Sealed feature_source_contract_v1 JSON for the exact feature "
            "snapshot used by the checkpoint. A run ID alone is insufficient."
        ),
    )
    parser.add_argument(
        "--base-reference-matrix-hash",
        default="",
        help="Optional exact historical model-matrix hash recorded by training.",
    )
    parser.add_argument(
        "--v9-predecessor-bundle",
        type=Path,
        help=(
            "Optional exact frozen V9 predecessor to install in the copied "
            "bundle. Use this when the source artifact contains a later "
            "adapter rather than the model used by historical policy replay."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_bundle.exists():
        raise FileExistsError(f"Output already exists: {args.output_bundle}")

    columns = _load_json(args.base_columns)
    feature_names = [str(value) for value in columns.get("feature_names", [])]
    if not feature_names:
        raise ValueError("Base checkpoint feature contract is empty")
    model = joblib.load(args.base_model)
    model_features = int(getattr(model, "n_features_in_", len(feature_names)))
    if model_features != len(feature_names):
        raise ValueError(
            f"Base model/contract mismatch: {model_features} != {len(feature_names)}"
        )
    feature_source_contract, feature_source_payload = (
        _load_sealed_feature_source_contract(
            args.feature_source_contract,
            feature_names=feature_names,
            run_id=args.feature_source_run_id,
        )
    )
    numeric_contract = build_model_input_numeric_contract(
        feature_names,
        reference_matrix_hash=str(args.base_reference_matrix_hash or ""),
    ).asdict()

    # Validate all external contracts before creating a partial output bundle.
    shutil.copytree(args.source_bundle, args.output_bundle, symlinks=True)
    # Runtime caches are keyed by artifact run and may contain feature matrices
    # from a different model/source contract. A packaged model must warm them
    # from its own immutable parity contract.
    shutil.rmtree(args.output_bundle / "live_state", ignore_errors=True)
    inferred_base_oos = args.base_model.parents[2] / "best_oos_scored_ledger.parquet"
    trained_symbol_summary = _install_trained_symbol_summary(
        args.output_bundle,
        base_oos_ledger=(args.base_oos_ledger or inferred_base_oos),
    )

    state_path = args.output_bundle / "models/trained_state.pkl"
    with state_path.open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, dict):
        raise TypeError("Inference trained state must be a dictionary")
    _install_base_checkpoint(
        state,
        model=model,
        feature_names=feature_names,
        input_numeric_contract_payload=numeric_contract,
    )
    state["feature_source_contract"] = feature_source_payload
    state.setdefault("bundle", {})["feature_source_contract"] = feature_source_payload
    state["base_oos_checkpoint_source"] = str(args.base_model)
    meta_override: dict[str, Any] | None = None
    if args.meta_model is not None:
        if (
            args.meta_columns is None
            or args.meta_manifest is None
            or args.meta_reliability_priors is None
        ):
            raise ValueError(
                "--meta-model requires --meta-columns, --meta-manifest, and "
                "--meta-reliability-priors"
            )
        meta_columns = _load_json(args.meta_columns)
        meta_features = [str(value) for value in meta_columns.get("feature_names", [])]
        meta_model = joblib.load(args.meta_model)
        if int(getattr(meta_model, "n_features_in_", -1)) != len(meta_features):
            raise ValueError("Meta checkpoint model/feature contract mismatch")
        meta_models = _install_meta_checkpoint(
            state, model=meta_model, feature_names=meta_features
        )
        meta_dir = args.output_bundle / "models/meta"
        for key in meta_models:
            target = meta_dir / str(key) / "base_soft_label.joblib"
            target.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(meta_model, target, compress=3)
        joblib.dump(
            {"run_id": args.output_bundle.name, "bundle": {"meta_models": meta_models}},
            args.output_bundle / "models/model_state_meta.pkl",
            compress=3,
        )
        _write_meta_feature_contract(
            args.output_bundle,
            model_keys=list(meta_models),
            feature_names=meta_features,
        )
        meta_source_manifest = _load_json(args.meta_manifest)
        reliability_prior_override = _install_meta_reliability_priors(
            args.output_bundle,
            source=args.meta_reliability_priors,
            valid_start=meta_source_manifest.get("valid_start"),
        )
        meta_override = {
            "contract": "saved_walk_forward_oos_checkpoint_v1",
            "model_source": str(args.meta_model),
            "model_sha256": _sha256(args.meta_model),
            "feature_source": str(args.meta_columns),
            "feature_count": len(meta_features),
            "feature_contract_hash": meta_columns.get("feature_contract_hash"),
            "valid_start": meta_source_manifest.get("valid_start"),
            "valid_end": meta_source_manifest.get("valid_end"),
            "fold": meta_source_manifest.get("fold"),
            "score_alignment": "identity_historical_oos_domain",
            "reliability_priors": reliability_prior_override,
        }
    with state_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ae_target = args.output_bundle / "ae_gmm_state/ae_gmm_state.pkl"
    ae_manifest_path = args.output_bundle / "ae_gmm_state/ae_gmm_state_manifest.json"
    source_ae_manifest_path = _ae_gmm_source_manifest_path(args.ae_gmm_state)
    # Start from the manifest that was emitted with this exact state. Reusing a
    # pre-existing bundle manifest can pair a new pickle with stale feature
    # order, transform rules, or cluster semantics.
    ae_manifest = _load_json(source_ae_manifest_path)
    validated_ae_contract = _validate_ae_gmm_source_contract(
        args.ae_gmm_state, ae_manifest
    )
    shutil.copy2(args.ae_gmm_state, ae_target)
    state_sha256 = _sha256(ae_target)
    ae_manifest.update(
        {
            "source": str(args.ae_gmm_state),
            "source_manifest": str(source_ae_manifest_path),
            "source_manifest_sha256": _sha256(source_ae_manifest_path),
            "sha256": state_sha256,
            "state_sha256": state_sha256,
            "contract": "single_cycle_frozen_ae_gmm_bundle_v2",
            "input_feature_count": validated_ae_contract["input_feature_count"],
            "input_feature_order_hash": validated_ae_contract[
                "input_feature_order_hash"
            ],
            "learned_transform_hash": validated_ae_contract[
                "learned_transform_hash"
            ],
            "cycle_state_hash": validated_ae_contract["learned_transform_hash"],
        }
    )
    ae_manifest_path.write_text(
        json.dumps(ae_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    feature_contract_target = (
        args.output_bundle / "feature_contracts/feature_source_contract.json"
    )
    feature_contract_target.parent.mkdir(parents=True, exist_ok=True)
    feature_contract_target.write_text(
        json.dumps(feature_source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base_manifest = _load_json(args.base_manifest)
    manifest_path = args.output_bundle / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest["run_id"] = args.output_bundle.name
    manifest["base_model_override"] = {
        "contract": "saved_walk_forward_oos_checkpoint_v1",
        "model_source": str(args.base_model),
        "model_sha256": _sha256(args.base_model),
        "feature_source": str(args.base_columns),
        "feature_count": len(feature_names),
        "feature_contract_hash": columns.get("feature_contract_hash"),
        "input_numeric_contract": BASE_INPUT_NUMERIC_CONTRACT,
        "input_numeric_contract_payload": numeric_contract,
        "feature_source_contract": str(feature_contract_target),
        "feature_source_contract_hash": feature_source_contract.contract_hash,
        "ae_gmm_source": str(args.ae_gmm_state),
        "ae_gmm_sha256": state_sha256,
        "ae_gmm_manifest": str(ae_manifest_path),
        "ae_gmm_manifest_sha256": _sha256(ae_manifest_path),
        "ae_gmm_input_feature_order_hash": ae_manifest.get(
            "input_feature_order_hash"
        ),
        "ae_gmm_learned_transform_hash": ae_manifest.get(
            "learned_transform_hash"
        ),
        "ae_gmm_cycle_state_hash": ae_manifest.get("cycle_state_hash"),
        "train_scope": base_manifest.get("leakage_contract", {}).get("fit_scope"),
        "valid_start": base_manifest.get("valid_start"),
        "valid_end": base_manifest.get("valid_end"),
        "fold": base_manifest.get("fold"),
    }
    if trained_symbol_summary is not None:
        manifest["base_model_override"]["trained_symbol_universe"] = (
            trained_symbol_summary
        )
    if meta_override is not None:
        manifest["meta_model_override"] = meta_override
    if args.v9_predecessor_bundle is not None:
        manifest["v9_predecessor_override"] = _install_v9_predecessor(
            args.output_bundle,
            source=args.v9_predecessor_bundle,
        )
    manifest["rebound_local_policy_artifacts"] = (
        _rebind_local_policy_artifact_paths(args.output_bundle)
    )
    manifest["training_live_parity_contracts"] = (
        _refresh_training_live_parity_contract(
            args.output_bundle,
            state=state,
            feature_source_run_id=(
                args.feature_source_run_id or feature_source_contract.run_id
            ),
        )
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["base_model_override"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
