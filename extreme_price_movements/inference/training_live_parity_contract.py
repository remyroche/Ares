"""Training-to-live parity contract artifact helpers.

The contract is intentionally small and JSON-only: it records the exact model
feature contracts, strategy set, and deployment artifact hashes that live
inference must use.  It is not a source of trading logic; it is the audit
manifest that proves all trading logic came from the same artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from extreme_price_movements.inference.model_orchestrator import (
    DELETED_MODEL_FEATURE_KEYS,
    ModelOrchestrator,
    _effective_alpha_feature_contract,
    _effective_selected_feature_contract,
)
from extreme_price_movements.inference.parity import strategy_core_id, strategy_side
from extreme_price_movements.path_utils import mode_file_candidates, resolve_mode_file


PARITY_CONTRACT_SCHEMA_VERSION = "training_live_parity_contract_v1"
PARITY_CONTRACT_FILENAME = "training_live_parity_contract.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_entry(path: Path) -> Dict[str, Any]:
    resolved = resolve_mode_file(path)
    return {
        "path": str(resolved),
        "exists": bool(resolved.exists()),
        "sha256": sha256_file(resolved),
    }


def _contract_path_candidates(data_root: str, run_id: str) -> Tuple[Path, ...]:
    run_root = Path(data_root) / "artifacts" / str(run_id)
    paths = [
        run_root / "policy_params" / PARITY_CONTRACT_FILENAME,
        run_root / "simple_policy_optimiser" / PARITY_CONTRACT_FILENAME,
        run_root / PARITY_CONTRACT_FILENAME,
    ]
    expanded = []
    for path in paths:
        expanded.extend(mode_file_candidates(path))
    return tuple(dict.fromkeys(expanded))


def parity_contract_output_paths(data_root: str, run_id: str) -> Tuple[Path, ...]:
    run_root = Path(data_root) / "artifacts" / str(run_id)
    return (
        run_root / "policy_params" / PARITY_CONTRACT_FILENAME,
        run_root / "simple_policy_optimiser" / PARITY_CONTRACT_FILENAME,
    )


def _normalise_strategy_ids(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    return tuple(sorted({str(v).strip() for v in values or [] if str(v).strip()}))


def _resolve_alpha(orchestrator: ModelOrchestrator, strategy_id: str) -> Tuple[str, Any]:
    side = strategy_side(strategy_id)
    core = strategy_core_id(strategy_id)
    alpha = getattr(orchestrator, "alpha_by_strategy", {}) or {}
    for key in (
        strategy_id,
        core,
        f"{side}_{strategy_id}" if side else "",
        f"{side}_{core}" if side and core else "",
    ):
        if key and isinstance(alpha.get(key), dict):
            return str(key), alpha.get(key)
    return str(strategy_id), None


def _resolve_meta(orchestrator: ModelOrchestrator, strategy_id: str) -> Tuple[str, Any]:
    side = strategy_side(strategy_id)
    core = strategy_core_id(strategy_id)
    meta = getattr(orchestrator, "meta_models", {}) or {}
    for key in (
        strategy_id,
        core,
        f"{side}_{strategy_id}" if side else "",
        f"{side}_{core}" if side and core else "",
        f"{strategy_id}_clf",
        f"{core}_clf" if core else "",
        f"{strategy_id}_tbm_clf",
        f"{core}_tbm_clf" if core else "",
    ):
        if key and key in meta:
            return str(key), meta.get(key)
    return str(strategy_id), None


def _model_contracts(
    orchestrator: ModelOrchestrator,
    strategy_ids: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for strategy_id in strategy_ids:
        alpha_key, alpha_info = _resolve_alpha(orchestrator, strategy_id)
        alpha_features = (
            _effective_alpha_feature_contract(alpha_info)
            if isinstance(alpha_info, dict)
            else []
        )
        meta_key, meta_model = _resolve_meta(orchestrator, strategy_id)
        meta_features = _effective_selected_feature_contract(meta_model)
        if not meta_features and meta_model is not None:
            meta_features = [
                str(c) for c in (getattr(meta_model, "feature_columns", []) or [])
            ]
        alpha_features = [
            str(c)
            for c in (alpha_features or [])
            if str(c) and str(c) not in DELETED_MODEL_FEATURE_KEYS
        ]
        meta_features = [
            str(c)
            for c in (meta_features or [])
            if str(c) and str(c) not in DELETED_MODEL_FEATURE_KEYS
        ]
        out[str(strategy_id)] = {
            "side": strategy_side(strategy_id),
            "strategy_core": strategy_core_id(strategy_id),
            "alpha_model_key": alpha_key,
            "alpha_feature_names": alpha_features,
            "alpha_feature_count": len(alpha_features),
            "meta_model_key": meta_key,
            "meta_feature_names": meta_features,
            "meta_feature_count": len(meta_features),
            "model_input_ordering": {
                "alpha": "alpha_feature_names",
                "meta": "meta_feature_names",
            },
        }
    return out


def _artifact_hashes(data_root: str, run_id: str) -> Dict[str, Any]:
    run_root = Path(data_root) / "artifacts" / str(run_id)
    paths = {
        "trained_state": run_root / "models" / "trained_state.pkl",
        "meta_state": run_root / "models" / "model_state_meta.pkl",
        "base_meta_contract": run_root / "base_meta_contract.json",
        "meta_feature_contract": run_root / "meta_oof" / "meta_feature_contract.json",
        "simple_policy_deployment": run_root / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
        "simple_policy_rank_manifest": run_root / "simple_policy_optimiser" / "rank_reference" / "manifest.json",
        "cross_strategy_rank_reference": run_root / "simple_policy_optimiser" / "rank_reference" / "cross_strategy_auction.parquet",
        "optimized_portfolio_policy": run_root / "policy_params" / "optimized_portfolio_policy_config.json",
    }
    return {name: _artifact_entry(path) for name, path in paths.items()}


def build_training_live_parity_contract(
    *,
    data_root: str,
    run_id: str,
    market_mode: str,
    orchestrator: Optional[ModelOrchestrator] = None,
    model_bundle: Optional[Dict[str, Any]] = None,
    strategy_ids: Optional[Iterable[str]] = None,
    deployment_payload: Optional[Dict[str, Any]] = None,
    portfolio_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if orchestrator is None:
        if model_bundle is None:
            raise ValueError("model_bundle or orchestrator is required")
        orchestrator = ModelOrchestrator(model_bundle, model_bundle)
    if strategy_ids is None:
        strategy_ids = sorted(str(k) for k in getattr(orchestrator, "alpha_by_strategy", {}).keys())
    strategy_ids_t = _normalise_strategy_ids(strategy_ids)
    strategy_cores_t = _normalise_strategy_ids(strategy_core_id(sid) for sid in strategy_ids_t)
    bundle = model_bundle if isinstance(model_bundle, dict) else {}
    loaded_bundle = bundle.get("bundle", {}) if isinstance(bundle.get("bundle"), dict) else {}
    return {
        "schema_version": PARITY_CONTRACT_SCHEMA_VERSION,
        "generated_by": "training_live_parity_contract",
        "run_id": str(run_id),
        "market_mode": str(market_mode),
        "strategy_contract": {
            "strategy_ids": list(strategy_ids_t),
            "strategy_cores": list(strategy_cores_t),
        },
        "model_contracts": _model_contracts(orchestrator, strategy_ids_t),
        "feature_transform_contract": {
            "hash": bundle.get("feature_transform_contract_hash")
            or loaded_bundle.get("feature_transform_contract_hash"),
            "manifest": bundle.get("feature_transform_manifest")
            or loaded_bundle.get("feature_transform_manifest"),
        },
        "rank_normalization": {
            "policy_rank_source": "policy_rank_reference_percentile",
            "cross_strategy_rank_source": "cross_strategy_auction_reference",
            "cross_strategy_reference_required": True,
        },
        "deployment_policy": deployment_payload or {},
        "portfolio_policy": portfolio_payload or {},
        "artifact_hashes": _artifact_hashes(data_root, run_id),
        "parity_policy": {
            "strict_model_inputs": True,
            "refuse_missing_model_features": True,
            "refuse_non_finite_model_features": True,
            "refuse_semantic_fallbacks": True,
            "cross_asset_universe_may_differ": True,
        },
    }


def persist_training_live_parity_contract(
    contract: Dict[str, Any],
    *,
    data_root: str,
    run_id: str,
) -> Tuple[Path, ...]:
    text = json.dumps(_json_safe(contract), indent=2, sort_keys=True)
    written = []
    for path in parity_contract_output_paths(data_root, run_id):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return tuple(written)


def load_training_live_parity_contract(
    *,
    data_root: str,
    run_id: str,
    require: bool = False,
) -> Dict[str, Any]:
    for path in _contract_path_candidates(data_root, run_id):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["_contract_path"] = str(path)
            payload["_contract_sha256"] = sha256_file(path)
            return payload
    if require:
        raise FileNotFoundError(
            "Training-live parity contract is required but missing: "
            f"{_contract_path_candidates(data_root, run_id)[0]}"
        )
    return {}


def validate_training_live_parity_contract(
    contract: Dict[str, Any],
    *,
    active_strategy_ids: Sequence[str],
    strict: bool = True,
) -> bool:
    expected = set(_normalise_strategy_ids(
        (contract.get("strategy_contract") or {}).get("strategy_ids")
    ))
    if not expected:
        return True
    active = set(_normalise_strategy_ids(active_strategy_ids))
    if active != expected:
        if strict:
            raise ValueError(
                "Training-live parity strategy contract mismatch: "
                f"active={sorted(active)} expected={sorted(expected)}"
            )
        return False
    return True

