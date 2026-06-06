"""Training-to-live parity contract artifact helpers.

The contract is intentionally small and JSON-only: it records the exact model
feature contracts, strategy set, and deployment artifact hashes that live
inference must use.  It is not a source of trading logic; it is the audit
manifest that proves all trading logic came from the same artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from extreme_price_movements.inference.model_orchestrator import (
    DELETED_MODEL_FEATURE_KEYS,
    ModelOrchestrator,
    _effective_alpha_feature_contract,
    _effective_selected_feature_contract,
)
from extreme_price_movements.inference.parity import (
    _policy_artifact_bases,
    strategy_core_id,
    strategy_side,
)
from extreme_price_movements.inference.policy_rank_reference import (
    _policy_oos_contract_valid,
    strategy_rank_reference_aliases,
)
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


def sha256_tree(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_dir():
        return None
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        with child.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _artifact_entry(path: Path) -> Dict[str, Any]:
    resolved = resolve_mode_file(path)
    if resolved.is_dir():
        return {
            "path": str(resolved),
            "exists": True,
            "sha256": sha256_tree(resolved),
            "artifact_type": "directory_tree",
        }
    return {
        "path": str(resolved),
        "exists": bool(resolved.exists()),
        "sha256": sha256_file(resolved),
        "artifact_type": "file",
    }


def _first_existing(paths: Iterable[Path]) -> Path:
    paths_t = tuple(paths)
    for path in paths_t:
        resolved = resolve_mode_file(path)
        if resolved.exists():
            return resolved
    return resolve_mode_file(paths_t[0]) if paths_t else Path("")


def _contract_path_candidates(data_root: str, run_id: str) -> Tuple[Path, ...]:
    paths = []
    for run_root in _policy_artifact_bases(data_root, str(run_id)):
        paths.extend(
            [
                run_root / "policy_params" / PARITY_CONTRACT_FILENAME,
                run_root / "simple_policy_optimiser" / PARITY_CONTRACT_FILENAME,
                run_root / PARITY_CONTRACT_FILENAME,
            ]
        )
    expanded = []
    for path in paths:
        expanded.extend(mode_file_candidates(path))
    return tuple(dict.fromkeys(expanded))


def parity_contract_output_paths(data_root: str, run_id: str) -> Tuple[Path, ...]:
    run_root = _policy_artifact_bases(data_root, str(run_id))[0]
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
    policy_root = _policy_artifact_bases(data_root, str(run_id))[0]
    paths = {
        "base_models_intermediate": run_root / "base_models_intermediate.pkl",
        "trained_state": run_root / "models" / "trained_state.pkl",
        "meta_state": run_root / "models" / "model_state_meta.pkl",
        "native_model_dir": run_root / "models" / "native",
        "base_meta_contract": run_root / "base_meta_contract.json",
        "meta_feature_contract": run_root / "meta_oof" / "meta_feature_contract.json",
        "simple_policy_deployment": policy_root / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
        "simple_policy_rank_manifest": policy_root / "simple_policy_optimiser" / "rank_reference" / "manifest.json",
        "cross_strategy_rank_reference": policy_root / "simple_policy_optimiser" / "rank_reference" / "cross_strategy_auction.parquet",
        "optimized_portfolio_policy": _first_existing(
            (
                policy_root / "policy_params" / "optimized_portfolio_policy_config.json",
                policy_root
                / "portfolio_policy_replay"
                / "optimized_portfolio_policy_config.json",
            )
        ),
    }
    return {name: _artifact_entry(path) for name, path in paths.items()}


def _policy_rank_reference_universe_report(data_root: str, run_id: str) -> Dict[str, Any]:
    policy_root = _policy_artifact_bases(data_root, str(run_id))[0]
    rank_root = policy_root / "simple_policy_optimiser" / "rank_reference"
    report: Dict[str, Any] = {
        "checked": False,
        "rank_reference_dir": str(rank_root),
        "trained_universe_symbols": 0,
        "files_checked": 0,
        "outside_symbols": 0,
        "by_file": [],
    }
    if not rank_root.exists():
        report["reason"] = "rank_reference_dir_missing"
        return report
    try:
        from extreme_price_movements.inference.config import load_trained_symbol_universe

        trained = {str(sym) for sym in load_trained_symbol_universe(data_root, run_id)}
    except Exception as exc:
        report["reason"] = f"trained_universe_unavailable:{exc}"
        return report
    if not trained:
        report["reason"] = "trained_universe_empty"
        return report

    try:
        import pandas as pd
    except Exception as exc:
        report["reason"] = f"pandas_unavailable:{exc}"
        return report

    outside_all: set[str] = set()
    for path in sorted(rank_root.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["symbol"])
        except Exception:
            continue
        if "symbol" not in frame.columns:
            continue
        symbols = {str(sym) for sym in frame["symbol"].dropna().astype(str)}
        outside = sorted(symbols - trained)
        report["files_checked"] += 1
        outside_all.update(outside)
        report["by_file"].append(
            {
                "file": path.name,
                "symbols": len(symbols),
                "outside_symbols": len(outside),
                "outside_sample": outside[:20],
            }
        )

    report["checked"] = True
    report["trained_universe_symbols"] = len(trained)
    report["outside_symbols"] = len(outside_all)
    report["outside_sample"] = sorted(outside_all)[:30]
    return report


def _policy_rank_reference_contract_report(
    data_root: str,
    run_id: str,
    active_strategy_ids: Sequence[str],
) -> Dict[str, Any]:
    policy_root = _policy_artifact_bases(data_root, str(run_id))[0]
    manifest_path = (
        policy_root
        / "simple_policy_optimiser"
        / "rank_reference"
        / "manifest.json"
    )
    report: Dict[str, Any] = {
        "checked": False,
        "manifest_path": str(manifest_path),
        "errors": [],
        "strategy_errors": {},
    }
    if not manifest_path.exists():
        report["reason"] = "rank_reference_manifest_missing"
        return report
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["errors"].append(f"rank_reference_manifest_unreadable:{exc}")
        report["checked"] = True
        return report
    if not isinstance(manifest, dict):
        report["errors"].append("rank_reference_manifest_not_object")
        report["checked"] = True
        return report
    strategies = manifest.get("strategies") or {}
    if not isinstance(strategies, dict) or not strategies:
        report["checked"] = True
        report["reason"] = "rank_reference_manifest_has_no_strategies"
        return report
    manifest_contract = manifest.get("policy_oos_contract")
    if not isinstance(manifest_contract, dict) or not manifest_contract:
        report["errors"].append("missing_manifest_policy_oos_contract")
    elif not _policy_oos_contract_valid({"policy_oos_contract": manifest_contract}):
        report["errors"].append("invalid_manifest_policy_oos_contract")
    for strategy_id in _normalise_strategy_ids(active_strategy_ids):
        entry = None
        for alias in strategy_rank_reference_aliases(
            strategy_id, strategy_side(strategy_id)
        ):
            candidate = strategies.get(alias)
            if isinstance(candidate, dict):
                entry = candidate
                break
        if entry is None:
            report["strategy_errors"][strategy_id] = [
                "missing_strategy_rank_reference_manifest_entry"
            ]
            continue
        errors: list[str] = []
        contract = entry.get("policy_oos_contract")
        if not isinstance(contract, dict) or not contract:
            errors.append("missing_strategy_policy_oos_contract")
        elif not _policy_oos_contract_valid(entry):
            errors.append("invalid_strategy_policy_oos_contract")
        if errors:
            report["strategy_errors"][strategy_id] = errors
    report["checked"] = True
    return report


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
        "feature_source": {
            "run_id": str(
                os.environ.get("EPM_LIVE_FEATURE_SOURCE_RUN_ID")
                or os.environ.get("EPM_FEATURE_SOURCE_RUN_ID")
                or os.environ.get("EPM_ARTIFACT_SOURCE_RUN_ID")
                or ""
            ).strip()
            or None,
            "data_root": str(
                os.environ.get("EPM_LIVE_FEATURE_DATA_ROOT")
                or os.environ.get("EPM_FEATURE_DATA_ROOT")
                or data_root
            ),
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
    data_root: Optional[str] = None,
    run_id: Optional[str] = None,
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
    if data_root is not None and run_id is not None:
        expected_hashes = contract.get("artifact_hashes") or {}
        required = {
            "base_models_intermediate",
            "trained_state",
            "meta_state",
            "native_model_dir",
            "simple_policy_rank_manifest",
            "cross_strategy_rank_reference",
        }
        missing = sorted(k for k in required if k not in expected_hashes)
        if missing:
            if strict:
                raise ValueError(
                    "Training-live parity artifact contract missing required "
                    f"hash entries: {missing}"
                )
            return False
        current_hashes = _artifact_hashes(data_root, run_id)
        mismatches = []
        for key, expected_entry in expected_hashes.items():
            if key.startswith("_"):
                continue
            current_entry = current_hashes.get(str(key))
            if not isinstance(expected_entry, dict) or current_entry is None:
                continue
            expected_exists = bool(expected_entry.get("exists"))
            current_exists = bool(current_entry.get("exists"))
            expected_sha = expected_entry.get("sha256")
            current_sha = current_entry.get("sha256")
            if expected_exists != current_exists or expected_sha != current_sha:
                mismatches.append(
                    {
                        "artifact": str(key),
                        "expected_exists": expected_exists,
                        "current_exists": current_exists,
                        "expected_sha256": expected_sha,
                        "current_sha256": current_sha,
                    }
                )
        if mismatches:
            if strict:
                raise ValueError(
                    "Training-live parity artifact hash mismatch: "
                    f"{mismatches}"
                )
            return False
        universe_report = _policy_rank_reference_universe_report(data_root, run_id)
        if (
            universe_report.get("checked")
            and int(universe_report.get("outside_symbols") or 0) > 0
            and not bool(
                (contract.get("parity_policy") or {}).get(
                    "allow_rank_reference_symbols_outside_trained_universe",
                    False,
                )
            )
        ):
            if strict:
                raise ValueError(
                    "Training-live parity rank-reference universe mismatch: "
                    f"{universe_report}"
            )
            return False
        rank_contract_report = _policy_rank_reference_contract_report(
            data_root,
            run_id,
            active_strategy_ids,
        )
        rank_contract_errors = list(rank_contract_report.get("errors") or [])
        rank_contract_strategy_errors = dict(
            rank_contract_report.get("strategy_errors") or {}
        )
        if rank_contract_report.get("checked") and (
            rank_contract_errors or rank_contract_strategy_errors
        ):
            if strict:
                raise ValueError(
                    "Training-live parity rank-reference provenance mismatch: "
                    f"{rank_contract_report}"
                )
            return False
    return True
