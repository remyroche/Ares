#!/usr/bin/env python3
"""Promote a full-scope model run using a previously selected live policy.

This does not rerun policy optimisation.  It copies the deployment policy
bundle selected on a policy-safe run into a full-scope final-fit model run and
then regenerates the training/live parity contract against the full-scope
model artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.parity import (
    resolve_deployment_strategy_filter,
    validate_deployment_model_coverage,
    validate_live_feature_contract,
    validate_meta_feature_contract_artifact,
)
from extreme_price_movements.inference.portfolio_policy import (
    load_portfolio_policy_config,
    validate_portfolio_strategy_contract,
)
from extreme_price_movements.inference.simple_policy_stop import (
    load_simple_policy_stop_params_by_strategy,
)
from extreme_price_movements.inference.training_live_parity_contract import (
    build_training_live_parity_contract,
    persist_training_live_parity_contract,
    validate_training_live_parity_contract,
)
from extreme_price_movements.model_loader import load_full_state


DEFAULT_POLICY_RUN = "20260612_183500_top2_reselect_labelhpo_drift_leaflite_native"
DEFAULT_MODEL_RUN = "20260612_203000_top2_fullscope_labelhpo_drift_leaflite_native"
DEFAULT_FEATURE_SOURCE_RUN = "20260523_015947"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _backup_or_remove(path: Path, backup_root: Path, *, overwrite: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / path.name
    if backup.exists() or backup.is_symlink():
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()
    shutil.move(str(path), str(backup))


def _copy_tree_or_file(src: Path, dst: Path, backup_root: Path, *, overwrite: bool) -> None:
    if not src.exists():
        return
    _backup_or_remove(dst, backup_root, overwrite=overwrite)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst)


def _selected_strategy_ids(payload: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in payload.get("strategies", []) or []:
        if not isinstance(row, Mapping) or row.get("selected") is False:
            continue
        sid = str(
            row.get("strategy_for_inference")
            or row.get("strategy_id")
            or row.get("canonical_strategy_id")
            or ""
        ).strip()
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return sorted(out)


def _required(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts: " + ", ".join(missing))


def _copy_policy_bundle(
    *,
    data_root: Path,
    policy_run: str,
    model_run: str,
    overwrite: bool,
) -> list[str]:
    src_root = data_root / "artifacts" / policy_run
    dst_root = data_root / "artifacts" / model_run
    backup_root = dst_root / f"policy_promotion_backup_{_now_tag()}"

    _required(
        [
            src_root / "policy_params" / "strategy_for_inference.json",
            src_root / "policy_params" / "best_policy_params.json",
            src_root / "policy_params" / "optimized_portfolio_policy_config.json",
            src_root / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
            src_root / "simple_policy_optimiser" / "rank_reference" / "manifest.json",
            src_root
            / "simple_policy_optimiser"
            / "rank_reference"
            / "cross_strategy_auction.parquet",
        ]
    )

    copied: list[str] = []
    rels: Sequence[str] = (
        "policy_params",
        "simple_policy_optimiser",
        "portfolio_policy_replay",
        "best_policy_params.json",
        "best_policy_params_perps.json",
        "strategy_for_inference.json",
        "strategy_for_inference_perps.json",
    )
    for rel in rels:
        src = src_root / rel
        dst = dst_root / rel
        if not src.exists():
            continue
        _copy_tree_or_file(src, dst, backup_root, overwrite=overwrite)
        copied.append(rel)
    return copied


def _remove_legacy_sizer_adapter(data_root: Path, model_run: str) -> list[str]:
    """Remove promotion-created legacy sizer adapter artifacts if present."""
    removed: list[str] = []
    path = data_root / "artifacts" / model_run / "ridge_sizer" / "strategy_params.json"
    if path.exists() or path.is_symlink():
        path.unlink()
        removed.append(str(path))
    return removed


def _simple_policy_stop_bucket_params(
    *,
    data_root: Path,
    model_run: str,
    strategy_ids: Sequence[str],
) -> dict[str, Any]:
    """Return the runtime bucket-param shape backed by simple_policy_optimiser.

    Live inference already loads stop parameters from
    ``simple_policy_optimiser/deployment/best_policy_params*.json``.  Keep the
    promotion validator on that same source instead of creating legacy
    ``ridge_sizer`` or ``simple_position_sizer`` adapter artifacts.
    """
    stop_params = load_simple_policy_stop_params_by_strategy(
        str(data_root),
        run_id=model_run,
    )
    missing = [sid for sid in strategy_ids if sid not in stop_params]
    if missing:
        raise ValueError(
            "Unable to derive simple-policy stop params for selected strategies: "
            + ", ".join(missing)
        )
    return {
        "simple_policy_stop_params_by_strategy": {
            sid: dict(stop_params[sid]) for sid in strategy_ids
        },
    }


def promote(
    *,
    data_root: Path,
    policy_run: str,
    model_run: str,
    market_mode: str,
    feature_source_run: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    model_root = data_root / "artifacts" / model_run
    policy_root = data_root / "artifacts" / policy_run
    if not model_root.exists():
        raise FileNotFoundError(model_root)
    if not policy_root.exists():
        raise FileNotFoundError(policy_root)
    _required(
        [
            model_root / "base_models_intermediate.pkl",
            model_root / "models" / "trained_state.pkl",
            model_root / "models" / "model_state_meta.pkl",
            model_root / "meta_oof" / "meta_feature_contract.json",
        ]
    )

    copied = _copy_policy_bundle(
        data_root=data_root,
        policy_run=policy_run,
        model_run=model_run,
        overwrite=overwrite,
    )
    removed_legacy_adapters = _remove_legacy_sizer_adapter(data_root, model_run)

    strategy_payload = _load_json(model_root / "policy_params" / "strategy_for_inference.json")
    portfolio_payload = _load_json(
        model_root / "policy_params" / "optimized_portfolio_policy_config.json"
    )
    strategy_ids = _selected_strategy_ids(strategy_payload)
    if not strategy_ids:
        raise ValueError("Copied strategy_for_inference has no selected strategies")
    simple_policy_bucket_params = _simple_policy_stop_bucket_params(
        data_root=data_root,
        model_run=model_run,
        strategy_ids=strategy_ids,
    )

    previous_feature_source = None
    previous_contract_path = policy_root / "policy_params" / "training_live_parity_contract.json"
    if previous_contract_path.exists():
        previous_contract = _load_json(previous_contract_path)
        previous_feature_source = previous_contract.get("feature_source")
        if feature_source_run is None and isinstance(previous_feature_source, Mapping):
            feature_source_run = str(previous_feature_source.get("run_id") or "") or None

    feature_source_run = str(feature_source_run or DEFAULT_FEATURE_SOURCE_RUN)
    os.environ["EPM_LIVE_FEATURE_SOURCE_RUN_ID"] = str(feature_source_run)
    full_state = load_full_state(model_run, str(data_root))
    model_bundle = full_state.get("bundle", full_state) if isinstance(full_state, dict) else {}
    if isinstance(model_bundle, dict) and isinstance(full_state, dict):
        model_bundle["bucket_params"] = simple_policy_bucket_params
        full_state["bucket_params"] = simple_policy_bucket_params
        model_bundle.setdefault("regime_adaptors", full_state.get("regime_adaptors", {}) or {})
    orchestrator = ModelOrchestrator(model_bundle, full_state)
    contract = build_training_live_parity_contract(
        data_root=str(data_root),
        run_id=model_run,
        market_mode=market_mode,
        orchestrator=orchestrator,
        model_bundle=full_state,
        strategy_ids=strategy_ids,
        deployment_payload=strategy_payload,
        portfolio_payload=portfolio_payload,
    )
    contract["policy_artifact_source"] = {
        "run_id": policy_run,
        "copied_into_run_id": model_run,
        "copied_artifacts": copied,
        "policy_params_source": (
            f"data_perp/artifacts/{model_run}/simple_policy_optimiser/"
            "deployment/best_policy_params.json"
        ),
        "stop_params_source": "simple_policy_optimiser",
        "removed_legacy_adapter_artifacts": removed_legacy_adapters,
    }
    if isinstance(previous_feature_source, Mapping):
        contract["source_policy_feature_source"] = dict(previous_feature_source)
    written = persist_training_live_parity_contract(
        contract,
        data_root=str(data_root),
        run_id=model_run,
    )

    accepted = set(resolve_deployment_strategy_filter(str(data_root), model_run) or [])
    if set(strategy_ids) != accepted:
        raise ValueError(
            "Deployment strategy resolution mismatch: "
            f"selected={strategy_ids} resolved={sorted(accepted)}"
        )

    validate_live_feature_contract(model_bundle, strict=True)
    validate_portfolio_strategy_contract(
        load_portfolio_policy_config(
            data_root=str(data_root),
            run_id=model_run,
            runtime_cfg={"market_mode": market_mode},
            require_artifact=True,
        ),
        sorted(accepted),
        strict=True,
    )
    validate_training_live_parity_contract(
        contract,
        active_strategy_ids=sorted(accepted),
        data_root=str(data_root),
        run_id=model_run,
        strict=True,
    )
    validate_meta_feature_contract_artifact(
        str(data_root),
        model_run,
        full_state,
        accepted,
        strict=True,
    )
    validate_deployment_model_coverage(model_bundle, accepted, strict=True)

    manifest = {
        "schema_version": "fullscope_policy_promotion_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": Path(__file__).name,
        "model_run_id": model_run,
        "policy_source_run_id": policy_run,
        "market_mode": market_mode,
        "feature_source_run_id": feature_source_run,
        "selected_strategy_ids": strategy_ids,
        "copied_artifacts": copied,
        "removed_legacy_adapter_artifacts": removed_legacy_adapters,
        "policy_params_source": (
            f"data_perp/artifacts/{model_run}/simple_policy_optimiser/"
            "deployment/best_policy_params.json"
        ),
        "stop_params_source": "simple_policy_optimiser",
        "parity_contract_paths": [str(path) for path in written],
        "validation": {
            "live_feature_contract": True,
            "portfolio_strategy_contract": True,
            "training_live_parity_contract": True,
            "meta_feature_contract": True,
            "deployment_model_coverage": True,
        },
    }
    _write_json(model_root / "policy_promotion_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--policy-run", default=DEFAULT_POLICY_RUN)
    parser.add_argument("--model-run", default=DEFAULT_MODEL_RUN)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument(
        "--feature-source-run",
        default=None,
        help=(
            "Explicit feature-source run. When omitted, inherit the policy "
            "contract source and then fall back to the repository default."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = promote(
        data_root=Path(args.data_root),
        policy_run=str(args.policy_run),
        model_run=str(args.model_run),
        market_mode=str(args.market_mode),
        feature_source_run=(
            str(args.feature_source_run) if args.feature_source_run else None
        ),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
