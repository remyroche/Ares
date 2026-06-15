#!/usr/bin/env python3
"""Build a combined live inference artifact from two deployment runs."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.inference.training_live_parity_contract import (
    build_training_live_parity_contract,
    persist_training_live_parity_contract,
)
from extreme_price_movements.model_loader import load_alpha_models, load_full_state


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _safe_unlink(dst)
    os.symlink(src.resolve(), dst)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _strategy_id(row: dict[str, Any]) -> str:
    return str(
        row.get("strategy_id")
        or row.get("strategy_for_inference")
        or row.get("canonical_strategy_id")
        or ""
    ).strip()


def _strategy_core(strategy_id: str) -> str:
    sid = str(strategy_id or "").strip()
    for prefix in ("long_", "short_"):
        if sid.startswith(prefix):
            return sid[len(prefix) :]
    return sid


def _merge_strategy_rows(*payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for row in payload.get("strategies") or []:
            if not isinstance(row, dict):
                continue
            sid = _strategy_id(row)
            if sid:
                rows[sid] = deepcopy(row)
    return [rows[sid] for sid in sorted(rows)]


def _merge_strategy_payload(
    *,
    output_run_id: str,
    current: dict[str, Any],
    addon: dict[str, Any],
    source_runs: list[str],
) -> dict[str, Any]:
    out = deepcopy(current or {})
    out["run_id"] = output_run_id
    out["generated_by"] = current.get("generated_by") or addon.get("generated_by")
    out["combined_generated_by"] = "build_combined_live_artifact"
    out["source_run_ids"] = list(source_runs)
    out["schema_version"] = out.get("schema_version") or addon.get("schema_version")
    out["market_mode"] = out.get("market_mode") or addon.get("market_mode") or "perps"
    out["strategies"] = _merge_strategy_rows(current, addon)
    for key in ("rejected_strategies",):
        merged = []
        for payload in (current, addon):
            values = payload.get(key)
            if isinstance(values, list):
                merged.extend(deepcopy(values))
        if merged:
            out[key] = merged
    for key in ("asset_exclusions",):
        merged: dict[str, Any] = {}
        for payload in (current, addon):
            value = payload.get(key)
            if isinstance(value, dict):
                merged.update(deepcopy(value))
        if merged:
            out[key] = merged
    return out


def _merge_best_policy_params(
    *,
    output_run_id: str,
    current: dict[str, Any],
    addon: dict[str, Any],
    source_runs: list[str],
    strategy_payload: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(current or {})
    out["run_id"] = output_run_id
    out["generated_by"] = current.get("generated_by") or addon.get("generated_by")
    out["combined_generated_by"] = "build_combined_live_artifact"
    out["source_run_ids"] = list(source_runs)
    out["schema_version"] = out.get("schema_version") or addon.get("schema_version")
    out["market_mode"] = out.get("market_mode") or addon.get("market_mode") or "perps"
    out["strategies"] = _merge_strategy_rows(current, addon)
    out["strategy_for_inference"] = deepcopy(strategy_payload)
    return out


def _merge_portfolio_policy(
    *,
    output_run_id: str,
    current: dict[str, Any],
    addon: dict[str, Any],
    strategy_ids: list[str],
    source_runs: list[str],
) -> dict[str, Any]:
    out = deepcopy(current or {})
    out["portfolio_policy_version"] = out.get("portfolio_policy_version") or addon.get(
        "portfolio_policy_version", "global_auction_v1"
    )
    out["generated_by"] = "build_combined_live_artifact"
    out["run_id"] = output_run_id
    out["source_run_ids"] = list(source_runs)
    out["strategy_contract"] = {
        "strategy_ids": sorted(strategy_ids),
        "strategy_cores": sorted({_strategy_core(sid) for sid in strategy_ids}),
    }
    return out


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _load_joblib(path: Path) -> Any:
    return joblib.load(path)


def _bundle(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    bundle = obj.get("bundle", obj)
    return bundle if isinstance(bundle, dict) else {}


def _merge_model_states(run_roots: list[Path], output_root: Path, output_run_id: str) -> None:
    trained_states: list[dict[str, Any]] = []
    meta_states: list[dict[str, Any]] = []
    for root in run_roots:
        trained_path = root / "models" / "trained_state.pkl"
        if trained_path.exists():
            state = _load_pickle(trained_path)
            if isinstance(state, dict):
                trained_states.append(state)
        meta_path = root / "models" / "model_state_meta.pkl"
        if meta_path.exists():
            state = _load_joblib(meta_path)
            if isinstance(state, dict):
                meta_states.append(state)

    if not trained_states:
        raise FileNotFoundError("No trained_state.pkl found in source runs")

    out_state = deepcopy(trained_states[0])
    out_bundle = deepcopy(_bundle(out_state))
    merged_meta: dict[str, Any] = {}
    merged_alpha: dict[str, Any] = {}
    for state in [*trained_states, *meta_states]:
        bundle = _bundle(state)
        alpha = bundle.get("alpha_models")
        if isinstance(alpha, dict):
            merged_alpha.update(alpha)
        meta = bundle.get("meta_models")
        if isinstance(meta, dict):
            merged_meta.update(meta)
    native_alpha = load_alpha_models(str(output_root / "models" / "native"))
    if native_alpha:
        merged_alpha = native_alpha
    if merged_alpha:
        out_bundle["alpha_models"] = merged_alpha
    if merged_meta:
        out_bundle["meta_models"] = merged_meta
    out_state["bundle"] = out_bundle
    out_state["run_id"] = output_run_id
    out_state["source_run_ids"] = [root.name for root in run_roots]

    models_dir = output_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / "trained_state.pkl").open("wb") as fh:
        pickle.dump(out_state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    meta_state = {
        "ts_trained": out_state.get("ts_trained"),
        "run_id": output_run_id,
        "source_run_ids": [root.name for root in run_roots],
        "bundle": {"meta_models": merged_meta},
    }
    joblib.dump(meta_state, models_dir / "model_state_meta.pkl")


def _merge_meta_feature_contract(run_roots: list[Path], output_root: Path, output_run_id: str) -> None:
    out: dict[str, Any] = {}
    meta_models: dict[str, Any] = {}
    missing: list[Any] = []
    for root in run_roots:
        payload = _load_json(root / "meta_oof" / "meta_feature_contract.json")
        if not out:
            out = deepcopy(payload)
        models = payload.get("meta_models")
        if isinstance(models, dict):
            meta_models.update(deepcopy(models))
        values = payload.get("missing_tbm_clf_contracts")
        if isinstance(values, list):
            missing.extend(deepcopy(values))
    out["run_id"] = output_run_id
    out["generated_by"] = "build_combined_live_artifact"
    out["source_run_ids"] = [root.name for root in run_roots]
    out["meta_models"] = meta_models
    out["missing_tbm_clf_contracts"] = missing
    _write_json(output_root / "meta_oof" / "meta_feature_contract.json", out)


def _concat_parquets(sources: list[Path], dst: Path) -> None:
    frames = []
    for path in sources:
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).to_parquet(dst, index=False)


def _merge_rank_reference(run_roots: list[Path], output_root: Path, output_run_id: str) -> None:
    rank_root = output_root / "simple_policy_optimiser" / "rank_reference"
    rank_root.mkdir(parents=True, exist_ok=True)
    manifests = [
        _load_json(root / "simple_policy_optimiser" / "rank_reference" / "manifest.json")
        for root in run_roots
    ]
    strategies: dict[str, Any] = {}
    contracts: list[dict[str, Any]] = []
    for root, manifest in zip(run_roots, manifests):
        root_rank = root / "simple_policy_optimiser" / "rank_reference"
        for alias, entry in (manifest.get("strategies") or {}).items():
            if not isinstance(entry, dict):
                continue
            entry_out = deepcopy(entry)
            rel = str(entry_out.get("path") or "")
            src = root / rel if rel else root_rank / f"{alias}.parquet"
            if src.exists():
                dst = rank_root / src.name
                _symlink(src, dst)
                entry_out["path"] = f"simple_policy_optimiser/rank_reference/{dst.name}"
            strategies[str(alias)] = entry_out
            contract = entry_out.get("policy_oos_contract")
            if isinstance(contract, dict) and contract:
                contracts.append(deepcopy(contract))

    auction_sources = [
        root / "simple_policy_optimiser" / "rank_reference" / "cross_strategy_auction.parquet"
        for root in run_roots
    ]
    auction_dst = rank_root / "cross_strategy_auction.parquet"
    _concat_parquets(auction_sources, auction_dst)
    auction_df = pd.read_parquet(auction_dst) if auction_dst.exists() else pd.DataFrame()
    score_col = "calibrated_score"
    auction = {
        "path": "simple_policy_optimiser/rank_reference/cross_strategy_auction.parquet",
        "score_col": score_col,
        "rank_col": "normalized_rank_score",
        "n_rows": int(len(auction_df)),
        "min_score": float(pd.to_numeric(auction_df.get(score_col), errors="coerce").min())
        if score_col in auction_df
        else None,
        "max_score": float(pd.to_numeric(auction_df.get(score_col), errors="coerce").max())
        if score_col in auction_df
        else None,
    }
    if contracts and all(contract == contracts[0] for contract in contracts[1:]):
        policy_contract = contracts[0]
    else:
        policy_contract = {
            "schema_version": "policy_rank_reference_policy_oos_contract_v1",
            "policy_oos_generation_source": "generated_from_train_meta_state:mixed",
            "policy_oos_source_model_fit_end": "mixed",
            "rank_normalization": "policy_rank_reference_percentile_from_policy_oos_clf",
            "strategy_contract_count": len(contracts),
        }
    manifest = deepcopy(manifests[0] if manifests else {})
    manifest.update(
        {
            "run_id": output_run_id,
            "generated_by": "build_combined_live_artifact",
            "source_run_ids": [root.name for root in run_roots],
            "schema_version": manifest.get("schema_version") or "policy_rank_reference_v1",
            "market_mode": "perps",
            "strategies": strategies,
            "auction": auction,
            "policy_oos_contract": policy_contract,
        }
    )
    _write_json(rank_root / "manifest.json", manifest)


def _merge_policy_candidates(run_roots: list[Path], output_root: Path) -> None:
    for name in ("simple_policy_candidates.parquet", "simple_policy_candidates_deployable.parquet"):
        _concat_parquets(
            [root / "simple_policy_optimiser" / name for root in run_roots],
            output_root / "simple_policy_optimiser" / name,
        )


def _link_native_models(run_roots: list[Path], output_root: Path) -> None:
    def _normalise_native_name(name: str) -> str:
        for side in ("long", "short"):
            doubled = f"{side}_{side}_"
            if name.startswith(doubled):
                return f"{side}_{name[len(doubled):]}"
        return name

    native_out = output_root / "models" / "native"
    native_out.mkdir(parents=True, exist_ok=True)
    for root in run_roots:
        native = root / "models" / "native"
        for child in sorted(native.iterdir()) if native.exists() else []:
            if child.is_dir():
                _symlink(child, native_out / _normalise_native_name(child.name))


def _link_oof_files(run_roots: list[Path], output_root: Path) -> None:
    for subdir in ("oof", "meta_oof", "policy_oos_predictions", "row_universe"):
        for root in run_roots:
            src_dir = root / subdir
            if not src_dir.exists():
                continue
            for child in sorted(src_dir.iterdir()):
                if child.name == "meta_feature_contract.json":
                    continue
                if child.is_file():
                    _symlink(child, output_root / subdir / child.name)


def _copy_manifests(run_roots: list[Path], output_root: Path) -> None:
    primary = run_roots[0]
    for rel in (
        "base_meta_contract.json",
        "base_models_intermediate.pkl",
        "base_models_intermediate.manifest.json",
        "models/training_exchange_contract.json",
    ):
        _copy_file(primary / rel, output_root / rel)


def _source_feature_run_id(root: Path, source_run: str) -> str:
    for rel in (
        "policy_params/training_live_parity_contract.json",
        "simple_policy_optimiser/training_live_parity_contract.json",
    ):
        payload = _load_json(root / rel)
        feature_source = payload.get("feature_source")
        if isinstance(feature_source, dict):
            run_id = str(feature_source.get("run_id") or "").strip()
            if run_id:
                return run_id
    return str(source_run)


def _write_training_live_parity_contract(
    *,
    data_root: Path,
    output_root: Path,
    output_run: str,
    run_roots: list[Path],
    source_runs: list[str],
    strategy_ids: list[str],
    strategy_payload: dict[str, Any],
    portfolio_payload: dict[str, Any],
) -> None:
    full_state = load_full_state(output_run, str(data_root))
    model_bundle = full_state.get("bundle", full_state) if isinstance(full_state, dict) else {}
    orchestrator = ModelOrchestrator(model_bundle, full_state)
    contract = build_training_live_parity_contract(
        data_root=str(data_root),
        run_id=output_run,
        market_mode="perps",
        orchestrator=orchestrator,
        model_bundle=full_state,
        strategy_ids=strategy_ids,
        deployment_payload=strategy_payload,
        portfolio_payload=portfolio_payload,
    )
    feature_sources: list[dict[str, str]] = []
    for root, source_run in zip(run_roots, source_runs):
        feature_sources.append(
            {
                "run_id": _source_feature_run_id(root, source_run),
                "source_run_id": str(source_run),
                "data_root": str(data_root),
            }
        )
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in feature_sources:
        run_id = str(source.get("run_id") or "").strip()
        if not run_id or run_id in seen:
            continue
        seen.add(run_id)
        deduped.append(source)
    if deduped:
        contract["feature_sources"] = deduped
        contract["feature_source"] = {
            "run_id": deduped[0]["run_id"],
            "data_root": str(data_root),
        }
    persist_training_live_parity_contract(
        contract,
        data_root=str(data_root),
        run_id=output_run,
    )


def build(data_root: Path, current_run: str, addon_run: str, output_run: str, overwrite: bool) -> None:
    run_roots = [data_root / "artifacts" / current_run, data_root / "artifacts" / addon_run]
    for root in run_roots:
        if not root.exists():
            raise FileNotFoundError(root)
    output_root = data_root / "artifacts" / output_run
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    current_strategy = _load_json(run_roots[0] / "policy_params" / "strategy_for_inference.json")
    addon_strategy = _load_json(run_roots[1] / "policy_params" / "strategy_for_inference.json")
    strategy_payload = _merge_strategy_payload(
        output_run_id=output_run,
        current=current_strategy,
        addon=addon_strategy,
        source_runs=[current_run, addon_run],
    )
    strategy_ids = [_strategy_id(row) for row in strategy_payload.get("strategies") or []]
    strategy_ids = sorted({sid for sid in strategy_ids if sid})

    current_best = _load_json(run_roots[0] / "policy_params" / "best_policy_params.json")
    addon_best = _load_json(run_roots[1] / "policy_params" / "best_policy_params.json")
    best_payload = _merge_best_policy_params(
        output_run_id=output_run,
        current=current_best,
        addon=addon_best,
        source_runs=[current_run, addon_run],
        strategy_payload=strategy_payload,
    )
    current_portfolio = _load_json(
        run_roots[0] / "policy_params" / "optimized_portfolio_policy_config.json"
    )
    addon_portfolio = _load_json(
        run_roots[1] / "policy_params" / "optimized_portfolio_policy_config.json"
    )
    portfolio_payload = _merge_portfolio_policy(
        output_run_id=output_run,
        current=current_portfolio,
        addon=addon_portfolio,
        strategy_ids=strategy_ids,
        source_runs=[current_run, addon_run],
    )

    for rel in (
        "policy_params/strategy_for_inference.json",
        "strategy_for_inference.json",
        "policy_params/strategy_for_inference_perps.json",
        "strategy_for_inference_perps.json",
    ):
        _write_json(output_root / rel, strategy_payload)
    for rel in (
        "policy_params/best_policy_params.json",
        "policy_params/best_policy_params_perps.json",
        "best_policy_params.json",
        "best_policy_params_perps.json",
        "simple_policy_optimiser/deployment/best_policy_params.json",
        "simple_policy_optimiser/deployment/best_policy_params_perps.json",
    ):
        _write_json(output_root / rel, best_payload)
    for rel in (
        "policy_params/optimized_portfolio_policy_config.json",
        "policy_params/portfolio_policy_config.json",
        "policy_params/portfolio_policy_config_perps.json",
        "portfolio_policy_replay/optimized_portfolio_policy_config.json",
    ):
        _write_json(output_root / rel, portfolio_payload)

    _copy_manifests(run_roots, output_root)
    _link_native_models(run_roots, output_root)
    _link_oof_files(run_roots, output_root)
    _merge_model_states(run_roots, output_root, output_run)
    _merge_meta_feature_contract(run_roots, output_root, output_run)
    _merge_rank_reference(run_roots, output_root, output_run)
    _merge_policy_candidates(run_roots, output_root)
    _write_training_live_parity_contract(
        data_root=data_root,
        output_root=output_root,
        output_run=output_run,
        run_roots=run_roots,
        source_runs=[current_run, addon_run],
        strategy_ids=strategy_ids,
        strategy_payload=strategy_payload,
        portfolio_payload=portfolio_payload,
    )

    provenance = {
        "run_id": output_run,
        "generated_by": "build_combined_live_artifact",
        "source_run_ids": [current_run, addon_run],
        "feature_source_run_ids": [
            _source_feature_run_id(root, source_run)
            for root, source_run in zip(run_roots, [current_run, addon_run])
        ],
        "strategy_count": len(strategy_ids),
        "strategy_ids": strategy_ids,
        "notes": [
            "Portfolio allocation and concurrency settings copied from the first/current run.",
            "Per-strategy policy thresholds, stop parameters, and rank references are merged from both runs.",
            "On model/meta key collisions, later source runs win.",
        ],
    }
    _write_json(output_root / "combined_live_artifact_manifest.json", provenance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--current-run", required=True)
    parser.add_argument("--addon-run", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(
        data_root=Path(args.data_root),
        current_run=args.current_run,
        addon_run=args.addon_run,
        output_run=args.output_run,
        overwrite=bool(args.overwrite),
    )
    print(f"combined artifact written: {Path(args.data_root) / 'artifacts' / args.output_run}")


if __name__ == "__main__":
    main()
