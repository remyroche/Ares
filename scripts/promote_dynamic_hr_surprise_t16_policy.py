#!/usr/bin/env python3
"""Promote the validated T16 dynamic HR-surprise threshold policy artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from extreme_price_movements.inference.dynamic_hr_surprise_threshold import (
    T16_POLICY_NAME,
    patch_portfolio_policy_payload_with_dynamic_hr_surprise,
    write_dynamic_hr_surprise_state_from_replay,
)
from extreme_price_movements.inference.symbol_mapping import normalise_symbol
from scripts.materialize_prehead_symbol_guard_ablation_candidates import _read_candidates


DEFAULT_REPLAY_DIR = Path(
    "data_perp/reports/prehead_symbol_guard_threshold_sweep_rel_disp_breadth10_20260630/"
    "A1_l4of5_24h/T16_recomputed_calendar_replay"
)
DEFAULT_PREHEAD_MATERIALIZED_DIR = Path(
    "data_perp/reports/prehead_symbol_guard_threshold_sweep_rel_disp_breadth10_20260630/"
    "A1_l4of5_24h/materialized/A1_loss_cooldown_3of4_24h"
)
DEFAULT_POLICY_RUN_ID = "20260629_050000_lgbm_mda"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _with_perps_sibling(path: Path) -> list[Path]:
    perps = path.with_name(f"{path.stem}_perps{path.suffix}")
    return [path, perps] if perps != path else [path]


def _head_side_blocked_map(blocked_rows: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for row in blocked_rows.to_dict("records"):
        head = str(row.get("head") or "")
        side = str(row.get("side") or "").lower()
        symbol = normalise_symbol(str(row.get("symbol") or ""))
        if not head or side not in {"long", "short"} or not symbol:
            continue
        out.setdefault(head, {}).setdefault(side, set()).add(symbol)
    return {
        head: {side: sorted(symbols) for side, symbols in sorted(side_map.items())}
        for head, side_map in sorted(out.items())
    }


def write_prehead_symbol_guard_state(
    *,
    materialized_dir: Path,
    original_candidates: Path,
    output_path: Path,
    policy_name: str,
    max_state_age_days: float,
) -> dict[str, Any]:
    decisions_path = materialized_dir / "prehead_symbol_guard_decisions.parquet"
    if not decisions_path.exists():
        raise FileNotFoundError(decisions_path)
    decisions = pd.read_parquet(decisions_path)
    original = _read_candidates(original_candidates, return_col="net_return")
    if len(decisions) != len(original):
        raise ValueError(
            "prehead decisions and original candidates have different lengths: "
            f"{len(decisions)} != {len(original)}"
        )
    joined = pd.concat(
        [
            original[["timestamp", "head", "symbol", "side"]].reset_index(drop=True),
            decisions.reset_index(drop=True),
        ],
        axis=1,
    )
    joined["timestamp"] = pd.to_datetime(joined["timestamp"], utc=True, errors="coerce")
    joined = joined.dropna(subset=["timestamp"]).copy()
    joined["day"] = joined["timestamp"].dt.floor("D")
    latest_day = joined["day"].max()
    latest = joined.loc[joined["day"].eq(latest_day)].copy()
    blocked = latest.loc[latest["prehead_symbol_guard_blocked"].astype(bool)].copy()
    summary_path = materialized_dir.parent / "prehead_symbol_guard_ablation_summary.parquet"
    summary = {}
    if summary_path.exists():
        summary = pd.read_parquet(summary_path).iloc[0].to_dict()
    payload = {
        "schema_version": "prehead_symbol_guard_state_v1",
        "policy_name": policy_name,
        "as_of": pd.Timestamp(latest_day).isoformat(),
        "blocked": _head_side_blocked_map(blocked),
        "source_materialized_dir": str(materialized_dir),
        "source_original_candidates": str(original_candidates),
        "max_state_age_days": float(max_state_age_days),
        "guard_config": {
            "variant": "A1_l4of5_24h",
            "mode": "loss_cooldown",
            "scope": "head_symbol_side",
            "loss_window": 5,
            "loss_threshold": 4,
            "cooldown_hours": 24.0,
            "require_relative_symbol_weakness": True,
            "relative_peer_min_symbols": 20,
            "relative_loss_peer_quantile": 0.75,
            "relative_loss_margin": 1.0,
            "max_blacklisted_asset_fraction": 0.10,
        },
        "latest_day_diagnostics": {
            "latest_day_rows": int(len(latest)),
            "latest_day_blocked_rows": int(len(blocked)),
            "latest_day_blocked_symbols": int(blocked["symbol"].nunique()) if not blocked.empty else 0,
            "latest_day_heads": sorted(blocked["head"].astype(str).unique().tolist()) if not blocked.empty else [],
        },
        "materialization_summary": summary,
    }
    _write_json(output_path, payload)
    return payload


def promote(args: argparse.Namespace) -> dict[str, Any]:
    replay_dir = Path(args.replay_dir)
    prehead_materialized_dir = Path(args.prehead_materialized_dir)
    data_root = Path(args.data_root)
    artifact_root = data_root / "artifacts" / str(args.policy_run_id)
    policy_params_dir = artifact_root / "policy_params"
    state_path = policy_params_dir / args.state_name
    prehead_state_path = policy_params_dir / args.prehead_state_name

    state_payload = write_dynamic_hr_surprise_state_from_replay(
        replay_dir,
        state_path,
        policy_name=T16_POLICY_NAME,
    )
    gate = state_payload.get("promotion_gate") or {}
    accepted = bool(gate.get("accepted"))
    if not accepted and not bool(args.allow_degrading):
        raise RuntimeError(
            "Refusing to enable dynamic HR surprise policy because promotion gate "
            f"failed: {gate}"
        )

    base_config_paths = [
        policy_params_dir / "optimized_portfolio_policy_config.json",
        policy_params_dir / "portfolio_policy_config.json",
        artifact_root / "portfolio_policy_replay" / "optimized_portfolio_policy_config.json",
    ]
    config_paths = [candidate for path in base_config_paths for candidate in _with_perps_sibling(path)]
    patched_paths: list[str] = []
    for config_path in config_paths:
        payload = _load_json(config_path)
        if not payload:
            continue
        patched = patch_portfolio_policy_payload_with_dynamic_hr_surprise(
            payload,
            artifact_path=str(state_path),
            enabled=accepted,
            max_state_age_days=float(args.max_state_age_days),
            use_deployed_floor=False,
            fallback_to_deployed=False,
            stale_fallback_to_deployed=True,
            lower_bound=-0.50,
            upper_bound=1.50,
        )
        selection = dict(patched.get("selection") or {})
        prehead_updates = {
            "prehead_symbol_guard_enabled": True,
            "prehead_symbol_guard_artifact_path": str(prehead_state_path),
            "prehead_symbol_guard_max_state_age_days": float(args.prehead_max_state_age_days),
        }
        selection.update(prehead_updates)
        patched["selection"] = selection
        patched.update(prehead_updates)
        _write_json(config_path, patched)
        patched_paths.append(str(config_path))

    prehead_payload = write_prehead_symbol_guard_state(
        materialized_dir=prehead_materialized_dir,
        original_candidates=Path(args.prehead_original_candidates),
        output_path=prehead_state_path,
        policy_name="A1_l4of5_24h",
        max_state_age_days=float(args.prehead_max_state_age_days),
    )

    manifest = {
        "policy_name": f"A1_l4of5_24h -> {T16_POLICY_NAME}",
        "status": "enabled" if accepted else "disabled_gate_failed",
        "policy_run_id": str(args.policy_run_id),
        "data_root": str(data_root),
        "replay_dir": str(replay_dir),
        "state_path": str(state_path),
        "prehead_materialized_dir": str(prehead_materialized_dir),
        "prehead_state_path": str(prehead_state_path),
        "patched_config_paths": patched_paths,
        "max_state_age_days": float(args.max_state_age_days),
        "prehead_max_state_age_days": float(args.prehead_max_state_age_days),
        "prehead_latest_day_diagnostics": prehead_payload.get("latest_day_diagnostics", {}),
        "promotion_gate": gate,
        "live_contract": {
            "prehead_symbol_guard": "A1_l4of5_24h relative symbol guard before T16 thresholds",
            "use_deployed_floor": False,
            "fallback_to_deployed_for_rejected_heads": False,
            "stale_or_missing_state_fallback": "deployed_threshold",
            "lower_bound": -0.50,
            "upper_bound": 1.50,
        },
    }
    manifest_path = policy_params_dir / "dynamic_hr_surprise_t16_promotion_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--prehead-materialized-dir", default=str(DEFAULT_PREHEAD_MATERIALIZED_DIR))
    parser.add_argument(
        "--prehead-original-candidates",
        default=(
            "data_perp/artifacts/finalfit_candidate_mask_native_candidates_20260627_6mo/"
            "simple_policy_optimiser/simple_policy_candidates_broad.parquet"
        ),
    )
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--policy-run-id", default=DEFAULT_POLICY_RUN_ID)
    parser.add_argument("--state-name", default="dynamic_hr_surprise_t16_state.json")
    parser.add_argument("--prehead-state-name", default="prehead_symbol_guard_a1_l4of5_state.json")
    parser.add_argument("--max-state-age-days", type=float, default=7.0)
    parser.add_argument("--prehead-max-state-age-days", type=float, default=7.0)
    parser.add_argument(
        "--allow-degrading",
        action="store_true",
        help="Patch configs disabled instead of raising when the non-degradation gate fails.",
    )
    args = parser.parse_args()
    manifest = promote(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
