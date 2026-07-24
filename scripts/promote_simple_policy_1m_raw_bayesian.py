#!/usr/bin/env python3
"""Refit and embed the 1m joint-trailing/raw-Bayesian production policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from extreme_price_movements.simple_policy_winner import (  # noqa: E402
    WINNER_BASE_SIZE_POWER,
    WINNER_FORWARD_BARS,
    WINNER_OOD_WEIGHT_GRID,
    WINNER_POLICY_PATHWAY_ID,
    WINNER_SIZE_STRENGTH_GRID,
    apply_raw_bayesian_sizing_state,
    fit_raw_bayesian_sizing_state,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    ExperimentData,
    _optimise,
)
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    _load_atr,
    _load_context,
    _weighted_evaluate,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sizing_rows(rows: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    for column in context.columns:
        result[column] = context[column].to_numpy(copy=False)
    result["side"] = np.where(
        pd.to_numeric(result["side"], errors="coerce").fillna(1.0) > 0.0,
        "long",
        "short",
    )
    return result


def _fit_sizing_state(
    data: ExperimentData,
    rows: pd.DataFrame,
    geometry: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    timestamps = pd.to_datetime(rows["timestamp"], utc=True)
    split_at = timestamps.sort_values().iloc[int(0.75 * (len(rows) - 1))]
    purge = pd.Timedelta(minutes=WINNER_FORWARD_BARS)
    fit_idx = np.flatnonzero((timestamps < split_at - purge).to_numpy() & data.valid)
    inner_idx = np.flatnonzero((timestamps >= split_at).to_numpy() & data.valid)
    if len(fit_idx) < 100 or len(inner_idx) < 100:
        raise RuntimeError("insufficient chronological support for sizing selection")
    fit_outputs = data.simulate(fit_idx, geometry, FAMILY_TRAILING_ONLY)
    inner_outputs = data.simulate(inner_idx, geometry, FAMILY_TRAILING_ONLY)
    grid: list[dict[str, float]] = []
    for strength in WINNER_SIZE_STRENGTH_GRID:
        for ood_weight in WINNER_OOD_WEIGHT_GRID:
            state = fit_raw_bayesian_sizing_state(
                rows.iloc[fit_idx],
                np.asarray(fit_outputs["net_return"], dtype=np.float64),
                strength=float(strength),
                ood_weight=float(ood_weight),
            )
            multipliers = np.ones(len(rows), dtype=np.float64)
            multipliers[inner_idx] = apply_raw_bayesian_sizing_state(
                rows.iloc[inner_idx], state
            )
            metrics = _weighted_evaluate(data, inner_idx, inner_outputs, multipliers)
            grid.append(
                {
                    "strength": float(strength),
                    "ood_weight": float(ood_weight),
                    "objective": float(metrics["objective"]),
                    "net_pnl_bankroll": float(metrics["net_pnl_bankroll"]),
                    "mean_net_return": float(metrics["mean_net_return"]),
                }
            )
    winner = max(grid, key=lambda item: (item["objective"], item["net_pnl_bankroll"]))
    full_idx = np.flatnonzero(data.valid)
    full_outputs = data.simulate(full_idx, geometry, FAMILY_TRAILING_ONLY)
    final_state = fit_raw_bayesian_sizing_state(
        rows.iloc[full_idx],
        np.asarray(full_outputs["net_return"], dtype=np.float64),
        strength=winner["strength"],
        ood_weight=winner["ood_weight"],
    )
    final_state["selection"] = {
        "method": "purged_chronological_inner_grid",
        "split_at_utc": split_at.isoformat(),
        "purge_minutes": WINNER_FORWARD_BARS,
        "grid": grid,
    }
    return final_state, {"winner": winner, "grid": grid}


def _runtime_side_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sl_mult": float(params["sl_mult"]),
        "trailing_activation_mult": float(params["trailing_activation_mult"]),
        "trailing_activation_cap_pct": float(params.get("trailing_activation_cap_pct", 0.0)),
        "trailing_activation_decay_half_life_bars": int(round(float(params.get("trailing_activation_decay_half_life_minutes", 0.0)))),
        "trailing_activation_decay_start_bars": int(round(float(params.get("trailing_activation_decay_start_minutes", 0.0)))),
        "trailing_activation_min_mult": float(params.get("trailing_activation_min_mult", 1.0)),
        "trailing_power": float(params["trailing_power"]),
        "trailing_squash_divisor": float(params["trailing_squash_divisor"]),
        "giveback_beta": float(params["giveback_beta"]),
        "adverse_exit_enabled": bool(params.get("adverse_exit_enabled", False)),
        "adverse_exit_alpha": 1.0,
        "adverse_exit_beta": 1.0,
        "adverse_exit_delta": 1.0,
        "adverse_exit_theta_quantile": 0.75,
        "adverse_exit_theta": float(params.get("adverse_exit_theta", 1.0e9)),
        "adverse_exit_fast_bars": int(round(float(params.get("adverse_exit_fast_minutes", 60.0)))),
        "adverse_exit_min_mae_atr": float(params.get("adverse_exit_min_mae_atr", 1.4)),
        "adverse_exit_min_speed": float(params.get("adverse_exit_min_speed_per_15m", 0.3)),
        "adverse_exit_max_mfe_atr": float(params.get("adverse_exit_max_mfe_atr", 0.25)),
    }


def _update_policy(
    payload: dict[str, Any],
    *,
    geometry: Mapping[str, Mapping[str, Any]],
    sizing_state: Mapping[str, Any],
    source_report: Path,
) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    policy_name = "s52_v9_tail95_mlp_hierev_ev70_jointtrailing1m_rawbayes_v1"
    result["policy_name"] = policy_name
    result["policy_pathway_id"] = WINNER_POLICY_PATHWAY_ID
    result["sizing_policy_id"] = "raw_bayesian_v1"
    result["replay_timeframe"] = "1m"
    result["forward_bars"] = WINNER_FORWARD_BARS
    result["exit_geometry_contract"] = {
        "policy_pathway_id": WINNER_POLICY_PATHWAY_ID,
        "family": "joint_trailing_only",
        "replay_timeframe": "1m",
        "horizon_minutes": WINNER_FORWARD_BARS,
        "trailing_activation_curve": "total_mfe",
        "capital_preservation_enabled": False,
        "atr_power": 1.0,
        "atr_multiplier": 1.0,
        "source_report": str(source_report),
    }
    for row in result.get("strategies", []):
        side = str(row.get("side") or "").lower()
        if side not in geometry:
            raise ValueError(f"strategy row has no fitted side geometry: {side!r}")
        # The 1m winner uses each candidate's causal entry ATR. Historical
        # deployment rows carried fixed archetype barriers that would override
        # the live entry-time ATR and break replay parity.
        row.pop("barrier_frac", None)
        row.pop("barrier_pct", None)
        row.update(_runtime_side_params(geometry[side]))
        row.update(
            {
                "policy_name": policy_name,
                "policy_pathway_id": WINNER_POLICY_PATHWAY_ID,
                "replay_timeframe": "1m",
                "forward_bars": WINNER_FORWARD_BARS,
                "trailing_activation_curve": "total_mfe",
                "capital_preservation_enabled": False,
                "sizing_policy_id": "raw_bayesian_v1",
                "raw_bayesian_sizing_state": _json_safe(sizing_state),
                "size_power": WINNER_BASE_SIZE_POWER,
                "enable_trailing": True,
                "sl_abs_cap_pct": 0.0,
                "atr_power": 1.0,
                "atr_multiplier": 1.0,
                "hard_tp_abs_pct": 0.0,
                "exit_pressure_enabled": False,
                "capital_protect_mfe_mult": 0.0,
                "capital_protect_regression_frac": 0.0,
                "capital_protect_lock_frac": 0.0,
                "capital_protect_min_lock_bps": 0.0,
                "capital_protect_spread_lock_mult": 0.0,
            }
        )
    result["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rich-ledger", type=Path, required=True)
    parser.add_argument("--posterior-state", type=Path, required=True)
    parser.add_argument("--deployed-parent-summary", type=Path, required=True)
    parser.add_argument("--path-cache-dir", type=Path, required=True)
    parser.add_argument("--atr-audit", type=Path, required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--output-policy", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--trials-per-seed", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows = pd.read_parquet(args.candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, provenance = _load_context(rows, args.rich_ledger, args.posterior_state)
    sizing_rows = _sizing_rows(rows, context)
    atr = _load_atr(rows, args.atr_audit)
    deployed, _ = _load_deployed_side_params(args.deployed_parent_summary)
    spec = ConstrainedReplaySpec()
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows,
        store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"),
        cache_dir=args.path_cache_dir,
        spec=spec,
        rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)
    valid_idx = np.flatnonzero(data.valid)
    seeds = [args.seed + 1000 * i for i in range(int(args.seeds))]
    geometry, geometry_hpo = _optimise(
        data,
        valid_idx,
        family=FAMILY_TRAILING_ONLY,
        joint=True,
        trials_per_seed=int(args.trials_per_seed),
        seeds=seeds,
        sampler_kind="tpe",
    )
    sizing_state, sizing_hpo = _fit_sizing_state(data, sizing_rows, geometry)
    source_payload = json.loads(args.source_policy.read_text(encoding="utf-8"))
    promoted = _update_policy(
        source_payload,
        geometry=geometry,
        sizing_state=sizing_state,
        source_report=args.source_report,
    )
    args.output_policy.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(promoted), indent=2, sort_keys=True) + "\n"
    tmp = args.output_policy.with_name(f".{args.output_policy.name}.tmp.{os.getpid()}")
    tmp.write_text(encoded, encoding="utf-8")
    os.replace(tmp, args.output_policy)
    manifest = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_pathway_id": WINNER_POLICY_PATHWAY_ID,
        "source_policy": str(args.source_policy),
        "output_policy": str(args.output_policy),
        "output_sha256": hashlib.sha256(args.output_policy.read_bytes()).hexdigest(),
        "candidate_rows": int(len(rows)),
        "valid_path_rows": int(data.valid.sum()),
        "fit_start_utc": rows["timestamp"].min().isoformat(),
        "fit_end_utc": rows["timestamp"].max().isoformat(),
        "geometry": geometry,
        "geometry_hpo": geometry_hpo,
        "sizing_hpo": sizing_hpo,
        "sizing_state_summary": {
            "fit_rows": sizing_state.get("fit_rows"),
            "strength": sizing_state.get("strength"),
            "ood_weight": sizing_state.get("ood_weight"),
            "cells": len(sizing_state.get("cells", [])),
            "train_normalizer": sizing_state.get("train_normalizer"),
        },
        "context_provenance": provenance,
        "path_manifest": path_manifest,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["sizing_state_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
