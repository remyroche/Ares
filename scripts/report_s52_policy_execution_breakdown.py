#!/usr/bin/env python3
"""Build execution replay breakdowns from an S52 side/archetype policy run."""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_simple_policy_exit_geometry import _load_bundles, _prepare_rows  # noqa: E402
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
    MAX_CONCURRENT_TRADES,
    _json_safe,
    _path_take,
    _without_concurrency_param,
    calculate_advanced_metrics,
    simulate_and_score,
)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(value)
        except Exception:
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"", "nan", "none", "null"}:
            return None
        if lower in {"true", "false"}:
            return lower == "true"
        try:
            return float(value)
        except Exception:
            return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parent_policy_maps(parent: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    params_by_strategy: dict[str, dict[str, Any]] = {}
    size_by_strategy: dict[str, float] = {}
    for _, row in parent.iterrows():
        strategy_id = str(row.get("strategy_id", ""))
        params: dict[str, Any] = {}
        for col, value in row.items():
            if not str(col).startswith("param_"):
                continue
            key = str(col).replace("param_", "", 1)
            value = _normalise_value(value)
            if value is not None:
                params[key] = value
        if strategy_id:
            params_by_strategy[strategy_id] = params
            size_by_strategy[strategy_id] = _safe_float(row.get("best_size_power"), 1.0)
    return params_by_strategy, size_by_strategy


def _archetype_policy_map(
    archetypes: pd.DataFrame,
) -> dict[tuple[str, str], tuple[dict[str, Any], float]]:
    out: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}
    if archetypes.empty:
        return out
    for _, row in archetypes.iterrows():
        strategy_id = str(row.get("strategy_id", ""))
        archetype = str(row.get("policy_archetype", ""))
        geometry = _parse_mapping(row.get("shrinkage_final_geometry"))
        if not strategy_id or not archetype or not geometry:
            continue
        size_power = _safe_float(geometry.pop("size_power", np.nan), np.nan)
        out[(strategy_id, archetype)] = (geometry, size_power)
    return out


def _period_values(rows: pd.DataFrame, period: str) -> pd.Series:
    ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    if period == "month":
        return ts.dt.strftime("%Y-%m")
    if period == "week":
        return ts.dt.to_period("W").astype(str)
    raise ValueError(f"Unsupported period {period!r}")


def _score_slice(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    params: Mapping[str, Any],
    size_power: float,
    cost_pct: float,
) -> dict[str, Any]:
    if rows.empty:
        return {}
    metrics = simulate_and_score(
        rows.copy(),
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        cost_pct=float(cost_pct),
        size_power=float(size_power),
        max_concurrent_trades=MAX_CONCURRENT_TRADES,
        max_concurrent_per_asset=DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
        **_without_concurrency_param(dict(params)),
    )
    adv = calculate_advanced_metrics(
        rows,
        metrics.get("raw_gains", np.array([])),
        metrics.get("sizes", np.array([])),
        metrics.get("selected_mask"),
        metrics.get("gross_gains"),
        metrics.get("exit_reason"),
        metrics.get("exit_bars"),
    )
    out: dict[str, Any] = {
        "candidate_rows": int(len(rows)),
        "candidate_symbols": int(rows["symbol"].astype(str).nunique()) if "symbol" in rows else 0,
        "sim_candidate_count": int(metrics.get("candidate_count", 0) or 0),
        "sim_valid_entry_count": int(metrics.get("valid_entry_count", 0) or 0),
        "sim_skipped_concurrency": int(metrics.get("skipped_concurrency", 0) or 0),
    }
    for key in (
        "n_trades",
        "avg_pnl_bankroll",
        "avg_pnl_sized",
        "avg_pnl_notional",
        "avg_gross_pnl_per_trade",
        "avg_gross_return_per_trade",
        "hit_rate",
        "pnl_positive_rate",
        "full_sl_exit_rate",
        "adverse_fast_exit_rate",
        "timeout_exit_rate",
        "trailing_profit_exit_rate",
        "capital_protect_exit_rate",
        "worst_week",
        "max_dd",
        "w_sortino",
        "m_sortino",
        "weekly_pnl_q10",
        "weekly_pnl_q50",
        "weekly_pnl_q90",
    ):
        if key in adv:
            out[key] = adv[key]
    return out


def _build_breakdown(
    *,
    bundles: list[Any],
    parent_params: Mapping[str, Mapping[str, Any]],
    parent_sizes: Mapping[str, float],
    archetype_policies: Mapping[tuple[str, str], tuple[Mapping[str, Any], float]],
    cost_pct: float,
    period: str,
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    for bundle in bundles:
        strategy_id = str(bundle.strategy_id)
        side = "short" if strategy_id.startswith("short") else "long"
        parent_policy = dict(parent_params.get(strategy_id) or bundle.base_params)
        parent_size = float(parent_sizes.get(strategy_id, bundle.best_size_power))
        work = bundle.rows.copy().reset_index(drop=True)
        work["_period"] = _period_values(work, period)
        if "policy_archetype" not in work.columns:
            work["policy_archetype"] = f"{side}__unknown"
        group_cols = ["_period", "policy_archetype"]
        for keys, group in work.groupby(group_cols, dropna=False, sort=True):
            period_value, archetype = keys
            idx = group.index.to_numpy(dtype=np.int64, copy=False)
            sub_paths = _path_take(bundle.paths, idx)
            key = (strategy_id, str(archetype))
            local_params = dict(parent_policy)
            local_size = parent_size
            policy_source = "side_parent"
            if key in archetype_policies:
                geometry, size_power = archetype_policies[key]
                local_params.update(dict(geometry))
                if np.isfinite(float(size_power)):
                    local_size = float(size_power)
                policy_source = "side_archetype_shrunk"
            metrics = _score_slice(
                group.drop(columns=["_period"]).reset_index(drop=True),
                sub_paths,
                params=local_params,
                size_power=local_size,
                cost_pct=cost_pct,
            )
            rows_out.append(
                {
                    "period_type": period,
                    "period": str(period_value),
                    "strategy_id": strategy_id,
                    "side": side,
                    "policy_archetype": str(archetype),
                    "policy_source": policy_source,
                    "size_power": float(local_size),
                    **metrics,
                }
            )
    return pd.DataFrame(rows_out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--market-mode", choices=["spot", "perps"], default="perps")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--min-rank", type=float, default=0.0)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.01)
    parser.add_argument("--start", default=None, help="Optional UTC inclusive start timestamp/date.")
    parser.add_argument("--end", default=None, help="Optional UTC exclusive end timestamp/date.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parent_path = args.policy_dir / "side_parent_policy_summary.csv"
    archetype_path = args.policy_dir / "side_archetype_policy_summary.csv"
    if not parent_path.exists():
        raise FileNotFoundError(parent_path)
    if not archetype_path.exists():
        raise FileNotFoundError(archetype_path)
    rows = _prepare_rows(
        args.candidates,
        min_rank=float(args.min_rank),
        apply_regime_ev_calibration_artifact=False,
    )
    if args.start or args.end:
        ts = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
        mask = pd.Series(True, index=rows.index)
        if args.start:
            mask &= ts.ge(pd.Timestamp(str(args.start), tz="UTC"))
        if args.end:
            mask &= ts.lt(pd.Timestamp(str(args.end), tz="UTC"))
        rows = rows.loc[mask].copy().reset_index(drop=True)
        if rows.empty:
            raise ValueError(
                f"No candidate rows left after date filter start={args.start!r} end={args.end!r}"
            )
    bundles = _load_bundles(
        rows,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        min_rows_per_strategy=5,
    )
    parent = pd.read_csv(parent_path)
    archetypes = pd.read_csv(archetype_path)
    parent_params, parent_sizes = _parent_policy_maps(parent)
    archetype_policies = _archetype_policy_map(archetypes)
    cost_pct = float(args.round_trip_cost_pct) / 2.0
    outputs: dict[str, str] = {}
    for period in ("month", "week"):
        frame = _build_breakdown(
            bundles=bundles,
            parent_params=parent_params,
            parent_sizes=parent_sizes,
            archetype_policies=archetype_policies,
            cost_pct=cost_pct,
            period=period,
        )
        path = args.out_dir / f"execution_{period}_side_archetype_metrics.csv"
        frame.to_csv(path, index=False)
        outputs[f"execution_{period}_side_archetype_metrics"] = str(path)
    manifest = {
        "generated_by": "report_s52_policy_execution_breakdown",
        "candidates": str(args.candidates),
        "policy_dir": str(args.policy_dir),
        "data_root": str(args.data_root),
        "market_mode": str(args.market_mode),
        "path_len": int(args.path_len),
        "min_rank": float(args.min_rank),
        "start": str(args.start) if args.start else None,
        "end": str(args.end) if args.end else None,
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "legacy_regime_ev_calibration_applied": False,
        "cost_pct_per_side": cost_pct,
        "outputs": outputs,
    }
    (args.out_dir / "execution_breakdown_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"event": "s52_policy_execution_breakdown_done", **manifest}), sort_keys=True))


if __name__ == "__main__":
    main()
