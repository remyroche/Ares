#!/usr/bin/env python3
"""Compare side-parent versus side x archetype exit geometry in portfolio replay.

The current live deployment can carry archetype-aware thresholds/rank context
while still using side-parent TP/SL/trailing geometry. This script materializes
matched candidate ledgers with recomputed execution-path returns under:

1. deployed side-parent exit geometry;
2. side x archetype shrinkage geometry, using the parent EV curve;
3. side x archetype shrinkage geometry, using a self-consistent EV curve.

The candidate universe, thresholds, rank columns, portfolio policy, and costs are
kept fixed unless explicitly changed by CLI arguments.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablate_simple_policy_exit_geometry import (  # noqa: E402
    _load_bundles,
    _path_take,
    _prepare_rows,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    compute_replay_metrics,
    fit_hierarchical_ev_curves,
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _json_safe,
    _without_concurrency_param,
    simulate_and_score,
)


DEFAULT_RUN_ID = "s59_s52_frozen_native_shadow_20260709"
DEFAULT_ARTIFACT_ROOT = Path("data_perp/artifacts") / DEFAULT_RUN_ID
DEFAULT_CANDIDATES = (
    DEFAULT_ARTIFACT_ROOT
    / "simple_policy_optimiser"
    / "simple_policy_candidates_deployable.parquet"
)
DEFAULT_POLICY_PARAMS = (
    DEFAULT_ARTIFACT_ROOT
    / "simple_policy_optimiser"
    / "deployment"
    / "best_policy_params_perps.json"
)
DEFAULT_PORTFOLIO_POLICY = (
    DEFAULT_ARTIFACT_ROOT / "policy_params" / "optimized_portfolio_policy_config.json"
)
DEFAULT_ARCHETYPE_POLICY = Path(
    "data_perp/reports/"
    "s59_h5_2025start_monthly_v4_base_configfull_mdafs120_hpo150_largestfold_oos15_ae3000_nocrossfit_k34567_payload300k_20260707_v4_trials96_juneh2_holdout_archfix/"
    "side_archetype_policy_summary.csv"
)
DEFAULT_OUT_DIR = Path("data_perp/reports/archetype_exit_geometry_replay_compare_20260710")

EXIT_PARAM_KEYS = {
    "sl_mult",
    "sl_abs_cap_pct",
    "trailing_activation_mult",
    "trailing_activation_cap_pct",
    "trailing_activation_decay_half_life_bars",
    "trailing_activation_decay_start_bars",
    "trailing_activation_min_mult",
    "trailing_power",
    "trailing_squash_divisor",
    "giveback_beta",
    "capital_protect_mfe_mult",
    "capital_protect_regression_frac",
    "capital_protect_lock_frac",
    "capital_protect_min_lock_bps",
    "atr_power",
    "atr_multiplier",
    "hard_tp_abs_pct",
    "adverse_exit_enabled",
    "adverse_exit_min_mae_atr",
    "adverse_exit_min_speed",
    "adverse_exit_theta_quantile",
    "adverse_exit_theta",
    "adverse_exit_alpha",
    "adverse_exit_beta",
    "adverse_exit_delta",
    "adverse_exit_fast_bars",
    "adverse_exit_max_mfe_atr",
    "exit_pressure_enabled",
    "exit_pressure_alpha",
    "exit_pressure_beta",
    "exit_pressure_delta",
    "exit_pressure_kappa",
    "exit_pressure_psi",
    "exit_pressure_omega",
    "exit_pressure_min_multiplier",
    "redeploy_scale_bps",
    "target_holding_hours",
    "churn_penalty_bps",
    "median_barrier_frac",
    "policy_median_barrier_frac",
}


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


def _clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return None
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            return float(text)
        except Exception:
            return text
    return value


def _normalise_archetype_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("policy_archetype_"):
        text = text[len("policy_archetype_") :]
    return text


def _load_parent_params(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("strategies") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError(f"Policy params missing strategies list: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("selected") is False:
            continue
        sid = str(row.get("strategy_id") or row.get("strategy_for_inference") or "").strip()
        if not sid:
            continue
        params: dict[str, Any] = {}
        for key in EXIT_PARAM_KEYS:
            if key in row:
                value = _clean_value(row.get(key))
                if value is not None:
                    params[key] = value
        out[sid] = params
    if not out:
        raise ValueError(f"No selected parent strategies loaded from {path}")
    return out


def _load_archetype_geometry(path: Path) -> dict[tuple[str, str], tuple[dict[str, Any], float]]:
    df = pd.read_csv(path)
    out: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}
    for _, row in df.iterrows():
        sid = str(row.get("strategy_id") or "").strip()
        arch_raw = str(row.get("policy_archetype") or "").strip()
        arch = _normalise_archetype_key(arch_raw)
        geometry = _parse_mapping(row.get("shrinkage_final_geometry"))
        if not sid or not arch or not geometry:
            continue
        clean_geometry: dict[str, Any] = {}
        for key, value in geometry.items():
            if key == "size_power":
                continue
            if key in EXIT_PARAM_KEYS:
                cleaned = _clean_value(value)
                if cleaned is not None:
                    clean_geometry[str(key)] = cleaned
        size_power = _clean_value(geometry.get("size_power"))
        try:
            size_power_f = float(size_power)
        except Exception:
            size_power_f = np.nan
        out[(sid, arch)] = (clean_geometry, size_power_f)
        if arch_raw != arch:
            out[(sid, arch_raw)] = (clean_geometry, size_power_f)
    if not out:
        raise ValueError(f"No archetype geometry rows loaded from {path}")
    return out


def _write_prepared_candidate_file(candidates: Path, out_dir: Path) -> Path:
    raw = pd.read_parquet(candidates)
    rows = raw.copy()
    if "rank_pct" not in rows.columns:
        if "normalized_rank_score" in rows.columns:
            rows["rank_pct"] = pd.to_numeric(rows["normalized_rank_score"], errors="coerce")
        elif "strategy_rank_pct" in rows.columns:
            rows["rank_pct"] = pd.to_numeric(rows["strategy_rank_pct"], errors="coerce")
        else:
            raise ValueError("Candidate table has no rank_pct/normalized_rank_score")
    if "barrier_pct" not in rows.columns:
        raise ValueError("Candidate table missing barrier_pct")
    rows["__source_row_id"] = np.arange(len(rows), dtype=np.int64)
    prepared = out_dir / "prepared_candidates_for_path_replay.parquet"
    rows.to_parquet(prepared, index=False)
    return prepared


def _simulate_rows(
    rows: pd.DataFrame,
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    params: Mapping[str, Any],
    cost_pct: float,
    market_mode: str,
    geometry_source: str,
    geometry_key: str,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    sim = simulate_and_score(
        rows.copy(),
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        cost_pct=float(cost_pct),
        size_power=1.0,
        max_concurrent_trades=10**9,
        max_concurrent_per_asset=10**9,
        market_mode=market_mode,
        **_without_concurrency_param(dict(params)),
    )
    mask = np.asarray(sim.get("selected_mask"), dtype=bool)
    if len(mask) != len(rows):
        raise ValueError(
            f"selected_mask length mismatch: rows={len(rows)} mask={len(mask)} source={geometry_source}"
        )
    selected = rows.loc[mask].copy().reset_index(drop=True)
    raw_gains = np.asarray(sim.get("raw_gains", []), dtype=np.float64)
    gross_gains = np.asarray(sim.get("gross_gains", []), dtype=np.float64)
    sizes = np.asarray(sim.get("sizes", []), dtype=np.float64)
    if len(selected) != len(raw_gains) or len(selected) != len(sizes):
        raise ValueError(
            f"simulation output mismatch: rows={len(selected)} gains={len(raw_gains)} sizes={len(sizes)}"
        )
    denom = np.where(np.abs(sizes) > 1e-12, sizes, np.nan)
    net_return = raw_gains / denom
    gross_return = gross_gains / denom
    holding_bars = np.maximum(
        1,
        np.asarray(sim.get("exit_bars", np.ones(len(selected))), dtype=np.int32),
    )
    timestamps = pd.to_datetime(selected["timestamp"], utc=True, errors="coerce")
    side = pd.to_numeric(selected.get("side"), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    entry = np.asarray(sim.get("entry_prices", selected.get("entry_price", 1.0)), dtype=np.float64)
    if len(entry) != len(selected):
        entry = np.ones(len(selected), dtype=np.float64)
    exit_price = entry * (1.0 + side * gross_return)
    selected["entry_price"] = entry
    selected["exit_price"] = exit_price
    selected["exit_timestamp"] = timestamps + pd.to_timedelta(holding_bars * 15, unit="m")
    selected["holding_bars"] = holding_bars
    selected["net_return"] = net_return
    selected["gross_return"] = gross_return
    selected["fees_bps"] = (gross_return - net_return) * 10_000.0
    selected["simple_policy_exit_reason"] = list(sim.get("exit_reason", [""] * len(selected)))
    selected["policy_geometry_source"] = geometry_source
    selected["policy_geometry_key"] = geometry_key
    selected["policy_sl_mult"] = float(params.get("sl_mult", np.nan))
    selected["policy_trailing_activation_mult"] = float(
        params.get("trailing_activation_mult", np.nan)
    )
    selected["policy_capital_protect_mfe_mult"] = float(
        params.get("capital_protect_mfe_mult", np.nan)
    )
    return selected


def _materialize_geometry_ledger(
    *,
    bundles: list[Any],
    parent_params: Mapping[str, Mapping[str, Any]],
    archetype_geometry: Mapping[tuple[str, str], tuple[Mapping[str, Any], float]],
    mode: str,
    cost_pct: float,
    market_mode: str,
    apply_size_power: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        sid = str(bundle.strategy_id)
        parent = dict(parent_params.get(sid) or bundle.base_params)
        if mode == "side_parent":
            frames.append(
                _simulate_rows(
                    bundle.rows,
                    bundle.paths,
                    params=parent,
                    cost_pct=cost_pct,
                    market_mode=market_mode,
                    geometry_source="side_parent",
                    geometry_key=sid,
                )
            )
            continue
        if "policy_archetype" in bundle.rows.columns:
            arch_col = "policy_archetype"
        elif "local_side_archetype" in bundle.rows.columns:
            arch_col = "local_side_archetype"
        else:
            arch_col = ""
        if not arch_col:
            frames.append(
                _simulate_rows(
                    bundle.rows,
                    bundle.paths,
                    params=parent,
                    cost_pct=cost_pct,
                    market_mode=market_mode,
                    geometry_source="side_parent_fallback",
                    geometry_key=sid,
                )
            )
            continue
        for arch_value, group in bundle.rows.groupby(arch_col, sort=True, dropna=False):
            idx = group.index.to_numpy(dtype=np.int64)
            arch = _normalise_archetype_key(arch_value)
            local = dict(parent)
            source = "side_parent_fallback"
            geometry_key = f"{sid}|{arch}"
            geometry = archetype_geometry.get((sid, arch)) or archetype_geometry.get(
                (sid, str(arch_value))
            )
            if geometry is not None:
                local.update(dict(geometry[0]))
                source = "side_archetype_shrunk_geometry"
            simulated = _simulate_rows(
                group.reset_index(drop=True),
                _path_take(bundle.paths, idx),
                params=local,
                cost_pct=cost_pct,
                market_mode=market_mode,
                geometry_source=source,
                geometry_key=geometry_key,
            )
            if apply_size_power and geometry is not None:
                size_power = geometry[1]
                if np.isfinite(size_power) and size_power > 0.0:
                    simulated["portfolio_rank_size_power"] = float(size_power)
                    simulated["policy_size_power"] = float(size_power)
            frames.append(simulated)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp", "strategy_id", "symbol", "side"]).reset_index(drop=True)
    return out


def _accepted_with_candidates(decisions: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    norm = normalise_candidate_table(candidates)
    indexed = norm.reset_index(drop=True).reset_index(names="candidate_index")
    accepted = decisions[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return accepted
    merged = accepted.merge(indexed, on="candidate_index", how="left", suffixes=("", "_candidate"))
    for col in ("net_return", "gross_return", "simple_policy_exit_reason"):
        pos_col = f"position_{col}" if col != "simple_policy_exit_reason" else "position_exit_reason"
        if pos_col in merged.columns:
            values = merged[pos_col]
            if col == "simple_policy_exit_reason":
                values = values.where(values.astype(str).str.len() > 0, merged[col])
            else:
                values = pd.to_numeric(values, errors="coerce").where(
                    pd.to_numeric(values, errors="coerce").notna(),
                    merged[col],
                )
            merged[col] = values
    merged["net_pnl"] = (
        pd.to_numeric(merged["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(merged["net_return"], errors="coerce").fillna(0.0)
    )
    merged["gross_pnl"] = (
        pd.to_numeric(merged["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(merged["gross_return"], errors="coerce").fillna(0.0)
    )
    return merged


def _period_summary(accepted: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(columns=group_cols)
    work = accepted.copy()
    ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["month"] = ts.dt.strftime("%Y-%m")
    work["week"] = ts.dt.to_period("W").astype(str)
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: value for col, value in zip(group_cols, keys, strict=False)}
        period_days = max(ts.loc[group.index].dt.date.nunique(), 1)
        net_return = pd.to_numeric(group["net_return"], errors="coerce").fillna(0.0)
        gross_return = pd.to_numeric(group["gross_return"], errors="coerce").fillna(0.0)
        position = pd.to_numeric(group["position_size"], errors="coerce").fillna(0.0)
        reason = group["simple_policy_exit_reason"].astype(str)
        rec.update(
            {
                "trades": int(len(group)),
                "trades_per_day": float(len(group) / period_days),
                "symbols": int(group["symbol"].astype(str).nunique()) if "symbol" in group else 0,
                "net_pnl": float(group["net_pnl"].sum()),
                "gross_pnl": float(group["gross_pnl"].sum()),
                "mean_net_return_per_trade": float(net_return.mean()) if len(group) else 0.0,
                "mean_gross_return_per_trade": float(gross_return.mean()) if len(group) else 0.0,
                "notional_weighted_net_return": float(
                    group["net_pnl"].sum() / max(float(position.sum()), 1e-12)
                ),
                "positive_net_rate": float((net_return > 0).mean()) if len(group) else 0.0,
                "full_sl_rate": float(reason.eq("full_sl").mean()) if len(group) else 0.0,
                "timeout_rate": float(reason.eq("timeout").mean()) if len(group) else 0.0,
                "trailing_rate": float(reason.eq("trailing").mean()) if len(group) else 0.0,
                "capital_protect_rate": float(reason.eq("capital_protect").mean()) if len(group) else 0.0,
                "mean_position_size": float(position.mean()) if len(group) else 0.0,
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _run_replay(
    candidates: pd.DataFrame,
    *,
    params_path: Path,
    ev_curve: dict[str, Any] | None,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    params = load_portfolio_policy_params(params_path)
    normalised = normalise_candidate_table(candidates)
    decisions, equity, metrics = replay_candidates(
        normalised,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    metrics = compute_replay_metrics(normalised, decisions, equity, params=params)
    accepted = _accepted_with_candidates(decisions, normalised)
    return decisions, equity, metrics, accepted


def _candidate_exit_summary(frame: pd.DataFrame, *, arm: str) -> dict[str, Any]:
    reason = frame["simple_policy_exit_reason"].astype(str)
    return {
        "arm": arm,
        "candidate_rows": int(len(frame)),
        "candidate_symbols": int(frame["symbol"].astype(str).nunique()) if "symbol" in frame else 0,
        "candidate_mean_net_return": float(pd.to_numeric(frame["net_return"], errors="coerce").mean()),
        "candidate_mean_gross_return": float(pd.to_numeric(frame["gross_return"], errors="coerce").mean()),
        "candidate_positive_net_rate": float((pd.to_numeric(frame["net_return"], errors="coerce") > 0).mean()),
        "candidate_full_sl_rate": float(reason.eq("full_sl").mean()),
        "candidate_timeout_rate": float(reason.eq("timeout").mean()),
        "candidate_trailing_rate": float(reason.eq("trailing").mean()),
        "candidate_capital_protect_rate": float(reason.eq("capital_protect").mean()),
        "candidate_mean_holding_bars": float(pd.to_numeric(frame["holding_bars"], errors="coerce").mean()),
        "candidate_archetype_size_power_rows": int(
            pd.to_numeric(frame.get("portfolio_rank_size_power"), errors="coerce").notna().sum()
        )
        if "portfolio_rank_size_power" in frame.columns
        else 0,
        "archetype_geometry_rows": int(
            frame["policy_geometry_source"].astype(str).eq("side_archetype_shrunk_geometry").sum()
        )
        if "policy_geometry_source" in frame
        else 0,
        "parent_fallback_rows": int(
            frame["policy_geometry_source"].astype(str).eq("side_parent_fallback").sum()
        )
        if "policy_geometry_source" in frame
        else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--policy-params", type=Path, default=DEFAULT_POLICY_PARAMS)
    parser.add_argument("--portfolio-policy", type=Path, default=DEFAULT_PORTFOLIO_POLICY)
    parser.add_argument("--archetype-policy-summary", type=Path, default=DEFAULT_ARCHETYPE_POLICY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument(
        "--exchange",
        default="krakenfutures",
        help="Exchange data component used for perps replay. Defaults to Kraken futures.",
    )
    parser.add_argument("--market-mode", choices=["spot", "perps"], default="perps")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.01)
    args = parser.parse_args()

    # The policy replay root is resolved from exchange environment variables.
    # Pin this comparison to Kraken futures so it cannot silently use Binance.
    for key in ("EPM_EXCHANGE", "EXCHANGE_NAME", "PRIMARY_EXCHANGE"):
        os.environ[key] = str(args.exchange)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = _write_prepared_candidate_file(args.candidates, args.out_dir)
    rows = _prepare_rows(prepared_path, min_rank=0.0)
    bundles = _load_bundles(
        rows,
        data_root=str(args.data_root),
        market_mode=str(args.market_mode),
        path_len=int(args.path_len),
        min_rows_per_strategy=1,
    )
    parent_params = _load_parent_params(args.policy_params)
    archetype_geometry = _load_archetype_geometry(args.archetype_policy_summary)
    cost_pct = float(args.round_trip_cost_pct) / 2.0

    parent_ledger = _materialize_geometry_ledger(
        bundles=bundles,
        parent_params=parent_params,
        archetype_geometry=archetype_geometry,
        mode="side_parent",
        cost_pct=cost_pct,
        market_mode=str(args.market_mode),
    )
    archetype_ledger = _materialize_geometry_ledger(
        bundles=bundles,
        parent_params=parent_params,
        archetype_geometry=archetype_geometry,
        mode="side_archetype",
        cost_pct=cost_pct,
        market_mode=str(args.market_mode),
        apply_size_power=False,
    )
    archetype_size_ledger = _materialize_geometry_ledger(
        bundles=bundles,
        parent_params=parent_params,
        archetype_geometry=archetype_geometry,
        mode="side_archetype",
        cost_pct=cost_pct,
        market_mode=str(args.market_mode),
        apply_size_power=True,
    )

    ledgers = {
        "side_parent_geometry": parent_ledger,
        "side_archetype_geometry_parent_evcurve": archetype_ledger,
        "side_archetype_geometry_self_evcurve": archetype_ledger,
        "side_archetype_geometry_size_parent_evcurve": archetype_size_ledger,
        "side_archetype_geometry_size_self_evcurve": archetype_size_ledger,
    }
    parent_ev_curve = fit_hierarchical_ev_curves(parent_ledger)
    replay_outputs: dict[str, dict[str, Any]] = {}
    overall_rows: list[dict[str, Any]] = []
    accepted_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    candidate_summary_rows: list[dict[str, Any]] = []

    ledger_dir = args.out_dir / "candidate_ledgers"
    ledger_dir.mkdir(exist_ok=True)
    parent_ledger.to_parquet(ledger_dir / "side_parent_geometry_candidates.parquet", index=False)
    archetype_ledger.to_parquet(ledger_dir / "side_archetype_geometry_candidates.parquet", index=False)
    archetype_size_ledger.to_parquet(
        ledger_dir / "side_archetype_geometry_size_candidates.parquet",
        index=False,
    )

    for arm, ledger in ledgers.items():
        ev_curve = (
            fit_hierarchical_ev_curves(ledger)
            if arm.endswith("_self_evcurve")
            else parent_ev_curve
        )
        decisions, equity, metrics, accepted = _run_replay(
            ledger,
            params_path=args.portfolio_policy,
            ev_curve=ev_curve,
            market_mode=str(args.market_mode),
        )
        decisions["arm"] = arm
        equity["arm"] = arm
        accepted["arm"] = arm
        decision_frames.append(decisions)
        equity_frames.append(equity)
        accepted_frames.append(accepted)
        candidate_summary_rows.append(_candidate_exit_summary(ledger, arm=arm))
        overall = {"arm": arm, **metrics}
        overall_rows.append(overall)
        replay_outputs[arm] = {
            "metrics": metrics,
            "decisions": str(args.out_dir / f"{arm}_decisions.parquet"),
            "equity": str(args.out_dir / f"{arm}_equity.parquet"),
            "accepted": str(args.out_dir / f"{arm}_accepted_trades.parquet"),
        }
        decisions.to_parquet(args.out_dir / f"{arm}_decisions.parquet", index=False)
        equity.to_parquet(args.out_dir / f"{arm}_equity.parquet", index=False)
        accepted.to_parquet(args.out_dir / f"{arm}_accepted_trades.parquet", index=False)

    accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
    decisions_all = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    equity_all = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()

    pd.DataFrame(overall_rows).to_csv(args.out_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(candidate_summary_rows).to_csv(args.out_dir / "candidate_exit_summary.csv", index=False)
    accepted_all.to_parquet(args.out_dir / "accepted_trades_all_arms.parquet", index=False)
    decisions_all.to_parquet(args.out_dir / "decisions_all_arms.parquet", index=False)
    equity_all.to_parquet(args.out_dir / "equity_all_arms.parquet", index=False)

    for name, cols in {
        "month": ["arm", "month"],
        "week": ["arm", "week"],
        "side": ["arm", "side"],
        "policy_archetype": ["arm", "policy_archetype"],
        "month_side_archetype": ["arm", "month", "side", "policy_archetype"],
        "week_side_archetype": ["arm", "week", "side", "policy_archetype"],
    }.items():
        _period_summary(accepted_all, group_cols=cols).to_csv(
            args.out_dir / f"{name}_metrics.csv",
            index=False,
        )

    # Accepted-membership overlap versus deployed side-parent geometry.
    overlap_rows: list[dict[str, Any]] = []
    key_cols = ["timestamp", "symbol", "side", "strategy_id"]
    parent_acc = accepted_all[accepted_all["arm"].eq("side_parent_geometry")]
    parent_set = set(map(tuple, parent_acc[key_cols].astype(str).to_numpy()))
    for arm, group in accepted_all.groupby("arm", sort=True):
        arm_set = set(map(tuple, group[key_cols].astype(str).to_numpy()))
        overlap_rows.append(
            {
                "arm": arm,
                "accepted_trades": int(len(arm_set)),
                "parent_accepted_trades": int(len(parent_set)),
                "overlap_with_parent": int(len(arm_set & parent_set)),
                "overlap_vs_parent_rate": float(len(arm_set & parent_set) / max(len(parent_set), 1)),
                "new_vs_parent": int(len(arm_set - parent_set)),
                "dropped_vs_parent": int(len(parent_set - arm_set)),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(args.out_dir / "accepted_overlap_vs_parent.csv", index=False)

    manifest = {
        "generated_by": "run_archetype_exit_geometry_replay_compare",
        "candidate_source": str(args.candidates),
        "prepared_candidate_source": str(prepared_path),
        "policy_params": str(args.policy_params),
        "portfolio_policy": str(args.portfolio_policy),
        "archetype_policy_summary": str(args.archetype_policy_summary),
        "data_root": str(args.data_root),
        "exchange": str(args.exchange),
        "market_mode": str(args.market_mode),
        "path_len": int(args.path_len),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_pct_per_side": float(cost_pct),
        "arms": list(ledgers),
        "bundle_count": int(len(bundles)),
        "input_rows": int(len(rows)),
        "parent_geometry_rows": int(len(parent_ledger)),
        "archetype_geometry_rows": int(len(archetype_ledger)),
        "archetype_geometry_size_rows": int(len(archetype_size_ledger)),
        "replay_outputs": replay_outputs,
        "notes": [
            "The same threshold-passed candidate universe is used for all arms.",
            "side_archetype_geometry_parent_evcurve isolates exit geometry while keeping parent EV priority mapping.",
            "side_archetype_geometry_self_evcurve represents a self-consistent promotion where EV curves are rebuilt from archetype-geometry returns.",
            "Round-trip cost is applied once in simulate_and_score as per-side cost_pct.",
        ],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"event": "archetype_exit_geometry_replay_compare_done", **manifest}), sort_keys=True))


if __name__ == "__main__":
    main()
