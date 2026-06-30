#!/usr/bin/env python3
"""Refit and ablate portfolio policy on reliability-blend candidate scores.

The simple-policy candidate table already contains executable outcomes, fees,
friction, and reliability-blend rank columns.  This script keeps that candidate
universe fixed, refits the portfolio allocator on the reliability-blend score
distribution, then compares a small set of mix-control ablations on historical
and optional OOS candidate tables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    INITIAL_WALLET,
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    load_portfolio_policy_params,
    normalise_candidate_table,
    optimise_params,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)


DEFAULT_TRAIN_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_OOS_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_matrix_native_candidates_20260624_jun15_22_forced_features_floor070_pnl"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_LEGACY_CONFIG = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_longdist050"
    "/policy_params/optimized_portfolio_policy_config.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/reliability_blend_portfolio_policy_ablation_20260624"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _variant_records(
    *,
    legacy_params: PortfolioPolicyParams,
    refit_params: PortfolioPolicyParams,
) -> list[tuple[str, PortfolioPolicyParams, str]]:
    legacy_floor = replace(
        legacy_params,
        global_threshold_floor=max(float(legacy_params.global_threshold_floor), 0.70),
    )
    refit_floor = replace(
        refit_params,
        global_threshold_floor=max(float(refit_params.global_threshold_floor), 0.70),
    )
    bar4 = max(4, int(refit_floor.max_new_entries_per_bar))
    max_pos10 = max(10, int(refit_floor.max_concurrent_positions))
    return [
        (
            "legacy_fixed_floor070",
            legacy_floor,
            "Prior optimized policy, applied to reliability-blend floor-0.70 candidates.",
        ),
        (
            "refit_best",
            refit_floor,
            "Optuna refit on historical reliability-blend floor-0.70 candidates.",
        ),
        (
            "refit_bar4_strategy_bar1",
            replace(
                refit_floor,
                max_new_entries_per_bar=bar4,
                max_new_entries_per_strategy_per_bar=1,
            ),
            "Refit params with explicit one-entry-per-strategy timestamp bucket.",
        ),
        (
            "refit_bar4_strategy_bar2",
            replace(
                refit_floor,
                max_new_entries_per_bar=bar4,
                max_new_entries_per_strategy_per_bar=2,
            ),
            "Refit params with looser two-entry-per-strategy timestamp bucket.",
        ),
        (
            "refit_mix_capacity10_strategy_bar1",
            replace(
                refit_floor,
                max_concurrent_positions=max_pos10,
                max_concurrent_per_strategy=max(
                    2,
                    min(
                        int(refit_floor.max_concurrent_per_strategy or max_pos10),
                        int(np.ceil(max_pos10 * 0.50)),
                    ),
                ),
                max_new_entries_per_bar=bar4,
                max_new_entries_per_strategy_per_bar=1,
            ),
            "Capacity-relaxed mix ablation with one new entry per strategy per bar.",
        ),
        (
            "refit_strategy_cap3_bar1",
            replace(
                refit_floor,
                max_concurrent_per_strategy=3,
                max_new_entries_per_strategy_per_bar=1,
            ),
            "Stricter strategy exposure cap with one new entry per strategy per bar.",
        ),
        (
            "refit_strategy_cap2_bar1",
            replace(
                refit_floor,
                max_concurrent_per_strategy=2,
                max_new_entries_per_strategy_per_bar=1,
            ),
            "Strict strategy exposure cap with one new entry per strategy per bar.",
        ),
    ]


def _accepted_trades(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    work = candidates.reset_index(drop=True).copy()
    accepted = decisions.loc[decisions["accepted"]].copy()
    if accepted.empty:
        return accepted
    accepted["candidate_index"] = pd.to_numeric(
        accepted["candidate_index"], errors="coerce"
    ).astype("Int64")
    valid = accepted["candidate_index"].notna()
    accepted = accepted.loc[valid].copy()
    cand = work.iloc[accepted["candidate_index"].astype(int).to_numpy()].reset_index(
        drop=True
    )
    accepted = accepted.reset_index(drop=True)
    cand = cand.rename(columns={col: f"candidate_{col}" for col in cand.columns})
    out = pd.concat([accepted, cand], axis=1)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if "candidate_timestamp" in out.columns:
        out["candidate_timestamp"] = pd.to_datetime(
            out["candidate_timestamp"], utc=True, errors="coerce"
        )
    return out


def _trade_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "trade_count": 0,
            "timestamp_count": 0,
            "symbol_count": 0,
            "strategy_count": 0,
            "win_rate": np.nan,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "cost_pnl": 0.0,
            "mean_net_return": np.nan,
            "median_net_return": np.nan,
            "q05_net_return": np.nan,
            "q25_net_return": np.nan,
            "notional_weighted_net_return": np.nan,
            "mean_position_size": np.nan,
            "strategy_concentration": np.nan,
        }
    net_return = pd.to_numeric(rows["candidate_net_return"], errors="coerce").fillna(0.0)
    gross_return = pd.to_numeric(
        rows.get("candidate_gross_return", rows["candidate_net_return"]),
        errors="coerce",
    ).fillna(0.0)
    size = pd.to_numeric(rows["position_size"], errors="coerce").fillna(0.0)
    net_pnl = size * net_return
    gross_pnl = size * gross_return
    strategy_counts = rows["strategy_id"].astype(str).value_counts(normalize=True)
    return {
        "trade_count": int(len(rows)),
        "timestamp_count": int(pd.to_datetime(rows["timestamp"], utc=True).nunique()),
        "symbol_count": int(rows["symbol"].astype(str).nunique()),
        "strategy_count": int(rows["strategy_id"].astype(str).nunique()),
        "win_rate": float((net_pnl > 0.0).mean()),
        "net_pnl": float(net_pnl.sum()),
        "gross_pnl": float(gross_pnl.sum()),
        "cost_pnl": float((gross_pnl - net_pnl).sum()),
        "mean_net_return": float(net_return.mean()),
        "median_net_return": float(net_return.median()),
        "q05_net_return": float(net_return.quantile(0.05)),
        "q25_net_return": float(net_return.quantile(0.25)),
        "notional_weighted_net_return": float(net_pnl.sum() / max(size.sum(), 1e-9)),
        "mean_position_size": float(size.mean()),
        "strategy_concentration": float(strategy_counts.max()) if len(strategy_counts) else np.nan,
    }


def _windowed_metrics(
    accepted: pd.DataFrame,
    *,
    sample: str,
    variant: str,
    max_timestamp: pd.Timestamp,
    group_cols: Iterable[str] = (),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if accepted.empty:
        windows = {"all": None, "last_1w": 7, "last_2w": 14, "last_4w": 28}
    else:
        accepted = accepted.copy()
        accepted["timestamp"] = pd.to_datetime(
            accepted["timestamp"], utc=True, errors="coerce"
        )
        windows = {"all": None, "last_1w": 7, "last_2w": 14, "last_4w": 28}
    for window, days in windows.items():
        if days is None:
            subset = accepted
        else:
            cutoff = max_timestamp - pd.Timedelta(days=int(days))
            subset = accepted.loc[accepted["timestamp"] >= cutoff]
        if group_cols:
            if subset.empty:
                continue
            for keys, group in subset.groupby(list(group_cols), dropna=False, sort=True):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                rec = {
                    "sample": sample,
                    "variant": variant,
                    "window": window,
                }
                rec.update({col: val for col, val in zip(group_cols, keys)})
                rec.update(_trade_metrics(group))
                rows.append(rec)
        else:
            rec = {"sample": sample, "variant": variant, "window": window}
            rec.update(_trade_metrics(subset))
            rows.append(rec)
    return rows


def _evaluate(
    *,
    sample: str,
    candidates: pd.DataFrame,
    variants: list[tuple[str, PortfolioPolicyParams, str]],
    ev_curve: dict[str, Any],
    output_dir: Path,
    market_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    strategy_window_rows: list[dict[str, Any]] = []
    max_ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").max()
    sample_dir = output_dir / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    for variant, params, description in variants:
        decisions, equity, metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode=market_mode,
        )
        accepted = _accepted_trades(candidates, decisions)
        accepted.to_parquet(sample_dir / f"{variant}_accepted_trades.parquet", index=False)
        decisions.to_parquet(sample_dir / f"{variant}_decisions.parquet", index=False)
        equity.to_parquet(sample_dir / f"{variant}_equity_curve.parquet", index=False)
        summary = {
            "sample": sample,
            "variant": variant,
            "description": description,
            "candidate_rows": int(len(candidates)),
            "timestamp_min": pd.to_datetime(candidates["timestamp"], utc=True).min().isoformat(),
            "timestamp_max": pd.to_datetime(candidates["timestamp"], utc=True).max().isoformat(),
            "objective": metrics.get("objective"),
            "compounded_return": metrics.get("compounded_return"),
            "net_pnl": metrics.get("net_pnl"),
            "gross_pnl": metrics.get("gross_pnl"),
            "final_wallet": metrics.get("final_wallet"),
            "max_drawdown": metrics.get("max_drawdown"),
            "trade_count": metrics.get("trade_count"),
            "full_sl_rate": metrics.get("full_sl_rate"),
            "timeout_rate": metrics.get("timeout_rate"),
            "strategy_concentration": metrics.get("strategy_concentration"),
            "side_concentration": metrics.get("side_concentration"),
            "avg_open_positions": metrics.get("avg_open_positions"),
            "position_utilization": metrics.get("position_utilization"),
            "missed_high_confidence_trades": metrics.get("missed_high_confidence_trades"),
            "max_concurrent_positions": params.max_concurrent_positions,
            "max_concurrent_per_strategy": params.max_concurrent_per_strategy,
            "max_new_entries_per_bar": params.max_new_entries_per_bar,
            "max_new_entries_per_strategy_per_bar": params.max_new_entries_per_strategy_per_bar,
            "global_threshold_floor": params.global_threshold_floor,
            "occupancy_threshold_alpha": params.occupancy_threshold_alpha,
            "occupancy_threshold_power": params.occupancy_threshold_power,
            "rank_size_power": params.rank_size_power,
            "max_total_wallet_allocation_pct": params.max_total_wallet_allocation_pct,
            "rejection_reasons": json.dumps(_json_safe(metrics.get("rejection_reasons", {}))),
            "params_json": json.dumps(_json_safe(params.to_live_config()), sort_keys=True),
        }
        summary_rows.append(summary)
        window_rows.extend(
            _windowed_metrics(
                accepted,
                sample=sample,
                variant=variant,
                max_timestamp=max_ts,
            )
        )
        strategy_window_rows.extend(
            _windowed_metrics(
                accepted,
                sample=sample,
                variant=variant,
                max_timestamp=max_ts,
                group_cols=("strategy_id",),
            )
        )
    return summary_rows, window_rows, strategy_window_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--oos-candidates", type=Path, default=DEFAULT_OOS_CANDIDATES)
    parser.add_argument("--legacy-config", type=Path, default=DEFAULT_LEGACY_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-evaluations", type=int, default=500)
    parser.add_argument("--market-mode", type=str, default="perps")
    parser.add_argument(
        "--reuse-refit-manifest",
        type=Path,
        default=None,
        help="Reuse refit_params from a previous ablation manifest instead of running Optuna.",
    )
    args = parser.parse_args()

    if not args.train_candidates.exists():
        raise FileNotFoundError(f"Missing train candidate table: {args.train_candidates}")
    if not args.legacy_config.exists():
        raise FileNotFoundError(f"Missing legacy policy config: {args.legacy_config}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train = normalise_candidate_table(pd.read_parquet(args.train_candidates))
    legacy_params = load_portfolio_policy_params(args.legacy_config)
    if args.reuse_refit_manifest is not None:
        prior = json.loads(args.reuse_refit_manifest.read_text(encoding="utf-8"))
        refit_params = portfolio_policy_params_from_live_config(
            prior.get("refit_params", {})
        )
        refit_train_metrics = dict(prior.get("refit_train_metrics", {}))
    else:
        refit_params, refit_train_metrics = optimise_params(
            train,
            max_evaluations=int(args.max_evaluations),
            market_mode=args.market_mode,
        )
    variants = _variant_records(
        legacy_params=legacy_params,
        refit_params=refit_params,
    )
    ev_curve = fit_hierarchical_ev_curves(train)
    summary_rows, window_rows, strategy_window_rows = _evaluate(
        sample="historical_refit",
        candidates=train,
        variants=variants,
        ev_curve=ev_curve,
        output_dir=output_dir,
        market_mode=args.market_mode,
    )
    if args.oos_candidates.exists():
        oos = normalise_candidate_table(pd.read_parquet(args.oos_candidates))
        s2, w2, sw2 = _evaluate(
            sample="oos_jun15_22",
            candidates=oos,
            variants=variants,
            ev_curve=ev_curve,
            output_dir=output_dir,
            market_mode=args.market_mode,
        )
        summary_rows.extend(s2)
        window_rows.extend(w2)
        strategy_window_rows.extend(sw2)

    summary = pd.DataFrame(summary_rows)
    windows = pd.DataFrame(window_rows)
    strategy_windows = pd.DataFrame(strategy_window_rows)
    summary.to_csv(output_dir / "portfolio_policy_ablation_summary.csv", index=False)
    windows.to_csv(output_dir / "portfolio_policy_ablation_windows.csv", index=False)
    strategy_windows.to_csv(
        output_dir / "portfolio_policy_ablation_windows_by_strategy.csv",
        index=False,
    )
    manifest = {
        "generated_by": "run_reliability_blend_portfolio_policy_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_candidates": str(args.train_candidates),
        "oos_candidates": str(args.oos_candidates),
        "legacy_config": str(args.legacy_config),
        "market_mode": args.market_mode,
        "max_evaluations": int(args.max_evaluations),
        "refit_params": refit_params.to_live_config(),
        "refit_train_metrics": refit_train_metrics,
        "variant_params": {
            name: params.to_live_config() for name, params, _ in variants
        },
        "outputs": {
            "summary": str(output_dir / "portfolio_policy_ablation_summary.csv"),
            "windows": str(output_dir / "portfolio_policy_ablation_windows.csv"),
            "windows_by_strategy": str(
                output_dir / "portfolio_policy_ablation_windows_by_strategy.csv"
            ),
        },
    }
    (output_dir / "portfolio_policy_ablation_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n"
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:8000])
    print(f"\nWrote {output_dir}")


if __name__ == "__main__":
    main()
