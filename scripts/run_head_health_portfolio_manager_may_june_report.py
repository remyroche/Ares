#!/usr/bin/env python3
"""Replay HeadHealth through the portfolio manager and report weekly metrics.

The report uses realised simple-policy candidate rows.  It compares the frozen
portfolio policy against the same policy after applying the frozen HeadHealth
overlay to portfolio-manager inputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    fit_monotone_ev_curve,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)
from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import _file_sha256
from scripts.run_head_health_portfolio_policy_ablation import (
    HeadHealthState,
    _apply_head_health,
    _read_base_config,
)
from scripts.run_reliability_blend_portfolio_policy_ablation import _accepted_trades


DEFAULT_TRAIN_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_OOS_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_matrix_native_candidates_20260624_jun15_22_forced_features_floor070_pnl"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_MATURED_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_symbol_native_candidates_20260624_jun23_00_08_matured_floor070_pnl"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_PORTFOLIO_MANIFEST = Path(
    "data_perp/reports/reliability_blend_portfolio_policy_ablation_20260624"
    "/portfolio_policy_ablation_manifest.json"
)
DEFAULT_HEAD_HEALTH_CONFIG = Path(
    "data_perp/reports/head_health_portfolio_policy_frozen_action_20260624"
    "/head_health_policy_freeze_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/head_health_portfolio_manager_may_june_20260624"
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


def _head_from_strategy(strategy_id: Any) -> str:
    text = str(strategy_id or "")
    if text.startswith("long_bars"):
        return "long_bars"
    if text.startswith("long_dist"):
        return "long_dist"
    if text.startswith("short_asset"):
        return "short_asset"
    if text.startswith("short_boll"):
        return "short_boll"
    return text.split("_", 2)[0] if text else "unknown"


def _load_portfolio_params(path: Path, variant: str):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    variants = manifest.get("variant_params")
    if isinstance(variants, dict) and variant in variants:
        payload = variants[variant]
    elif isinstance(manifest.get("refit_params"), dict):
        payload = manifest["refit_params"]
    else:
        raise RuntimeError(f"Could not resolve portfolio variant {variant!r} from {path}")
    params = portfolio_policy_params_from_live_config(payload)
    return replace(params, global_threshold_floor=max(float(params.global_threshold_floor), 0.70))


def _load_candidates(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["_candidate_source_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No candidate paths supplied.")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = normalise_candidate_table(out)
    out = out.drop_duplicates(
        ["timestamp", "strategy_id", "symbol", "side"],
        keep="last",
    ).sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    out["head"] = out["strategy_id"].map(_head_from_strategy)
    return out.reset_index(drop=True)


def _week_start(ts: pd.Series) -> pd.Series:
    values = pd.to_datetime(ts, utc=True, errors="coerce")
    return values.dt.floor("D") - pd.to_timedelta(values.dt.weekday, unit="D")


def _metric_rows(
    accepted: pd.DataFrame,
    *,
    variant: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or []
    rows = accepted.copy()
    if rows.empty:
        return pd.DataFrame()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.loc[(rows["timestamp"] >= start) & (rows["timestamp"] <= end)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["week_start"] = _week_start(rows["timestamp"])
    rows["week_end"] = rows["week_start"] + pd.Timedelta(days=6)
    rows["head"] = rows["strategy_id"].map(_head_from_strategy)
    net_return = pd.to_numeric(rows["candidate_net_return"], errors="coerce").fillna(0.0)
    gross_return = pd.to_numeric(
        rows.get("candidate_gross_return", rows["candidate_net_return"]),
        errors="coerce",
    ).fillna(0.0)
    position_size = pd.to_numeric(rows["position_size"], errors="coerce").fillna(0.0)
    rows["_net_pnl"] = net_return * position_size
    rows["_gross_pnl"] = gross_return * position_size
    rows["_cost_pnl"] = rows["_gross_pnl"] - rows["_net_pnl"]
    rows["_win"] = rows["_net_pnl"] > 0.0
    reason = rows.get("candidate_simple_policy_exit_reason", pd.Series("", index=rows.index))
    reason = reason.astype(str).str.lower()
    rows["_full_sl"] = reason.str.contains("full_sl", regex=False)
    rows["_timeout"] = reason.str.contains("timeout", regex=False)
    keys = ["week_start", "week_end"] + group_cols
    out: list[dict[str, Any]] = []
    for keys_value, group in rows.groupby(keys, dropna=False, sort=True):
        if not isinstance(keys_value, tuple):
            keys_value = (keys_value,)
        rec = {key: value for key, value in zip(keys, keys_value)}
        net = pd.to_numeric(group["candidate_net_return"], errors="coerce").fillna(0.0)
        size = pd.to_numeric(group["position_size"], errors="coerce").fillna(0.0)
        rec.update(
            {
                "variant": variant,
                "trade_count": int(len(group)),
                "timestamp_count": int(group["timestamp"].nunique()),
                "symbol_count": int(group["symbol"].astype(str).nunique()),
                "net_pnl": float(group["_net_pnl"].sum()),
                "gross_pnl": float(group["_gross_pnl"].sum()),
                "cost_pnl": float(group["_cost_pnl"].sum()),
                "win_rate": float(group["_win"].mean()),
                "full_sl_rate": float(group["_full_sl"].mean()),
                "timeout_rate": float(group["_timeout"].mean()),
                "mean_net_return": float(net.mean()),
                "median_net_return": float(net.median()),
                "q05_net_return": float(net.quantile(0.05)),
                "q25_net_return": float(net.quantile(0.25)),
                "notional_weighted_net_return": float(group["_net_pnl"].sum() / max(size.sum(), 1e-9)),
                "mean_position_size": float(size.mean()),
            }
        )
        out.append(rec)
    return pd.DataFrame(out)


def _complete_zero_trade_weeks(
    metrics: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    variants: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    group_col: str | None = None,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    weeks = (
        pd.DataFrame({"week_start": _week_start(timestamps.loc[(timestamps >= start) & (timestamps <= end)])})
        .dropna()
        .drop_duplicates()
        .sort_values("week_start")
    )
    if weeks.empty:
        return metrics
    group_values: list[Any | None] = [None]
    if group_col is not None:
        group_values = sorted(candidates[group_col].dropna().astype(str).unique().tolist())
    existing = set()
    if not metrics.empty:
        key_cols = ["variant", "week_start"] + ([group_col] if group_col is not None else [])
        for row in metrics[key_cols].itertuples(index=False, name=None):
            existing.add(tuple(row))
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for week_start in weeks["week_start"]:
            week_end = pd.Timestamp(week_start) + pd.Timedelta(days=6)
            for group_value in group_values:
                key = (variant, week_start) if group_col is None else (variant, week_start, group_value)
                if key in existing:
                    continue
                rec: dict[str, Any] = {
                    "week_start": week_start,
                    "week_end": week_end,
                    "variant": variant,
                    "trade_count": 0,
                    "timestamp_count": 0,
                    "symbol_count": 0,
                    "net_pnl": 0.0,
                    "gross_pnl": 0.0,
                    "cost_pnl": 0.0,
                    "win_rate": np.nan,
                    "full_sl_rate": np.nan,
                    "timeout_rate": np.nan,
                    "mean_net_return": np.nan,
                    "median_net_return": np.nan,
                    "q05_net_return": np.nan,
                    "q25_net_return": np.nan,
                    "notional_weighted_net_return": np.nan,
                    "mean_position_size": np.nan,
                }
                if group_col is not None:
                    rec[group_col] = group_value
                rows.append(rec)
    if rows:
        metrics = pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True, sort=False)
    sort_cols = ["week_start", "variant"] + ([group_col] if group_col is not None else [])
    return metrics.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _overall_metrics(accepted: pd.DataFrame, *, variant: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    weekly = _metric_rows(accepted, variant=variant, start=start, end=end)
    if weekly.empty:
        return pd.DataFrame()
    total = weekly.copy()
    numeric_sum = ["trade_count", "timestamp_count", "symbol_count", "net_pnl", "gross_pnl", "cost_pnl"]
    rec: dict[str, Any] = {"variant": variant, "period": f"{start.isoformat()}..{end.isoformat()}"}
    for col in numeric_sum:
        rec[col] = float(total[col].sum())
    rec["trade_count"] = int(rec["trade_count"])
    rec["timestamp_count"] = int(pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce").loc[
        lambda s: (s >= start) & (s <= end)
    ].nunique())
    rec["symbol_count"] = int(accepted.loc[
        (pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce") >= start)
        & (pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce") <= end),
        "symbol",
    ].astype(str).nunique())
    rec["win_rate"] = float(
        (pd.to_numeric(accepted.get("candidate_net_return"), errors="coerce").fillna(0.0)
         * pd.to_numeric(accepted.get("position_size"), errors="coerce").fillna(0.0)
        ).loc[
            (pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce") >= start)
            & (pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce") <= end)
        ].gt(0.0).mean()
    )
    return pd.DataFrame([rec])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-deprecated-head-health",
        action="store_true",
        help="Run this deprecated historical audit tool. HeadHealth is disabled from active policy logic.",
    )
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--portfolio-manifest", type=Path, default=DEFAULT_PORTFOLIO_MANIFEST)
    parser.add_argument("--portfolio-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--head-health-config", type=Path, default=DEFAULT_HEAD_HEALTH_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-start", default="2026-05-01 00:00:00")
    parser.add_argument("--report-end", default="2026-06-30 23:59:59")
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()
    if not bool(args.allow_deprecated_head_health):
        raise SystemExit(
            "HeadHealth active execution is deprecated and disabled. Use the "
            "reliability-blend parity/portfolio ablation path instead, or pass "
            "--allow-deprecated-head-health for historical audit reproduction."
        )

    candidate_paths = args.candidate or [
        DEFAULT_TRAIN_CANDIDATES,
        DEFAULT_OOS_CANDIDATES,
        DEFAULT_MATURED_CANDIDATES,
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = normalise_candidate_table(pd.read_parquet(args.train_candidates))
    candidates = _load_candidates(candidate_paths)
    params = _load_portfolio_params(args.portfolio_manifest, args.portfolio_variant)
    config = _read_base_config(args.head_health_config)
    config["base_max_new_entries_per_bar"] = int(params.max_new_entries_per_bar)
    config["base_max_new_entries_per_strategy_per_bar"] = int(
        params.max_new_entries_per_strategy_per_bar
        if params.max_new_entries_per_strategy_per_bar is not None
        else params.max_new_entries_per_bar
    )
    config["base_max_concurrent_per_strategy"] = int(
        params.max_concurrent_per_strategy
        if params.max_concurrent_per_strategy is not None
        else params.max_concurrent_positions
    )
    health_state = HeadHealthState.fit(train, config)
    head_health_candidates = _apply_head_health(
        candidates,
        history=candidates,
        reference=train,
        config=config,
        state=health_state,
    )
    ev_curve = fit_monotone_ev_curve(train)
    baseline_decisions, baseline_equity, baseline_metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=args.market_mode,
    )
    health_decisions, health_equity, health_metrics = replay_candidates(
        head_health_candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=args.market_mode,
    )
    baseline_accepted = _accepted_trades(candidates, baseline_decisions)
    health_accepted = _accepted_trades(head_health_candidates, health_decisions)
    start = pd.Timestamp(args.report_start, tz="UTC")
    end = pd.Timestamp(args.report_end, tz="UTC")

    weekly_global = pd.concat(
        [
            _metric_rows(baseline_accepted, variant="baseline_portfolio", start=start, end=end),
            _metric_rows(health_accepted, variant="head_health_portfolio", start=start, end=end),
        ],
        ignore_index=True,
    )
    weekly_global = _complete_zero_trade_weeks(
        weekly_global,
        candidates=candidates,
        variants=["baseline_portfolio", "head_health_portfolio"],
        start=start,
        end=end,
    )
    weekly_by_head = pd.concat(
        [
            _metric_rows(
                baseline_accepted,
                variant="baseline_portfolio",
                start=start,
                end=end,
                group_cols=["head"],
            ),
            _metric_rows(
                health_accepted,
                variant="head_health_portfolio",
                start=start,
                end=end,
                group_cols=["head"],
            ),
        ],
        ignore_index=True,
    )
    weekly_by_head = _complete_zero_trade_weeks(
        weekly_by_head,
        candidates=candidates,
        variants=["baseline_portfolio", "head_health_portfolio"],
        start=start,
        end=end,
        group_col="head",
    )
    baseline_accepted.to_parquet(args.output_dir / "baseline_accepted_trades.parquet", index=False)
    health_accepted.to_parquet(args.output_dir / "head_health_accepted_trades.parquet", index=False)
    baseline_decisions.to_parquet(args.output_dir / "baseline_decisions.parquet", index=False)
    health_decisions.to_parquet(args.output_dir / "head_health_decisions.parquet", index=False)
    baseline_equity.to_parquet(args.output_dir / "baseline_equity_curve.parquet", index=False)
    health_equity.to_parquet(args.output_dir / "head_health_equity_curve.parquet", index=False)
    candidates.to_parquet(args.output_dir / "combined_candidates.parquet", index=False)
    head_health_candidates.to_parquet(args.output_dir / "head_health_adjusted_candidates.parquet", index=False)
    weekly_global.to_csv(args.output_dir / "weekly_global_metrics.csv", index=False)
    weekly_by_head.to_csv(args.output_dir / "weekly_by_head_metrics.csv", index=False)

    manifest = {
        "generated_by": "run_head_health_portfolio_manager_may_june_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_type": "realised_native_simple_policy_portfolio_replay",
        "costs_included": True,
        "market_mode": args.market_mode,
        "report_start": start.isoformat(),
        "report_end": end.isoformat(),
        "candidate_paths": [str(path) for path in candidate_paths],
        "candidate_sha256": {str(path): _file_sha256(path) for path in candidate_paths},
        "train_candidates": str(args.train_candidates),
        "train_candidates_sha256": _file_sha256(args.train_candidates),
        "portfolio_manifest": str(args.portfolio_manifest),
        "portfolio_manifest_sha256": _file_sha256(args.portfolio_manifest),
        "portfolio_variant": args.portfolio_variant,
        "head_health_config": str(args.head_health_config),
        "head_health_config_sha256": _file_sha256(args.head_health_config),
        "candidate_rows": int(len(candidates)),
        "candidate_timestamp_min": pd.to_datetime(candidates["timestamp"], utc=True).min().isoformat(),
        "candidate_timestamp_max": pd.to_datetime(candidates["timestamp"], utc=True).max().isoformat(),
        "baseline_replay_metrics_all_rows": baseline_metrics,
        "head_health_replay_metrics_all_rows": health_metrics,
        "outputs": {
            "weekly_global": str(args.output_dir / "weekly_global_metrics.csv"),
            "weekly_by_head": str(args.output_dir / "weekly_by_head_metrics.csv"),
            "baseline_accepted": str(args.output_dir / "baseline_accepted_trades.parquet"),
            "head_health_accepted": str(args.output_dir / "head_health_accepted_trades.parquet"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(weekly_global.to_string(index=False))
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
