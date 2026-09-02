#!/usr/bin/env python3
"""Materialize optimized side/archetype exits and replay the global auction.

The simple-policy optimizer historically persisted aggregate diagnostics only.
This utility applies its saved parent and shrunk local geometries row by row so
the downstream portfolio replay consumes the optimized execution outcomes,
rather than label first-touch proxy outcomes.
"""

from __future__ import annotations

import argparse
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

os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")
os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "0")
os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "1")

from scripts.ablate_simple_policy_exit_geometry import _load_bundles, _prepare_rows  # noqa: E402
from scripts.run_s52_side_archetype_simple_policy_optimiser import (  # noqa: E402
    _geometry_params_from_archetype_row,
    _params_from_parent_summary_row,
)
from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    load_portfolio_policy_params,
    normalise_candidate_table,
    replay_candidates,
)
from extreme_price_movements.regime_ev_calibration import (  # noqa: E402
    default_regime_ev_calibration_artifact,
    default_regime_ev_feature_handoff,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _attach_policy_archetype_column,
    _with_policy_spread_cost_columns,
    simulate_and_score,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _policy_summary_path(policy_dir: Path, stem: str) -> Path:
    canonical = policy_dir / f"{stem}.csv"
    if canonical.exists():
        return canonical
    promoted = policy_dir / f"best_{stem}.csv"
    if promoted.exists():
        return promoted
    raise FileNotFoundError(
        f"Missing {stem}.csv or best_{stem}.csv under {policy_dir}"
    )


def _apply_policy_spread_to_returns(rows: pd.DataFrame) -> pd.DataFrame:
    """Record that executable prices already contain entry and exit spread.

    ``simulate_and_score`` reanchors entry prices and every exit fill using the
    attached half-spreads. Its gross return is therefore already spread-aware.
    Deducting the diagnostic spread columns again here double-counts spread.
    """
    if rows.empty:
        return rows
    out = rows.copy()
    out["policy_spread_applied_to_returns"] = False
    out["policy_spread_embedded_in_executable_prices"] = True
    return out


def _policy_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["strategy_id"]), str(row["policy_archetype"])): row
        for row in frame.to_dict("records")
    }


def _materialize_exit_rows(
    bundles: list[Any],
    *,
    parent_summary: pd.DataFrame,
    archetype_summary: pd.DataFrame,
    cost_pct: float,
) -> pd.DataFrame:
    parent_lookup = {
        str(row["strategy_id"]): row for row in parent_summary.to_dict("records")
    }
    local_lookup = _policy_lookup(archetype_summary)
    output: list[pd.DataFrame] = []

    for bundle in bundles:
        strategy_id = str(bundle.strategy_id)
        parent_row = parent_lookup.get(strategy_id)
        if parent_row is None:
            raise KeyError(f"Missing parent policy for {strategy_id}")
        parent_params, parent_size = _params_from_parent_summary_row(parent_row)
        work = _attach_policy_archetype_column(
            bundle.rows.copy(), strategy_id=strategy_id
        ).reset_index(drop=True)

        for local_key, indices in work.groupby("policy_archetype", sort=True).groups.items():
            idx = np.asarray(list(indices), dtype=np.int64)
            sub = work.iloc[idx].copy().reset_index(drop=True)
            paths = tuple(path[idx] for path in bundle.paths)
            local_row = local_lookup.get((strategy_id, str(local_key)))
            if local_row is None:
                params, size_power = parent_params, parent_size
                policy_source = "side_parent_fallback"
            else:
                params, size_power = _geometry_params_from_archetype_row(
                    local_row,
                    parent_params=parent_params,
                    parent_size_power=parent_size,
                )
                policy_source = "side_archetype_shrunk_geometry"

            # Portfolio competition is applied after geometry replay. Disable
            # the simulator's internal concurrency filter here to avoid applying
            # capacity limits twice.
            sim_params = dict(params)
            sim_params.pop("max_concurrent_trades", None)
            sim_params.pop("max_concurrent_per_asset", None)
            metrics = simulate_and_score(
                sub,
                *paths,
                cost_pct=float(cost_pct),
                size_power=float(size_power),
                max_concurrent_trades=max(len(sub), 1),
                max_concurrent_per_asset=max(len(sub), 1),
                **sim_params,
            )
            mask = np.asarray(metrics.get("selected_mask", []), dtype=bool)
            if mask.size != len(sub):
                raise ValueError(
                    f"Selected-mask mismatch for {strategy_id}/{local_key}: "
                    f"{mask.size} != {len(sub)}"
                )
            selected = sub.iloc[np.flatnonzero(mask)].copy().reset_index(drop=True)
            sizes = np.asarray(metrics.get("sizes", []), dtype=np.float64)
            net_gain = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
            gross_gain = np.asarray(metrics.get("gross_gains", []), dtype=np.float64)
            exit_bars = np.asarray(metrics.get("exit_bars", []), dtype=np.int64)
            exit_reason = np.asarray(metrics.get("exit_reason", []), dtype=object)
            entry_prices = np.asarray(metrics.get("entry_prices", []), dtype=np.float64)
            exit_prices = np.asarray(metrics.get("exit_prices", []), dtype=np.float64)
            fee_returns = np.asarray(metrics.get("fee_returns", []), dtype=np.float64)
            expected = len(selected)
            if not all(
                len(v) == expected
                for v in (
                    sizes,
                    net_gain,
                    gross_gain,
                    exit_bars,
                    exit_reason,
                    entry_prices,
                    exit_prices,
                    fee_returns,
                )
            ):
                raise ValueError(f"Simulation output mismatch for {strategy_id}/{local_key}")
            denom = np.where(np.abs(sizes) > 1e-12, sizes, np.nan)
            selected["net_return"] = net_gain / denom
            selected["gross_return"] = gross_gain / denom
            selected["fee_return"] = fee_returns
            selected["entry_price"] = entry_prices
            selected["exit_price"] = exit_prices
            selected["policy_size_multiplier"] = sizes
            selected["holding_bars"] = exit_bars
            selected["exit_timestamp"] = pd.to_datetime(
                selected["timestamp"], utc=True, errors="coerce"
            ) + pd.to_timedelta(exit_bars * 15, unit="m")
            selected["simple_policy_exit_reason"] = exit_reason.astype(str)
            selected["execution_policy_source"] = policy_source
            selected["execution_policy_key"] = str(local_key)
            selected["policy_spread_embedded_in_executable_prices"] = True
            output.append(selected)

    if not output:
        return pd.DataFrame()
    rows = pd.concat(output, ignore_index=True, copy=False)
    rows = rows.sort_values(["timestamp", "symbol", "strategy_id"], kind="stable")
    return rows.reset_index(drop=True)


def _load_label_geometry(path: Path) -> dict[str, Mapping[str, Any]]:
    frame = pd.read_csv(path)
    required = {"policy_key", "tp_r", "sl_r", "trail_r", "max_bars_to_mfe"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Label geometry manifest is missing {sorted(missing)}")
    return {
        str(row["policy_key"]): row
        for row in frame.to_dict("records")
    }


def _materialize_label_exit_rows(
    bundles: list[Any],
    *,
    label_geometry: Mapping[str, Mapping[str, Any]],
    cost_pct: float,
) -> pd.DataFrame:
    """Replay the materialized label geometry on the executable path engine."""
    output: list[pd.DataFrame] = []
    for bundle in bundles:
        strategy_id = str(bundle.strategy_id)
        work = _attach_policy_archetype_column(
            bundle.rows.copy(), strategy_id=strategy_id
        ).reset_index(drop=True)
        for local_key, indices in work.groupby("archetype_policy_key", sort=True).groups.items():
            idx = np.asarray(list(indices), dtype=np.int64)
            sub = work.iloc[idx].copy().reset_index(drop=True)
            paths = tuple(path[idx] for path in bundle.paths)
            geometry = label_geometry.get(str(local_key))
            if geometry is None:
                raise KeyError(f"Missing label geometry for {strategy_id}/{local_key}")
            params = {
                "sl_mult": float(geometry["sl_r"]),
                "sl_abs_cap_pct": 0.0,
                "trailing_activation_mult": float(geometry["tp_r"]),
                "trailing_activation_cap_pct": 0.0,
                "trailing_activation_decay_half_life_bars": 0.0,
                "trailing_activation_decay_start_bars": 0,
                "trailing_activation_min_mult": 1.0,
                "trailing_activation_max_bars": int(geometry["max_bars_to_mfe"]),
                "fixed_trailing_gap_mult": float(geometry["trail_r"]),
                "capital_protect_mfe_mult": 0.0,
                "adverse_exit_enabled": False,
                "atr_power": 1.0,
                "atr_multiplier": 1.0,
                "hard_tp_abs_pct": 0.0,
            }
            metrics = simulate_and_score(
                sub,
                *paths,
                cost_pct=float(cost_pct),
                size_power=1.0,
                max_concurrent_trades=max(len(sub), 1),
                max_concurrent_per_asset=max(len(sub), 1),
                **params,
            )
            selected_mask = np.asarray(metrics.get("selected_mask", []), dtype=bool)
            selected = sub.iloc[np.flatnonzero(selected_mask)].copy().reset_index(drop=True)
            sizes = np.asarray(metrics.get("sizes", []), dtype=np.float64)
            net_gain = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
            gross_gain = np.asarray(metrics.get("gross_gains", []), dtype=np.float64)
            exit_bars = np.asarray(metrics.get("exit_bars", []), dtype=np.int64)
            exit_reason = np.asarray(metrics.get("exit_reason", []), dtype=object)
            expected = len(selected)
            if not all(len(v) == expected for v in (sizes, net_gain, gross_gain, exit_bars, exit_reason)):
                raise ValueError(f"Label simulation output mismatch for {strategy_id}/{local_key}")
            denom = np.where(np.abs(sizes) > 1e-12, sizes, np.nan)
            selected["net_return"] = net_gain / denom
            selected["gross_return"] = gross_gain / denom
            selected["policy_size_multiplier"] = sizes
            selected["holding_bars"] = exit_bars
            selected["exit_timestamp"] = pd.to_datetime(
                selected["timestamp"], utc=True, errors="coerce"
            ) + pd.to_timedelta(exit_bars * 15, unit="m")
            selected["simple_policy_exit_reason"] = exit_reason.astype(str)
            selected["execution_policy_source"] = "materialized_label_geometry"
            selected["execution_policy_key"] = str(local_key)
            output.append(selected)
    if not output:
        return pd.DataFrame()
    return (
        pd.concat(output, ignore_index=True, copy=False)
        .sort_values(["timestamp", "symbol", "strategy_id"], kind="stable")
        .reset_index(drop=True)
    )


def _paired_geometry_report(
    label_rows: pd.DataFrame,
    optimized_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["timestamp", "symbol", "strategy_id"]
    keep = keys + [
        "side_name",
        "archetype_policy_key",
        "net_return",
        "gross_return",
        "holding_bars",
        "simple_policy_exit_reason",
    ]
    paired = label_rows[keep].merge(
        optimized_rows[keep],
        on=keys,
        how="inner",
        suffixes=("_label", "_optimized"),
        validate="one_to_one",
    )
    paired["delta_net_return"] = paired["net_return_optimized"] - paired["net_return_label"]
    paired["month"] = pd.to_datetime(paired["timestamp"], utc=True).dt.strftime("%Y-%m")
    paired["week"] = (
        pd.to_datetime(paired["timestamp"], utc=True)
        .dt.to_period("W")
        .astype(str)
    )
    records: list[dict[str, Any]] = []
    group_specs = {
        "overall": [],
        "month": ["month"],
        "week": ["week"],
        "side": ["side_name_label"],
        "archetype": ["side_name_label", "archetype_policy_key_label"],
        "label_exit": ["simple_policy_exit_reason_label"],
        "optimized_exit": ["simple_policy_exit_reason_optimized"],
    }
    for scope, columns in group_specs.items():
        grouped = [((), paired)] if not columns else paired.groupby(columns, dropna=False, sort=True)
        for values, group in grouped:
            if not isinstance(values, tuple):
                values = (values,)
            row: dict[str, Any] = {"scope": scope, "rows": int(len(group))}
            row.update({column: value for column, value in zip(columns, values, strict=False)})
            row.update(
                {
                    "label_net_return": float(group["net_return_label"].mean()),
                    "optimized_net_return": float(group["net_return_optimized"].mean()),
                    "delta_net_return": float(group["delta_net_return"].mean()),
                    "label_positive_rate": float((group["net_return_label"] > 0.0).mean()),
                    "optimized_positive_rate": float((group["net_return_optimized"] > 0.0).mean()),
                    "label_timeout_rate": float(group["simple_policy_exit_reason_label"].eq("timeout").mean()),
                    "optimized_timeout_rate": float(group["simple_policy_exit_reason_optimized"].eq("timeout").mean()),
                    "label_full_sl_rate": float(group["simple_policy_exit_reason_label"].eq("full_sl").mean()),
                    "optimized_full_sl_rate": float(group["simple_policy_exit_reason_optimized"].eq("full_sl").mean()),
                }
            )
            records.append(row)
    return paired, pd.DataFrame(records)


def _load_ev_curve(path: Path) -> dict[str, Any]:
    ref = pd.read_parquet(path)
    rename = {}
    if "rank_pct" not in ref and "normalized_rank_score" in ref:
        rename["normalized_rank_score"] = "rank_pct"
    if "ret_net_notional" not in ref and "net_return" in ref:
        rename["net_return"] = "ret_net_notional"
    ref = ref.rename(columns=rename)
    required = {"rank_pct", "ret_net_notional"}
    missing = required.difference(ref.columns)
    if missing:
        raise ValueError(f"EV reference is missing {sorted(missing)}")
    ref["normalized_rank_score"] = pd.to_numeric(ref["rank_pct"], errors="coerce")
    ref["net_return"] = pd.to_numeric(ref["ret_net_notional"], errors="coerce")
    if "timestamp" not in ref:
        ref["timestamp"] = pd.Timestamp("2026-01-01", tz="UTC")
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], utc=True, errors="coerce")
    ref["base_strategy_threshold"] = 0.0
    ref["entry_price"] = 1.0
    ref["exit_price"] = 1.0
    ref["exit_timestamp"] = ref["timestamp"] + pd.Timedelta(minutes=15)
    ref["gross_return"] = ref["net_return"] + 0.01
    ref["holding_bars"] = 1
    ref["simple_policy_exit_reason"] = "historical_reference"
    return fit_hierarchical_ev_curves(ref)


def _portfolio_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    side = pd.to_numeric(rows["side"], errors="coerce").to_numpy(dtype=np.float64)
    gross = pd.to_numeric(rows["gross_return"], errors="coerce").to_numpy(dtype=np.float64)
    entry = np.ones(len(rows), dtype=np.float64)
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(rows["timestamp"], utc=True, errors="coerce"),
            "symbol": rows["symbol"].astype(str),
            "side": side,
            "side_name": rows["side_name"].astype(str),
            "strategy_id": rows["strategy_id"].astype(str),
            "policy_archetype": rows["side_name"].astype(str)
            + "__"
            + rows["archetype_policy_key"].astype(str),
            "local_side_archetype": rows["side_name"].astype(str)
            + "__"
            + rows["archetype_policy_key"].astype(str),
            "normalized_rank_score": pd.to_numeric(rows["rank_pct"], errors="coerce"),
            "strategy_rank_pct": pd.to_numeric(rows["rank_pct"], errors="coerce"),
            "base_strategy_threshold": 0.0,
            "calibrated_score": pd.to_numeric(rows["calibrated_score"], errors="coerce"),
            "entry_price": entry,
            "exit_timestamp": pd.to_datetime(rows["exit_timestamp"], utc=True, errors="coerce"),
            "exit_price": entry + side * gross,
            "net_return": pd.to_numeric(rows["net_return"], errors="coerce"),
            "gross_return": gross,
            "holding_bars": pd.to_numeric(rows["holding_bars"], errors="coerce"),
            "simple_policy_exit_reason": rows["simple_policy_exit_reason"].astype(str),
            "fees_bps": 100.0,
            "slippage_bps": 0.0,
            "expected_friction_bps": 100.0,
            "price_gap_bps": 0.0,
            "liquidity_capacity_weight": 1.0,
            "archetype_policy_key": rows["archetype_policy_key"].astype(str),
        }
    )
    return normalise_candidate_table(out)


def _write_replay_breakdowns(
    decisions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Persist accepted-trade metrics using the replay's net return contract."""
    accepted = decisions.loc[decisions["accepted"].fillna(False)].copy()
    candidate_context = candidates.reset_index(drop=True).reset_index(
        names="candidate_index"
    )
    context_columns = [
        column
        for column in (
            "candidate_index",
            "side_name",
            "policy_archetype",
            "local_side_archetype",
            "archetype_policy_key",
        )
        if column in candidate_context.columns
    ]
    accepted = accepted.merge(
        candidate_context[context_columns],
        on="candidate_index",
        how="left",
        validate="one_to_one",
    )
    accepted["timestamp"] = pd.to_datetime(
        accepted["timestamp"], utc=True, errors="coerce"
    )
    accepted["day"] = accepted["timestamp"].dt.floor("D")
    accepted["week_start"] = (
        accepted["day"]
        - pd.to_timedelta(accepted["day"].dt.weekday, unit="D")
    )
    accepted["net_return"] = pd.to_numeric(
        accepted["position_net_return"], errors="coerce"
    )
    accepted["gross_return"] = pd.to_numeric(
        accepted["position_gross_return"], errors="coerce"
    )
    accepted["notional"] = pd.to_numeric(
        accepted["position_size"], errors="coerce"
    )
    accepted["net_pnl"] = accepted["notional"] * accepted["net_return"]
    accepted["positive"] = accepted["net_return"] > 0.0
    accepted["full_sl"] = accepted["position_exit_reason"].astype(str).eq("full_sl")
    accepted["timeout"] = accepted["position_exit_reason"].astype(str).eq("timeout")

    def aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
        grouped = frame.groupby(group_columns, dropna=False, observed=True)
        report = grouped.agg(
            trades=("net_return", "size"),
            net_ev_per_trade=("net_return", "mean"),
            gross_return_per_trade=("gross_return", "mean"),
            positive_rate=("positive", "mean"),
            full_sl_rate=("full_sl", "mean"),
            timeout_rate=("timeout", "mean"),
            mean_notional=("notional", "mean"),
            net_pnl=("net_pnl", "sum"),
        ).reset_index()
        return report

    reports = {
        "daily_metrics.csv": aggregate(accepted, ["day"]),
        "weekly_metrics.csv": aggregate(accepted, ["week_start"]),
        "side_metrics.csv": aggregate(accepted, ["side_name"]),
        "side_archetype_metrics.csv": aggregate(
            accepted, ["side_name", "archetype_policy_key"]
        ),
    }
    for filename, report in reports.items():
        report.to_csv(output_dir / filename, index=False)

    elapsed_days = max(
        (accepted["timestamp"].max() - accepted["timestamp"].min()).total_seconds()
        / 86_400.0,
        1.0,
    ) if len(accepted) else 1.0
    metrics = {
        "accepted_trades": int(len(accepted)),
        "trades_per_day": float(len(accepted) / elapsed_days),
        "net_ev_per_trade": float(accepted["net_return"].mean()),
        "gross_return_per_trade": float(accepted["gross_return"].mean()),
        "positive_rate": float(accepted["positive"].mean()),
        "full_sl_rate": float(accepted["full_sl"].mean()),
        "timeout_rate": float(accepted["timeout"].mean()),
        "net_pnl": float(accepted["net_pnl"].sum()),
        "mean_notional": float(accepted["notional"].mean()),
    }
    (output_dir / "replay_metrics_summary.json").write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--portfolio-ev-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--round-trip-cost-pct", type=float, default=0.01)
    parser.add_argument(
        "--label-geometry-manifest",
        type=Path,
        default=None,
        help="Optional side_archetype_label_manifest.csv for a same-path comparator.",
    )
    parser.add_argument(
        "--reuse-exit-rows",
        action="store_true",
        help="Reuse output-dir/exit_policy_rows.parquet and run only portfolio replay.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exit_path = args.output_dir / "exit_policy_rows.parquet"
    if args.reuse_exit_rows:
        if not exit_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing exit rows: {exit_path}")
        exit_rows = pd.read_parquet(exit_path)
        input_rows = len(pd.read_parquet(args.candidates, columns=["timestamp"]))
    else:
        rows = _prepare_rows(
            args.candidates,
            min_rank=0.0,
            rank_score_col="rank_pct",
            rank_scope="per_strategy",
            apply_regime_ev_calibration_artifact=False,
        )
        input_rows = len(rows)
        bundles = _load_bundles(
            rows,
            data_root=str(args.data_root),
            market_mode="perps",
            path_len=int(args.path_len),
            min_rows_per_strategy=5,
        )
        parent = pd.read_csv(
            _policy_summary_path(args.policy_dir, "side_parent_policy_summary")
        )
        local = pd.read_csv(
            _policy_summary_path(args.policy_dir, "side_archetype_policy_summary")
        )
        exit_rows = _materialize_exit_rows(
            bundles,
            parent_summary=parent,
            archetype_summary=local,
            cost_pct=float(args.round_trip_cost_pct) / 2.0,
        )
        if args.label_geometry_manifest is not None:
            label_rows = _materialize_label_exit_rows(
                bundles,
                label_geometry=_load_label_geometry(args.label_geometry_manifest),
                cost_pct=float(args.round_trip_cost_pct) / 2.0,
            )
            label_rows = _with_policy_spread_cost_columns(label_rows, market_mode="perps")
            label_rows = _apply_policy_spread_to_returns(label_rows)
            paired, comparison = _paired_geometry_report(label_rows, exit_rows)
            label_rows.to_parquet(
                args.output_dir / "label_geometry_exit_rows.parquet",
                index=False,
                compression="zstd",
            )
            paired.to_parquet(
                args.output_dir / "paired_label_vs_optimized_rows.parquet",
                index=False,
                compression="zstd",
            )
            comparison.to_csv(
                args.output_dir / "paired_label_vs_optimized_metrics.csv",
                index=False,
            )
    exit_rows = _with_policy_spread_cost_columns(exit_rows, market_mode="perps")
    exit_rows = _apply_policy_spread_to_returns(exit_rows)
    exit_rows.to_parquet(exit_path, index=False, compression="zstd")

    candidates = _portfolio_candidates(exit_rows)
    params = load_portfolio_policy_params(args.portfolio_config)
    decisions, equity, summary = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=_load_ev_curve(args.portfolio_ev_reference),
        market_mode="perps",
    )
    decisions.to_parquet(
        args.output_dir / "portfolio_decisions_after_exit_policy.parquet",
        index=False,
        compression="zstd",
    )
    replay_breakdowns = _write_replay_breakdowns(
        decisions,
        candidates,
        output_dir=args.output_dir,
    )
    equity.to_parquet(
        args.output_dir / "portfolio_equity_after_exit_policy.parquet",
        index=False,
        compression="zstd",
    )
    label_portfolio_summary: dict[str, Any] | None = None
    label_exit_path = args.output_dir / "label_geometry_exit_rows.parquet"
    if label_exit_path.exists():
        label_exit_rows = pd.read_parquet(label_exit_path)
        label_decisions, label_equity, label_portfolio_summary = replay_candidates(
            _portfolio_candidates(label_exit_rows),
            params,
            mode="global_auction",
            ev_curve=_load_ev_curve(args.portfolio_ev_reference),
            market_mode="perps",
        )
        label_decisions.to_parquet(
            args.output_dir / "portfolio_decisions_label_geometry.parquet",
            index=False,
            compression="zstd",
        )
        label_equity.to_parquet(
            args.output_dir / "portfolio_equity_label_geometry.parquet",
            index=False,
            compression="zstd",
        )
    manifest = {
        "schema": "canonical_exit_policy_portfolio_replay_v1",
        "source_candidates": str(args.candidates),
        "policy_dir": str(args.policy_dir),
        "portfolio_config": str(args.portfolio_config),
        "portfolio_ev_reference": str(args.portfolio_ev_reference),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "fee_contract": "1% round trip applied once by simulate_and_score",
        "spread_contract": (
            "Per-symbol Kraken baseline full spread embedded once through executable "
            "entry/exit prices; diagnostic spread columns are not deducted again."
        ),
        "mean_expected_spread_bps": float(
            pd.to_numeric(exit_rows["expected_spread_bps"], errors="coerce").mean()
        ),
        "median_expected_spread_bps": float(
            pd.to_numeric(exit_rows["expected_spread_bps"], errors="coerce").median()
        ),
        "p90_expected_spread_bps": float(
            pd.to_numeric(exit_rows["expected_spread_bps"], errors="coerce").quantile(0.90)
        ),
        "internal_geometry_concurrency": "disabled; global_auction_v1 applies capacity once",
        "input_rows": int(input_rows),
        "exit_rows": int(len(exit_rows)),
        "path_survival": float(len(exit_rows) / max(input_rows, 1)),
        "portfolio_summary": summary,
        "replay_breakdowns": replay_breakdowns,
        "label_geometry_portfolio_summary": label_portfolio_summary,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
