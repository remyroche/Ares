#!/usr/bin/env python3
"""Replay the promoted July policy on causal one-minute executable paths.

The input frontier is a historical policy candidate set, not a production-trade
ledger. Hourly feature timestamps identify the opening time of the completed
signal candle, so execution begins at the following hour. This script applies
the promoted exit geometry, frozen Bayesian sizing, and the deployed
capital-aware global auction. Unresolved paths exit at the last close in the
requested horizon; they are never booked as full losses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    load_portfolio_policy_params,
    replay_candidates,
)
from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from extreme_price_movements.simple_policy_winner import (  # noqa: E402
    apply_raw_bayesian_sizing_state,
)
from scripts.report_simple_policy_1m_winner_forward_july import (  # noqa: E402
    _forward_context,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    ExperimentData,
    _causal_entry_atr,
)


DEFAULT_ARTIFACT = Path(
    "data_perp/artifacts/"
    "s59_s52_base66_residualstate_meta_v9tail95_mlp_hierev_ev70_geometry_20260717_v2"
)
DEFAULT_CHAMPION = Path(
    "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1"
)
DEFAULT_BASE = Path(
    "data_perp/reports/meta_v9_recovery_20260717/"
    "residual_state_mda95_hier_newaegmm_downstream_retrain_v1"
)
DEFAULT_OLD = Path(
    "data_perp/reports/july_01_16_current_policy_metrics_20260717/"
    "current_policy_candidates_through_july16.parquet"
)
DEFAULT_FORWARD = Path(
    "data_perp/reports/july_1m_replay_v9_hierev_nomlp_20260720/"
    "forward_8h_candidate_frontier.parquet"
)
DEFAULT_FORWARD_CONTEXT = Path(
    "data_perp/reports/july_1m_replay_v9_hierev_nomlp_20260720/"
    "forward_8h_sizing_context.parquet"
)
DEFAULT_STORE = Path("data_perp/exchanges/krakenfutures/execution_1m")


REASON_NAMES = {
    0: "timeout",
    1: "full_stop",
    2: "capital_protection",
    3: "trailing",
    4: "adverse_path",
}


def _parent_strategy(payload: dict, side: str) -> dict:
    key = f"{side}__parent"
    for strategy in payload.get("strategies", []):
        if str(strategy.get("canonical_strategy_id")) == key:
            return strategy
    raise KeyError(f"Missing {key} in policy artifact")


def _side_params(payload: dict) -> dict[str, dict]:
    return {side: _parent_strategy(payload, side) for side in ("long", "short")}


def _old_context(rows: pd.DataFrame) -> pd.DataFrame:
    direct = {
        "expected_net_ev_after_1pct": "expected_net_ev_after_1pct_mlp_direct",
        "meta_hit_probability_uncertainty_p1mp": "meta_hit_probability_uncertainty_p1mp",
        "gmm_ood_score": "gmm_ood_score",
        "cluster_entropy_norm": "cluster_entropy_norm",
    }
    if all(column in rows.columns for column in direct):
        context = rows[list(direct)].rename(columns=direct).copy()
        numeric = context.apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=np.float64, copy=False)).all():
            raise RuntimeError("Materialized candidate sizing context is non-finite")
        return numeric
    context, _ = _forward_context(rows)
    return context.rename(
        columns={"expected_net_ev_after_1pct_mlp_direct": "expected_net_ev_after_1pct"}
    )


def _sizing_frame(rows: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    out = rows[["timestamp", "symbol", "side_name", "policy_archetype", "rank_pct"]].copy()
    for column in context.columns:
        out[column] = context[column].to_numpy(copy=False)
    corrected = pd.to_numeric(
        rows.get("threshold_basis_corrected_expected_ev"), errors="coerce"
    )
    base_ev = pd.to_numeric(out.get("expected_net_ev_after_1pct"), errors="coerce")
    out["expected_net_ev_after_1pct"] = corrected.where(corrected.notna(), base_ev)
    if "meta_hit_probability_uncertainty_p1mp" not in out:
        rank = pd.to_numeric(rows["rank_pct"], errors="coerce").clip(0.0, 1.0)
        out["meta_hit_probability_uncertainty_p1mp"] = rank * (1.0 - rank)
    return out


def _portfolio_candidates(
    rows: pd.DataFrame,
    outputs: dict[str, np.ndarray],
    size_multiplier: np.ndarray,
) -> pd.DataFrame:
    out = rows.copy().reset_index(drop=True)
    rank = pd.to_numeric(
        out.get("threshold_basis_corrected_expected_ev_rank"), errors="coerce"
    )
    fallback = pd.to_numeric(out["rank_pct"], errors="coerce")
    out["normalized_rank_score"] = rank.where(rank.notna(), fallback).clip(0.0, 1.0)
    out["base_strategy_threshold"] = 0.90
    out["entry_price"] = np.asarray(outputs["entry_price"], dtype=float)
    out["exit_price"] = np.asarray(outputs["exit_price"], dtype=float)
    out["gross_return"] = np.asarray(outputs["gross_return"], dtype=float)
    out["net_return"] = np.asarray(outputs["net_return"], dtype=float)
    out["holding_bars"] = np.asarray(outputs["exit_bars"], dtype=np.int32) + 1
    out["exit_timestamp"] = pd.to_datetime(out["timestamp"], utc=True) + pd.to_timedelta(
        out["holding_bars"], unit="min"
    )
    out["simple_policy_exit_reason"] = [
        REASON_NAMES.get(int(value), f"unknown_{int(value)}") for value in outputs["reason"]
    ]
    out["fees_bps"] = 100.0
    out["price_gap_bps"] = 0.0
    out["expected_friction_bps"] = 100.0 + pd.to_numeric(
        out.get("expected_spread_bps", 0.0), errors="coerce"
    ).fillna(0.0)
    out["liquidity_capacity_weight"] = 1.0
    out["portfolio_size_multiplier"] = np.asarray(size_multiplier, dtype=float)
    out["portfolio_rank_size_power"] = 1.5
    return out


def _daily_table(
    decisions: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"]].copy()
    day_source = "signal_bar_ts" if "signal_bar_ts" in accepted else "timestamp"
    accepted["day"] = pd.to_datetime(accepted[day_source], utc=True).dt.floor("D")
    accepted["notional_pnl"] = accepted["position_size"] * accepted["position_net_return"]
    accepted["gross_notional_pnl"] = (
        accepted["position_size"] * accepted["position_gross_return"]
    )
    candidates = candidates.copy()
    day_source = "signal_bar_ts" if "signal_bar_ts" in candidates else "timestamp"
    candidates["day"] = pd.to_datetime(candidates[day_source], utc=True).dt.floor("D")
    days = pd.date_range(start.floor("D"), end_exclusive.floor("D"), freq="D", inclusive="left")
    candidate_count = candidates.groupby("day").size()
    grouped = accepted.groupby("day")
    records = []
    for day in days:
        group = grouped.get_group(day) if day in grouped.groups else accepted.iloc[:0]
        records.append(
            {
                "day": day,
                "candidate_rows": int(candidate_count.get(day, 0)),
                "accepted_trades": int(len(group)),
                "trades_per_day": float(len(group)),
                "net_return_per_trade": float(group["position_net_return"].mean()) if len(group) else np.nan,
                "gross_return_per_trade": float(group["position_gross_return"].mean()) if len(group) else np.nan,
                "positive_trade_rate": float((group["position_net_return"] > 0).mean()) if len(group) else np.nan,
                "net_pnl": float(group["notional_pnl"].sum()),
                "gross_pnl": float(group["gross_notional_pnl"].sum()),
                "mean_position_notional": float(group["position_size"].mean()) if len(group) else np.nan,
                "full_stop_rate": float((group["position_exit_reason"] == "full_stop").mean()) if len(group) else np.nan,
                "trailing_rate": float((group["position_exit_reason"] == "trailing").mean()) if len(group) else np.nan,
                "timeout_rate": float((group["position_exit_reason"] == "timeout").mean()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--base-report", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--old-candidates", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--forward-candidates", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--forward-context", type=Path, default=DEFAULT_FORWARD_CONTEXT)
    parser.add_argument("--all-candidates", type=Path, default=None)
    parser.add_argument("--all-context", type=Path, default=None)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--horizon-minutes", type=int, default=1_440)
    parser.add_argument("--start", default="2026-07-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-07-21T00:00:00Z")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/july_1m_replay_v9_hierev_nomlp_20260720"),
    )
    parser.add_argument("--rebuild-path-cache", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_candidates is not None:
        rows = pd.read_parquet(args.all_candidates)
        if args.all_context is not None:
            context = pd.read_parquet(args.all_context)
            if len(context) != len(rows):
                raise RuntimeError("Unified context is not row-aligned with unified candidates")
        else:
            context = _old_context(rows)
    else:
        old = pd.read_parquet(args.old_candidates)
        forward = pd.read_parquet(args.forward_candidates)
        old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
        forward["timestamp"] = pd.to_datetime(forward["timestamp"], utc=True)
        old_context = _old_context(old)
        forward_context = pd.read_parquet(args.forward_context)
        if len(forward_context) != len(forward):
            raise RuntimeError("Forward context is not row-aligned with forward candidates")
        forward_context = forward_context.rename(
            columns={"expected_net_ev_after_1pct_mlp_direct": "expected_net_ev_after_1pct"}
        )
        rows = pd.concat([old, forward], ignore_index=True, copy=False)
        context = pd.concat([old_context, forward_context], ignore_index=True, copy=False)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    order = rows.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).index.to_numpy()
    rows = rows.iloc[order].reset_index(drop=True)
    context = context.iloc[order].reset_index(drop=True)
    keep = ~rows.duplicated(["timestamp", "symbol", "side"], keep="first")
    rows = rows.loc[keep].reset_index(drop=True)
    context = context.loc[keep].reset_index(drop=True)

    # Stored hourly timestamps identify [t, t+1h) signal candles. Starting the
    # executable path at t leaks that candle's future OHLC into the replay.
    rows["signal_bar_ts"] = rows["timestamp"]
    rows["signal_bar_close_ts"] = rows["signal_bar_ts"] + pd.Timedelta(hours=1)
    rows["timestamp"] = rows["signal_bar_close_ts"]
    if not rows["timestamp"].ge(rows["signal_bar_close_ts"]).all():
        raise RuntimeError("Historical replay entry precedes signal-bar close")

    policy_path = args.artifact / "simple_policy_optimiser/deployment/best_policy_params.json"
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    side_params = _side_params(policy_payload)
    parent_summary = args.base_report / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
    deployed, _ = _load_deployed_side_params(parent_summary)
    spec = ConstrainedReplaySpec(horizon_minutes=int(args.horizon_minutes))
    atr, atr_audit, atr_manifest = _causal_entry_atr(
        rows,
        store_root=args.store,
        deployed_by_side=deployed,
        parent_summary=parent_summary,
        warmup_hours=48,
    )
    atr_audit.to_parquet(args.output_dir / "causal_entry_atr_audit_8h.parquet", index=False)
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows,
        store_root=args.store,
        cache_dir=args.output_dir / "path_cache_8h",
        spec=spec,
        rebuild=args.rebuild_path_cache,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)
    valid_idx = np.flatnonzero(data.valid)
    outputs_valid = data.simulate(valid_idx, side_params, FAMILY_TRAILING_ONLY)
    rows_valid = rows.iloc[valid_idx].reset_index(drop=True)
    context_valid = context.iloc[valid_idx].reset_index(drop=True)
    sizing_rows = _sizing_frame(rows_valid, context_valid)
    # Long and short sizing states are fitted independently.  Applying the
    # long state to the combined frame silently mis-sizes every short trade.
    size_multiplier = np.ones(len(sizing_rows), dtype=np.float64)
    sizing_sides = sizing_rows["side_name"].astype("string")
    for side in ("long", "short"):
        side_mask = sizing_sides.eq(side).to_numpy()
        if not side_mask.any():
            continue
        sizing_state = _parent_strategy(policy_payload, side)["raw_bayesian_sizing_state"]
        size_multiplier[side_mask] = apply_raw_bayesian_sizing_state(
            sizing_rows.loc[side_mask].reset_index(drop=True), sizing_state
        )
    outputs_valid["entry_price"] = np.asarray(open0[valid_idx], dtype=float)
    candidates = _portfolio_candidates(rows_valid, outputs_valid, size_multiplier)

    portfolio_config = args.artifact / "policy_params/optimized_portfolio_policy_config.json"
    portfolio_params = load_portfolio_policy_params(portfolio_config)
    policy_mask = rows_valid.get(
        "policy_admitted_before_portfolio", pd.Series(True, index=rows_valid.index)
    ).fillna(False).astype(bool).to_numpy()
    portfolio_candidates = candidates.loc[policy_mask].reset_index(drop=True)
    decisions, equity, summary = replay_candidates(
        portfolio_candidates,
        portfolio_params,
        mode="global_auction",
        initial_wallet=10_000.0,
        market_mode="perps",
    )
    source_columns = [
        column
        for column in (
            "side_name",
            "archetype_label_family",
            "policy_archetype",
            "local_side_archetype",
            "archetype_policy_key",
            "raw_global_top10_selected",
            "ev_mapped_global_top10_selected",
            "policy_admitted_before_portfolio",
        )
        if column in portfolio_candidates.columns
    ]
    source_index = pd.to_numeric(decisions["candidate_index"], errors="raise").astype(np.int64)
    if len(source_index) and (source_index.min() < 0 or source_index.max() >= len(portfolio_candidates)):
        raise RuntimeError("Portfolio decision candidate_index is outside the admitted candidate table")
    decision_context = portfolio_candidates.iloc[source_index.to_numpy()][source_columns].reset_index(drop=True)
    for column in source_columns:
        if column not in decisions.columns:
            decisions[column] = decision_context[column].to_numpy(copy=False)
    decisions["portfolio_net_pnl"] = (
        pd.to_numeric(decisions["position_size"], errors="coerce")
        * pd.to_numeric(decisions["position_net_return"], errors="coerce")
    )
    decisions["portfolio_gross_pnl"] = (
        pd.to_numeric(decisions["position_size"], errors="coerce")
        * pd.to_numeric(decisions["position_gross_return"], errors="coerce")
    )
    start = pd.Timestamp(args.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end_exclusive = pd.Timestamp(args.end_exclusive)
    end_exclusive = (
        end_exclusive.tz_localize("UTC")
        if end_exclusive.tzinfo is None
        else end_exclusive.tz_convert("UTC")
    )
    daily = _daily_table(
        decisions, portfolio_candidates, start=start, end_exclusive=end_exclusive
    )
    decisions.to_parquet(args.output_dir / "portfolio_decisions_8h.parquet", index=False)
    candidates.to_parquet(args.output_dir / "execution_candidate_outcomes_8h.parquet", index=False)
    equity.to_parquet(args.output_dir / "portfolio_equity_8h.parquet", index=False)
    daily.to_csv(args.output_dir / "daily_metrics_8h.csv", index=False)
    manifest = {
        "horizon_minutes": int(args.horizon_minutes),
        "timeout_semantics": "last available minute close with exit spread; 1pct fee charged once",
        "replay_scope": "causal historical policy counterfactual; not actual production admissions",
        "entry_timestamp_contract": "completed hourly signal bar close (signal timestamp plus one hour)",
        "candidate_rows": int(len(rows)),
        "portfolio_candidate_rows": int(len(portfolio_candidates)),
        "valid_path_rows": int(data.valid.sum()),
        "path_coverage": float(data.valid.mean()),
        "accepted_trades": int(decisions["accepted"].sum()),
        "portfolio_summary": summary,
        "path_manifest": path_manifest,
        "atr_manifest": atr_manifest,
        "policy_artifact": str(args.artifact),
        "portfolio_config": str(portfolio_config),
    }
    (args.output_dir / "replay_manifest_8h.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str))
    print(daily.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
