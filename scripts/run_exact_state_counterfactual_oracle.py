#!/usr/bin/env python3
"""Exact-state counterfactual oracle for portfolio actions.

This script evaluates local strategy actions from cloned portfolio states.  It is
intended to validate whether dynamic strategy control has economic headroom
before training another action model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioState,
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    replay_candidates,
)
from scripts.run_global_portfolio_period_multiplier import (  # noqa: E402
    DEFAULT_POLICY_MANIFEST,
    DEFAULT_TRAIN_BROAD,
    DEFAULT_TRAIN_DEPLOYABLE,
    _accepted_trades,
    _json_safe,
    _load_candidates,
    _load_policy_params,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/exact_state_counterfactual_oracle_20260625")
SIZE_ACTIONS = (0.0, 0.50, 0.75, 1.0)
THRESHOLD_UPLIFTS = (0.0, 0.02, 0.05, 0.10)


def _timestamp_sample(timestamps: pd.Series, max_timestamps: int | None) -> pd.DatetimeIndex:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()
    if max_timestamps is None or int(max_timestamps) <= 0 or len(ts) <= int(max_timestamps):
        return pd.DatetimeIndex(ts)
    # Stratified time sample: preserve the first/last points and spread the rest.
    idx = np.linspace(0, len(ts) - 1, int(max_timestamps)).round().astype(int)
    return pd.DatetimeIndex(ts.iloc[np.unique(idx)])


def _load_target_actions(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"timestamp", "strategy_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"target actions are missing required columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    if "action_family" not in out.columns:
        out["action_family"] = "size"
    out["action_family"] = out["action_family"].astype(str)
    if "action_value" not in out.columns:
        out["action_value"] = np.nan
    out["action_value"] = pd.to_numeric(out["action_value"], errors="coerce")
    out = out.drop_duplicates(["timestamp", "strategy_id", "action_family", "action_value"], keep="last")
    return out.reset_index(drop=True)


def _target_strategy_ids_for_timestamp(targets: pd.DataFrame, timestamp: pd.Timestamp, default: list[str]) -> list[str]:
    if targets.empty:
        return list(default)
    rows = targets.loc[targets["timestamp"].eq(pd.Timestamp(timestamp))]
    if rows.empty:
        return []
    return sorted(str(x) for x in rows["strategy_id"].dropna().astype(str).unique())


def _target_action_values(
    targets: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
    strategy_id: str,
    action_family: str,
    default: tuple[float, ...],
) -> list[float]:
    if targets.empty:
        return [float(x) for x in default]
    rows = targets.loc[
        targets["timestamp"].eq(pd.Timestamp(timestamp))
        & targets["strategy_id"].astype(str).eq(str(strategy_id))
        & targets["action_family"].astype(str).eq(str(action_family))
    ]
    if rows.empty:
        return []
    vals = pd.to_numeric(rows["action_value"], errors="coerce").dropna()
    if vals.empty:
        return [float(x) for x in default]
    return sorted(float(x) for x in vals.unique())


def _slice_from_timestamp(candidates: pd.DataFrame, timestamp: pd.Timestamp, horizon_hours: int) -> pd.DataFrame:
    ts = pd.Timestamp(timestamp)
    end = ts + pd.Timedelta(hours=int(horizon_hours))
    work = candidates.loc[(candidates["timestamp"] >= ts) & (candidates["timestamp"] <= end)].copy()
    work.attrs.pop("portfolio_policy_candidates_normalised", None)
    return work


def _apply_local_action(
    candidates: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
    strategy_id: str,
    action_family: str,
    action_value: float,
) -> pd.DataFrame:
    work = candidates.copy()
    mask = work["timestamp"].eq(pd.Timestamp(timestamp)) & work["strategy_id"].astype(str).eq(str(strategy_id))
    if not bool(mask.any()):
        return work
    if action_family == "size":
        base = (
            pd.to_numeric(work.get("portfolio_size_multiplier"), errors="coerce").fillna(1.0)
            if "portfolio_size_multiplier" in work.columns
            else pd.Series(1.0, index=work.index)
        )
        work["portfolio_size_multiplier"] = base
        work.loc[mask, "portfolio_size_multiplier"] = (
            pd.to_numeric(work.loc[mask, "portfolio_size_multiplier"], errors="coerce").fillna(1.0)
            * float(action_value)
        ).clip(lower=0.0, upper=1.0)
    elif action_family == "threshold":
        base = pd.to_numeric(work["base_strategy_threshold"], errors="coerce").fillna(1.0)
        work["base_strategy_threshold"] = base
        work.loc[mask, "base_strategy_threshold"] = (base.loc[mask] + float(action_value)).clip(upper=0.999)
    else:
        raise ValueError(f"Unknown action family: {action_family}")
    work.attrs.pop("portfolio_policy_candidates_normalised", None)
    return work


def _utility(
    accepted: pd.DataFrame,
    *,
    start: pd.Timestamp,
    horizon_hours: int,
    lambda_dd: float,
    lambda_turnover: float,
) -> dict[str, float]:
    if accepted.empty:
        return {
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "cost_pnl": 0.0,
            "turnover": 0.0,
            "worst_trade_pnl": 0.0,
            "J": 0.0,
        }
    end = pd.Timestamp(start) + pd.Timedelta(hours=int(horizon_hours))
    work = accepted.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.loc[(work["timestamp"] >= pd.Timestamp(start)) & (work["timestamp"] <= end)].copy()
    if work.empty:
        return {
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "cost_pnl": 0.0,
            "turnover": 0.0,
            "worst_trade_pnl": 0.0,
            "J": 0.0,
        }
    size = pd.to_numeric(work["position_size"], errors="coerce").fillna(0.0)
    net = size * pd.to_numeric(work["net_return"], errors="coerce").fillna(0.0)
    gross = size * pd.to_numeric(work["gross_return"], errors="coerce").fillna(0.0)
    net_pnl = float(net.sum())
    gross_pnl = float(gross.sum())
    cost_pnl = float(gross_pnl - net_pnl)
    turnover = float(size.sum())
    worst_trade_pnl = float(net.min()) if len(net) else 0.0
    dd_penalty = max(-worst_trade_pnl, 0.0)
    J = net_pnl - float(lambda_dd) * dd_penalty - float(lambda_turnover) * turnover
    return {
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "cost_pnl": cost_pnl,
        "turnover": turnover,
        "worst_trade_pnl": worst_trade_pnl,
        "J": float(J),
    }


def _decision_signature(decisions: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=["symbol", "strategy_id", "accepted", "position_size", "rejection_reason"])
    work = decisions.loc[pd.to_datetime(decisions["timestamp"], utc=True, errors="coerce").eq(pd.Timestamp(timestamp))].copy()
    cols = ["symbol", "strategy_id", "accepted", "position_size", "rejection_reason"]
    return work[cols].sort_values(["strategy_id", "symbol"]).reset_index(drop=True)


def _accepted_at_timestamp(accepted: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    if accepted.empty:
        return accepted.copy()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    return accepted.loc[ts.eq(pd.Timestamp(timestamp))].copy()


def _capture_snapshots(candidates: pd.DataFrame, params: Any, ev_curve: dict[str, Any], market_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, PortfolioState]]:
    snapshots: dict[pd.Timestamp, PortfolioState] = {}

    def callback(ts: pd.Timestamp, state: PortfolioState, _group_idx: np.ndarray, _cache: Any) -> None:
        snapshots[pd.Timestamp(ts)] = state

    decisions, equity, _ = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
        pre_decision_snapshot_callback=callback,
    )
    return decisions, equity, snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-candidates", type=Path, default=DEFAULT_TRAIN_BROAD)
    parser.add_argument("--deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--target-actions",
        type=Path,
        default=None,
        help=(
            "Optional CSV/parquet with timestamp,strategy_id and optional "
            "action_family,action_value. When supplied, evaluate only those "
            "timestamps/strategies/actions."
        ),
    )
    parser.add_argument("--horizon-hours", type=int, default=72)
    parser.add_argument("--max-timestamps", type=int, default=48)
    parser.add_argument("--lambda-dd", type=float, default=0.25)
    parser.add_argument("--lambda-turnover", type=float, default=0.0)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params, policy_payload = _load_policy_params(args.policy_manifest, args.policy_variant)
    broad = normalise_candidate_table(_load_candidates(args.broad_candidates))
    deployable = normalise_candidate_table(_load_candidates(args.deployable_candidates))
    target_actions = _load_target_actions(args.target_actions)
    start = pd.Timestamp(args.start, tz="UTC") if args.start else broad["timestamp"].min()
    end = pd.Timestamp(args.end, tz="UTC") if args.end else broad["timestamp"].max()
    broad = broad.loc[(broad["timestamp"] >= start) & (broad["timestamp"] <= end)].copy()
    deployable_train = deployable.loc[deployable["timestamp"] < start].copy()
    if deployable_train.empty:
        deployable_train = deployable.copy()
    ev_curve = fit_hierarchical_ev_curves(deployable_train)

    if broad.empty:
        raise RuntimeError(f"No candidates in requested interval {start} to {end}")

    baseline_decisions, _baseline_equity, snapshots = _capture_snapshots(
        broad,
        params,
        ev_curve,
        args.market_mode,
    )
    if not target_actions.empty:
        target_actions = target_actions.loc[
            target_actions["timestamp"].ge(start) & target_actions["timestamp"].le(end)
        ].copy()
        sampled_ts = pd.DatetimeIndex(target_actions["timestamp"].dropna().drop_duplicates().sort_values())
    else:
        sampled_ts = _timestamp_sample(broad["timestamp"], args.max_timestamps)
    strategy_ids = sorted(str(x) for x in broad["strategy_id"].dropna().astype(str).unique())

    rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    for ts in sampled_ts:
        timestamp = pd.Timestamp(ts)
        state = snapshots.get(timestamp)
        if state is None:
            continue
        suffix = _slice_from_timestamp(broad, timestamp, int(args.horizon_hours))
        if suffix.empty:
            continue
        base_decisions, _base_eq, _base_metrics = replay_candidates(
            suffix,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            initial_state=state,
            market_mode=args.market_mode,
        )
        base_accepted = _accepted_trades(suffix, base_decisions)
        base_full = _utility(
            base_accepted,
            start=timestamp,
            horizon_hours=int(args.horizon_hours),
            lambda_dd=float(args.lambda_dd),
            lambda_turnover=float(args.lambda_turnover),
        )
        base_immediate = _utility(
            _accepted_at_timestamp(base_accepted, timestamp),
            start=timestamp,
            horizon_hours=int(args.horizon_hours),
            lambda_dd=float(args.lambda_dd),
            lambda_turnover=float(args.lambda_turnover),
        )
        baseline_sig = _decision_signature(baseline_decisions, timestamp)
        clone_sig = _decision_signature(base_decisions, timestamp)
        parity_rows.append(
            {
                "timestamp": timestamp,
                "decision_rows_baseline": int(len(baseline_sig)),
                "decision_rows_clone": int(len(clone_sig)),
                "noop_decision_signature_equal": bool(baseline_sig.equals(clone_sig)),
                "noop_open_positions": int(len(state.open_positions)),
                "noop_wallet": float(state.wallet),
            }
        )
        strategy_ids_for_ts = _target_strategy_ids_for_timestamp(target_actions, timestamp, strategy_ids)
        for strategy_id in strategy_ids_for_ts:
            local_candidates = suffix.loc[
                suffix["timestamp"].eq(timestamp) & suffix["strategy_id"].astype(str).eq(str(strategy_id))
            ]
            if local_candidates.empty:
                continue
            for action_family, default_action_values, baseline_value in (
                ("size", SIZE_ACTIONS, 1.0),
                ("threshold", THRESHOLD_UPLIFTS, 0.0),
            ):
                action_values = _target_action_values(
                    target_actions,
                    timestamp=timestamp,
                    strategy_id=strategy_id,
                    action_family=action_family,
                    default=default_action_values,
                )
                for action_value in action_values:
                    action_suffix = _apply_local_action(
                        suffix,
                        timestamp=timestamp,
                        strategy_id=strategy_id,
                        action_family=action_family,
                        action_value=float(action_value),
                    )
                    action_decisions, _action_eq, _action_metrics = replay_candidates(
                        action_suffix,
                        params,
                        mode="global_auction",
                        ev_curve=ev_curve,
                        initial_state=state,
                        market_mode=args.market_mode,
                    )
                    action_accepted = _accepted_trades(action_suffix, action_decisions)
                    action_full = _utility(
                        action_accepted,
                        start=timestamp,
                        horizon_hours=int(args.horizon_hours),
                        lambda_dd=float(args.lambda_dd),
                        lambda_turnover=float(args.lambda_turnover),
                    )
                    action_immediate = _utility(
                        _accepted_at_timestamp(action_accepted, timestamp),
                        start=timestamp,
                        horizon_hours=int(args.horizon_hours),
                        lambda_dd=float(args.lambda_dd),
                        lambda_turnover=float(args.lambda_turnover),
                    )
                    action_sig = _decision_signature(action_decisions, timestamp)
                    binds = not bool(action_sig.equals(clone_sig))
                    rows.append(
                        {
                            "timestamp": timestamp,
                            "strategy_id": strategy_id,
                            "action_family": action_family,
                            "action_value": float(action_value),
                            "is_baseline_action": bool(abs(float(action_value) - float(baseline_value)) < 1e-12),
                            "action_binds": bool(binds),
                            "base_immediate_J": base_immediate["J"],
                            "action_immediate_J": action_immediate["J"],
                            "delta_immediate_J": action_immediate["J"] - base_immediate["J"],
                            "base_full_J": base_full["J"],
                            "action_full_J": action_full["J"],
                            "delta_full_J": action_full["J"] - base_full["J"],
                            "base_full_net_pnl": base_full["net_pnl"],
                            "action_full_net_pnl": action_full["net_pnl"],
                            "delta_full_net_pnl": action_full["net_pnl"] - base_full["net_pnl"],
                            "base_full_cost_pnl": base_full["cost_pnl"],
                            "action_full_cost_pnl": action_full["cost_pnl"],
                            "delta_full_cost_pnl": action_full["cost_pnl"] - base_full["cost_pnl"],
                            "base_full_turnover": base_full["turnover"],
                            "action_full_turnover": action_full["turnover"],
                            "delta_full_turnover": action_full["turnover"] - base_full["turnover"],
                            "base_immediate_trades": int(len(_accepted_at_timestamp(base_accepted, timestamp))),
                            "action_immediate_trades": int(len(_accepted_at_timestamp(action_accepted, timestamp))),
                        }
                    )

    labels = pd.DataFrame(rows)
    parity = pd.DataFrame(parity_rows)
    if labels.empty:
        raise RuntimeError("No counterfactual labels generated")
    labels.to_csv(args.output_dir / "exact_state_counterfactual_labels.csv", index=False)
    parity.to_csv(args.output_dir / "exact_state_noop_parity.csv", index=False)

    oracle_rows: list[dict[str, Any]] = []
    for family, group in labels.groupby("action_family", sort=True):
        best = group.sort_values(["timestamp", "strategy_id", "delta_full_J", "delta_immediate_J"], ascending=[True, True, False, False]).drop_duplicates(["timestamp", "strategy_id"])
        oracle_rows.append(
            {
                "arm": f"C_oracle_{family}",
                "evaluated_pairs": int(len(best)),
                "positive_delta_share": float((best["delta_full_J"] > 0.0).mean()),
                "bind_share": float(best["action_binds"].mean()),
                "mean_delta_full_J": float(best["delta_full_J"].mean()),
                "median_delta_full_J": float(best["delta_full_J"].median()),
                "sum_delta_full_J": float(best["delta_full_J"].sum()),
                "mean_delta_immediate_J": float(best["delta_immediate_J"].mean()),
                "sum_delta_immediate_J": float(best["delta_immediate_J"].sum()),
                "mean_delta_turnover": float(best["delta_full_turnover"].mean()),
                "sum_delta_turnover": float(best["delta_full_turnover"].sum()),
            }
        )
    summary = pd.DataFrame(oracle_rows)
    summary.to_csv(args.output_dir / "exact_state_oracle_summary.csv", index=False)

    manifest = {
        "generated_by": "run_exact_state_counterfactual_oracle",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "broad_candidates": str(args.broad_candidates),
        "deployable_candidates": str(args.deployable_candidates),
        "target_actions": str(args.target_actions) if args.target_actions else None,
        "target_action_rows": int(len(target_actions)),
        "policy_manifest": str(args.policy_manifest),
        "policy_variant": str(args.policy_variant),
        "policy_manifest_run_id": policy_payload.get("run_id"),
        "policy_params": asdict(params),
        "start": str(start),
        "end": str(end),
        "horizon_hours": int(args.horizon_hours),
        "max_timestamps": int(args.max_timestamps),
        "sampled_timestamps": int(len(sampled_ts)),
        "lambda_dd": float(args.lambda_dd),
        "lambda_turnover": float(args.lambda_turnover),
        "size_actions": list(SIZE_ACTIONS),
        "threshold_uplifts": list(THRESHOLD_UPLIFTS),
        "noop_parity_pass": bool(parity["noop_decision_signature_equal"].all()) if not parity.empty else False,
        "outputs": {
            "labels": str(args.output_dir / "exact_state_counterfactual_labels.csv"),
            "parity": str(args.output_dir / "exact_state_noop_parity.csv"),
            "summary": str(args.output_dir / "exact_state_oracle_summary.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2) + "\n")
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
