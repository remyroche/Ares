#!/usr/bin/env python3
"""Replay immutable MC1_d2 ablation predictions through the canonical engine.

This is a reporting-only companion to ``run_strict_r3_mc1_d2_controlled_ablation``.
It adds no model, target, calibration, or selection decision.  Its purpose is
to prevent a simplified occupancy proxy from being mistaken for the full
wallet/exposure/concurrency portfolio replay used by MC1 research.
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
    PortfolioPolicyParams,
    normalise_candidate_table,
    replay_candidates,
)


CAUSAL_AUCTION_CURVE = {
    "schema": "monotone_ev_curve_v1", "x": [0.0, 1.0], "y": [0.0, 1.0],
    "ev_span": 1.0, "n_rows": 0,
    "source": "fixed MC1 controlled-ablation rank-only auction",
}


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _params() -> PortfolioPolicyParams:
    """The 7x / 10%-margin-slot MC1 research portfolio contract."""
    return PortfolioPolicyParams(
        capacity_mode="pre_leverage_wallet", enforce_position_count_cap=True,
        max_concurrent_positions=8, max_concurrent_per_side=8,
        max_concurrent_per_strategy=None, max_concurrent_per_symbol=1,
        max_new_entries_per_bar=2, max_new_entries_per_strategy_per_bar=2,
        max_total_wallet_allocation_pct=.80, perp_default_leverage=7.0,
        max_position_quote_notional=1_000_000_000.0,
        margin_slot_wallet_fraction=.10, global_threshold_floor=0.0,
        threshold_viability_margin=0.0, occupancy_threshold_alpha=0.0,
        allocation_threshold_alpha=0.0, rank_size_power=1.0,
        rank_multiplier_min=1.0, rank_multiplier_max=1.0,
        max_signal_gap_bps=None, min_liquidity_capacity_weight=None,
        cooldown_hours_after_loss=0.0, max_consecutive_losing_trades=0,
        global_loss_cooldown_hours=0.0,
        max_consecutive_losing_trades_per_archetype=0,
        archetype_loss_cooldown_hours=0.0,
        portfolio_policy_version="strict_r3_mc1_controlled_ablation_v1",
        strategy_ids=("strict_r3_mc1_controlled_long",),
    )


def _candidate_table(
    prediction: pd.DataFrame,
    policy: pd.DataFrame,
    threshold_bps: float,
    *,
    invalid_outcome_mode: str,
) -> pd.DataFrame:
    data = prediction.merge(policy, on="candidate_id", how="left", validate="one_to_one", suffixes=("", "__policy"))
    data["__decision_ts__"] = _utc(data["__decision_ts__"])
    if data.policy_path_valid.isna().any():
        raise ValueError("policy ledger does not cover every prediction identity")
    valid_all = (
        data.policy_path_valid.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(data.policy_net_bps, errors="coerce"))
        & np.isfinite(pd.to_numeric(data.policy_gross_bps, errors="coerce"))
        & np.isfinite(pd.to_numeric(data.policy_exit_bar_15m, errors="coerce"))
    )
    if invalid_outcome_mode == "exclude":
        # This is a label-complete research evaluation: unresolved/invalid
        # future paths never become zero-return pseudo-trades or reserve
        # portfolio capacity.  It is explicitly not a live eligibility gate.
        data = data.loc[valid_all].copy()
    elif invalid_outcome_mode != "reserve":
        raise ValueError(f"unknown invalid outcome mode: {invalid_outcome_mode}")
    admitted = data.loc[pd.to_numeric(data.mc1_expected_bps, errors="coerce").ge(threshold_bps)].copy()
    # This deliberately mirrors the historical MC1 replay adapter: expected
    # EV decides admission, but the rank coordinate used for sizing is the
    # frozen final-score percentile *within the admitted timestamp cohort*.
    # It is target-free and does not give mapper values auction authority.
    admitted["auction_rank"] = admitted.groupby("__decision_ts__", sort=False)["final_score"].rank(
        pct=True, method="average",
    )
    valid = (
        admitted.policy_path_valid.fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(admitted.policy_net_bps, errors="coerce"))
        & np.isfinite(pd.to_numeric(admitted.policy_gross_bps, errors="coerce"))
        & np.isfinite(pd.to_numeric(admitted.policy_exit_bar_15m, errors="coerce"))
    )
    exit_bar = pd.to_numeric(admitted.policy_exit_bar_15m, errors="coerce").where(valid, 47).astype(int)
    decision = _utc(admitted["__decision_ts__"])
    candidate = pd.DataFrame({
        "timestamp": decision,
        "symbol": admitted["__symbol__"].astype(str), "side": "long",
        "strategy_id": "strict_r3_mc1_controlled_long",
        "policy_archetype": "strict_r3_mc1_controlled_long",
        "normalized_rank_score": admitted.auction_rank.to_numpy(float),
        "strategy_rank_pct": admitted.auction_rank.to_numpy(float),
        "base_strategy_threshold": 0.0,
        "calibrated_score": admitted.auction_rank.to_numpy(float),
        "entry_price": pd.to_numeric(admitted.policy_entry_price, errors="coerce").where(valid, 1.0),
        "exit_timestamp": decision + pd.to_timedelta((exit_bar + 1) * 15, unit="min"),
        "exit_price": pd.to_numeric(admitted.policy_exit_price, errors="coerce").where(valid, 1.0),
        "net_return": pd.to_numeric(admitted.policy_net_bps, errors="coerce").where(valid, 0.0) / 10_000.0,
        "gross_return": pd.to_numeric(admitted.policy_gross_bps, errors="coerce").where(valid, 0.0) / 10_000.0,
        "holding_bars": exit_bar + 1,
        "simple_policy_exit_reason": admitted.policy_exit_reason.where(valid, "OUTCOME_UNAVAILABLE_RESERVED_H12").astype(str),
        "fees_bps": 100.0, "slippage_bps": 0.0, "expected_friction_bps": 100.0,
        "price_gap_bps": 0.0, "liquidity_capacity_weight": 1.0,
        "source_month": decision.dt.strftime("%Y-%m"), "candidate_id": admitted.candidate_id.astype(str),
        "mapped_expected_net_bps": pd.to_numeric(admitted.mc1_expected_bps, errors="coerce"),
        "policy_outcome_available": valid.to_numpy(bool),
    })
    return normalise_candidate_table(candidate)


def _metrics(decisions: pd.DataFrame, equity: pd.DataFrame, arm: str, period: str) -> dict[str, object]:
    wallet_series = (
        pd.to_numeric(equity["wallet"], errors="coerce").dropna()
        if "wallet" in equity.columns else pd.Series(dtype=float)
    )
    if decisions.empty:
        return {
            "arm": arm, "period": period, "accepted_rows": 0, "realised_rows": 0,
            "outcome_coverage": float("nan"), "net_ev_bps_per_realised_trade": float("nan"),
            "net_sum_bps_realised": 0.0, "net_ev_bps_per_selected_trade": float("nan"),
            "net_sum_bps_selected": 0.0, "worst_month_bps": float("nan"),
            "worst_week_bps": float("nan"), "positive_month_fraction": float("nan"),
            "max_drawdown": 0.0, "final_wallet": float(wallet_series.iloc[-1]) if len(wallet_series) else 1000.0,
        }
    accepted = decisions.loc[decisions.accepted.fillna(False).astype(bool)].copy()
    outcome = accepted.policy_outcome_available.fillna(False).astype(bool)
    realised = accepted.loc[outcome].copy()
    net = pd.to_numeric(realised.position_net_return, errors="coerce") * 10_000.0
    all_net = pd.to_numeric(accepted.position_net_return, errors="coerce") * 10_000.0
    monthly = realised.groupby(_utc(realised.timestamp).dt.strftime("%Y-%m"), sort=True).apply(
        lambda group: pd.to_numeric(group.position_net_return, errors="coerce").mean() * 10_000.0,
        include_groups=False,
    )
    weekly = realised.groupby(_utc(realised.timestamp).dt.strftime("%G-W%V"), sort=True).apply(
        lambda group: pd.to_numeric(group.position_net_return, errors="coerce").mean() * 10_000.0,
        include_groups=False,
    )
    wallet = wallet_series
    drawdown = float((wallet / wallet.cummax() - 1.0).min()) if len(wallet) else float("nan")
    return {
        "arm": arm, "period": period, "accepted_rows": int(len(accepted)),
        "realised_rows": int(len(realised)),
        "outcome_coverage": float(len(realised) / max(1, len(accepted))),
        "net_ev_bps_per_realised_trade": float(net.mean()) if len(net) else float("nan"),
        "net_sum_bps_realised": float(net.sum()),
        "net_ev_bps_per_selected_trade": float(all_net.mean()) if len(all_net) else float("nan"),
        "net_sum_bps_selected": float(all_net.sum()),
        "worst_month_bps": float(monthly.min()) if len(monthly) else float("nan"),
        "worst_week_bps": float(weekly.min()) if len(weekly) else float("nan"),
        "positive_month_fraction": float(monthly.gt(0).mean()) if len(monthly) else float("nan"),
        "max_drawdown": drawdown,
        "final_wallet": float(wallet.iloc[-1]) if len(wallet) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold-bps", type=float, default=50.0,
                        help="offline MC1 admission threshold; default preserves the frozen +50-bps control")
    parser.add_argument(
        "--invalid-outcome-mode", choices=("reserve", "exclude"), default="reserve",
        help=(
            "reserve preserves legacy capacity-reserving pseudo-trade reporting; "
            "exclude is the canonical label-complete research evaluation"
        ),
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    policy_cols = [
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_gross_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_exit_reason",
    ]
    ledger = pd.read_parquet(
        args.ledger,
        columns=[*policy_cols, "__decision_ts__", "final_score"],
    )
    policy = ledger.loc[:, policy_cols].copy()
    if policy.candidate_id.duplicated().any():
        raise ValueError("policy ledger candidate identity is not unique")
    rows: list[dict[str, object]] = []
    for prediction_path in sorted(args.prediction_dir.glob("predictions_*.parquet")):
        arm = prediction_path.stem.removeprefix("predictions_")
        prediction = pd.read_parquet(prediction_path)
        for year in (2025, 2026):
            piece = prediction.loc[_utc(prediction["__decision_ts__"]).dt.year.eq(year)].copy()
            if piece.empty:
                continue
            candidates = _candidate_table(
                piece, policy, args.threshold_bps,
                invalid_outcome_mode=args.invalid_outcome_mode,
            )
            decisions, equity, _ = replay_candidates(
                candidates, _params(), mode="global_auction", ev_curve=CAUSAL_AUCTION_CURVE,
                market_mode="perps", initial_wallet=1000.0,
            )
            if decisions.empty:
                decisions = decisions.copy()
                decisions["policy_outcome_available"] = pd.Series(dtype=bool)
            else:
                if "candidate_index" not in decisions:
                    raise ValueError("non-empty portfolio decision is missing candidate provenance")
                outcome_lookup = candidates.loc[:, ["candidate_id", "policy_outcome_available"]].reset_index(drop=True)
                outcome_lookup.index.name = "candidate_index"
                decisions = decisions.merge(
                    outcome_lookup, on="candidate_index", how="left", validate="many_to_one",
                )
                if decisions["policy_outcome_available"].isna().any():
                    raise ValueError("portfolio decision is missing candidate outcome provenance")
            decisions.to_parquet(args.out_dir / f"{arm}_{year}_decisions.parquet", index=False, compression="zstd")
            equity.to_parquet(args.out_dir / f"{arm}_{year}_equity.parquet", index=False, compression="zstd")
            metric = _metrics(decisions, equity, arm, str(year))
            metric["admission_threshold_bps"] = float(args.threshold_bps)
            rows.append(metric)
            accepted_count = int(decisions["accepted"].fillna(False).sum()) if "accepted" in decisions.columns else 0
            print(json.dumps({"event": "portfolio_complete", "arm": arm, "year": year, "accepted": accepted_count}), flush=True)
    pd.DataFrame(rows).to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False)
    pd.DataFrame(rows).to_csv(args.out_dir / "portfolio_metrics.csv", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_mc1_d2_controlled_portfolio_v1", "status": "complete",
        "purpose": "canonical portfolio replay of immutable MC1_d2 controlled-ablation predictions",
        "source_predictions": str(args.prediction_dir), "source_ledger": str(args.ledger),
        "admission": f"MC1 expected policy net >= {args.threshold_bps:g} bps", "auction": "final-score-only percentile within admitted timestamp cohort",
        "portfolio": "long-only, 7x, 10%-margin slots, 2 new entries, 8 concurrent, 80% wallet cap",
        "exclusions": ["R5", "live state", "exchange I/O"],
        "invalid_outcome_mode": args.invalid_outcome_mode,
        "outcome_rule": (
            "invalid accepted candidates reserve H12 capacity; realised metrics report coverage separately"
            if args.invalid_outcome_mode == "reserve" else
            "invalid/unresolved policy paths are excluded before the research portfolio replay; "
            "they neither produce pseudo returns nor reserve capacity"
        ),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
