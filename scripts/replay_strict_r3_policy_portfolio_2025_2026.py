#!/usr/bin/env python3
"""Portfolio-aware replay of canonical 15m simple-policy outcomes.

The exit path is already materialised by the canonical simple-policy
optimiser.  This utility turns those rows into the portfolio engine's native
candidate contract, selects a score threshold only on 2025, then applies the
frozen threshold and explicit concurrency/exposure/asset limits to both the
2025 development period and the forward 2026 period.
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
    PortfolioPolicyParams, normalise_candidate_table, replay_candidates,
)


OUTCOMES = ROOT / 'data_perp/artifacts/strict_r3_simple_policy_optimised_replay_2025_2026_v1/candidate_policy_outcomes.parquet'
THRESHOLDS = (0.950, 0.970, 0.980, 0.990, 0.995)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outcomes', type=Path, default=OUTCOMES)
    p.add_argument('--out-dir', type=Path, default=ROOT / 'data_perp/artifacts/strict_r3_simple_policy_optimised_portfolio_2025_2026_v1')
    p.add_argument('--initial-wallet', type=float, default=10_000.0)
    p.add_argument('--perp-leverage', type=float, default=10.0)
    p.add_argument(
        '--margin-slot-wallet-fraction', type=float, default=None,
        help='Opt-in wallet fraction used as initial margin per open slot.',
    )
    return p.parse_args()


def _candidates(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[rows.policy_path_valid.astype(bool) & np.isfinite(rows.policy_net_bps)].copy()
    if x.empty:
        raise ValueError('No valid policy rows.')
    entry_ts = pd.to_datetime(x.__decision_ts__, utc=True)
    # An exit in bar zero still occupies the first 15-minute bar.  This is
    # essential for portfolio capacity to be meaningful at the entry instant.
    exit_ts = entry_ts + pd.to_timedelta(x.policy_exit_bar_15m.astype(int).add(1) * 15, unit='min')
    out = pd.DataFrame({
        'timestamp': entry_ts,
        'symbol': x.__symbol__.astype(str), 'side': 'long',
        'strategy_id': 'strict_r3_base_plus_consensus25_long',
        'policy_archetype': 'strict_r3_base_plus_consensus25_long',
        'normalized_rank_score': pd.to_numeric(x.final_score, errors='coerce'),
        'strategy_rank_pct': pd.to_numeric(x.final_score, errors='coerce'),
        'base_strategy_threshold': 0.0,
        'calibrated_score': pd.to_numeric(x.final_score, errors='coerce'),
        'entry_price': pd.to_numeric(x.policy_entry_price, errors='coerce'),
        'exit_timestamp': exit_ts,
        'exit_price': pd.to_numeric(x.policy_exit_price, errors='coerce'),
        'net_return': pd.to_numeric(x.policy_net_bps, errors='coerce') / 10_000.0,
        'gross_return': pd.to_numeric(x.policy_gross_bps, errors='coerce') / 10_000.0,
        'holding_bars': pd.to_numeric(x.policy_exit_bar_15m, errors='coerce').add(1),
        'simple_policy_exit_reason': x.policy_exit_reason.astype(str),
        'fees_bps': 100.0, 'slippage_bps': 0.0, 'expected_friction_bps': 100.0,
        'price_gap_bps': 0.0, 'liquidity_capacity_weight': 1.0,
        'source_month': x.month.astype(str), 'candidate_id': x.candidate_id.astype(str),
    })
    if out[['entry_price', 'exit_price', 'net_return', 'gross_return']].isna().any().any():
        raise ValueError('Policy handoff has non-finite required portfolio fields.')
    return normalise_candidate_table(out)


def _params(
    threshold: float,
    *,
    perp_leverage: float,
    margin_slot_wallet_fraction: float | None,
    strategy_ids: tuple[str, ...] = ('strict_r3_base_plus_consensus25_long',),
) -> PortfolioPolicyParams:
    """Frozen single-strategy portfolio contract, independent of outcomes."""
    return PortfolioPolicyParams(
        capacity_mode='pre_leverage_wallet', enforce_position_count_cap=True,
        max_concurrent_positions=8, max_concurrent_per_side=8,
        max_concurrent_per_strategy=None, max_concurrent_per_symbol=1,
        max_new_entries_per_bar=2, max_new_entries_per_strategy_per_bar=2,
        max_total_wallet_allocation_pct=0.80,
        perp_default_leverage=float(perp_leverage),
        # The legacy $5k ceiling is a live-test safeguard.  Capacity studies
        # using explicit margin slots must not silently inherit it.
        max_position_quote_notional=1_000_000_000.0,
        margin_slot_wallet_fraction=margin_slot_wallet_fraction,
        global_threshold_floor=float(threshold),
        threshold_viability_margin=0.0, occupancy_threshold_alpha=0.0,
        allocation_threshold_alpha=0.0,
        rank_size_power=1.0, rank_multiplier_min=1.0, rank_multiplier_max=1.0,
        max_signal_gap_bps=None, min_liquidity_capacity_weight=None,
        cooldown_hours_after_loss=0.0, max_consecutive_losing_trades=0,
        global_loss_cooldown_hours=0.0, max_consecutive_losing_trades_per_archetype=0,
        archetype_loss_cooldown_hours=0.0,
        portfolio_policy_version='strict_r3_frozen_15m_global_auction_v1',
        strategy_ids=tuple(strategy_ids),
    )


def _summary(decisions: pd.DataFrame, equity: pd.DataFrame, metrics: dict, period: str, threshold: float) -> dict:
    accepted = decisions[decisions.accepted.fillna(False)].copy()
    returns = pd.to_numeric(accepted.get('position_net_return'), errors='coerce')
    gross = pd.to_numeric(accepted.get('position_gross_return'), errors='coerce')
    start = pd.to_datetime(accepted.timestamp, utc=True).min() if len(accepted) else pd.NaT
    end = pd.to_datetime(accepted.timestamp, utc=True).max() if len(accepted) else pd.NaT
    days = max((end - start).total_seconds() / 86_400.0, 1.0) if pd.notna(start) and pd.notna(end) else 1.0
    return {
        'period': period, 'threshold': threshold, 'input_candidates': int(len(decisions)),
        'accepted_trades': int(len(accepted)), 'trades_per_day': float(len(accepted) / days),
        'gross_bps_per_trade': float(gross.mean() * 10_000.0) if len(accepted) else np.nan,
        'net_bps_per_trade': float(returns.mean() * 10_000.0) if len(accepted) else np.nan,
        'net_sum_bps': float(returns.sum() * 10_000.0) if len(accepted) else 0.0,
        'positive_rate': float(returns.gt(0).mean()) if len(accepted) else np.nan,
        'final_wallet': float(equity.wallet.iloc[-1]) if len(equity) and 'wallet' in equity else np.nan,
        'portfolio_net_pnl': float(equity.wallet.iloc[-1] - equity.wallet.iloc[0]) if len(equity) and 'wallet' in equity else np.nan,
        'replay_metric_summary': json.dumps(metrics, default=str),
    }


def _monthly(decisions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    accepted = decisions[decisions.accepted.fillna(False)].copy()
    accepted['month'] = pd.to_datetime(accepted.timestamp, utc=True).dt.to_period('M').astype(str)
    accepted['net_bps'] = pd.to_numeric(accepted.position_net_return, errors='coerce') * 10_000.0
    accepted['gross_bps'] = pd.to_numeric(accepted.position_gross_return, errors='coerce') * 10_000.0
    return accepted.groupby('month', as_index=False).agg(
        trades=('accepted', 'size'), net_bps_per_trade=('net_bps', 'mean'), gross_bps_per_trade=('gross_bps', 'mean'),
        net_sum_bps=('net_bps', 'sum'), positive_rate=('net_bps', lambda v: float((v > 0).mean()))
    ).assign(threshold=threshold)


def _run(
    candidates: pd.DataFrame,
    threshold: float,
    period: str,
    *,
    initial_wallet: float,
    perp_leverage: float,
    margin_slot_wallet_fraction: float | None,
    ev_curve: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # Pre-filtering makes a fixed global score gate explicit.  The portfolio
    # engine still applies its identical threshold and all capacity checks.
    eligible = candidates[candidates.calibrated_score.ge(threshold)].copy()
    strategy_ids = tuple(sorted(eligible['strategy_id'].astype(str).unique()))
    if not strategy_ids:
        strategy_ids = ('strict_r3_base_plus_consensus25_long',)
    decisions, equity, metrics = replay_candidates(
        eligible,
        _params(
            threshold,
            perp_leverage=perp_leverage,
            margin_slot_wallet_fraction=margin_slot_wallet_fraction,
            strategy_ids=strategy_ids,
        ),
        mode='global_auction', ev_curve=ev_curve, market_mode='perps',
        initial_wallet=float(initial_wallet),
    )
    return decisions, equity, _monthly(decisions, threshold), _summary(decisions, equity, metrics, period, threshold)


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f'{args.out_dir} exists; use a new immutable output directory.')
    args.out_dir.mkdir(parents=True)
    rows = pd.read_parquet(args.outcomes)
    candidates = _candidates(rows)
    c2025 = candidates[pd.to_datetime(candidates.timestamp, utc=True).dt.year.eq(2025)].copy()
    c2026 = candidates[pd.to_datetime(candidates.timestamp, utc=True).dt.year.eq(2026)].copy()
    sweep_rows = []
    for threshold in THRESHOLDS:
        decisions, equity, _, summary = _run(
            c2025, threshold, '2025_policy_selection',
            initial_wallet=args.initial_wallet, perp_leverage=args.perp_leverage,
            margin_slot_wallet_fraction=args.margin_slot_wallet_fraction,
        )
        sweep_rows.append(summary)
    sweep = pd.DataFrame(sweep_rows)
    # Objective is net PnL, subject to minimum evidence.  This matches the
    # simple-policy optimisation convention while avoiding a one-trade tail.
    viable = sweep[sweep.accepted_trades.ge(250)].copy()
    if viable.empty:
        viable = sweep.copy()
    winner = viable.sort_values(['portfolio_net_pnl', 'net_bps_per_trade', 'threshold'], ascending=[False, False, False]).iloc[0]
    threshold = float(winner.threshold)
    all_summaries = []
    for period, frame in [('2025_development', c2025), ('2026_forward_oos', c2026), ('2025_2026_combined', candidates)]:
        decisions, equity, monthly, summary = _run(
            frame, threshold, period,
            initial_wallet=args.initial_wallet, perp_leverage=args.perp_leverage,
            margin_slot_wallet_fraction=args.margin_slot_wallet_fraction,
        )
        decisions.to_parquet(args.out_dir / f'{period}_portfolio_decisions.parquet', index=False, compression='zstd')
        equity.to_parquet(args.out_dir / f'{period}_portfolio_equity.parquet', index=False, compression='zstd')
        monthly.to_parquet(args.out_dir / f'{period}_monthly_metrics.parquet', index=False)
        all_summaries.append(summary)
    sweep.to_parquet(args.out_dir / 'threshold_selection_2025.parquet', index=False)
    pd.DataFrame(all_summaries).to_parquet(args.out_dir / 'portfolio_summary.parquet', index=False)
    (args.out_dir / 'selected_portfolio_policy.json').write_text(json.dumps({
        'selection_period': '2025 development only', 'threshold_grid': list(THRESHOLDS), 'winner_threshold': threshold,
        'winner_objective': 'max 2025 portfolio net PnL subject to >=250 accepted trades',
        'initial_wallet': args.initial_wallet,
        'params': _params(
            threshold, perp_leverage=args.perp_leverage,
            margin_slot_wallet_fraction=args.margin_slot_wallet_fraction,
        ).to_live_config(),
    }, indent=2))
    (args.out_dir / 'run_manifest.json').write_text(json.dumps({
        'policy_outcomes': str(args.outcomes), 'policy_exit_contract': 'SimplePolicyOptimiser selected 15m trailing policy; H12; 100 bps cost exactly once',
        'portfolio_protocol': 'threshold selected only on 2025; same frozen policy evaluated forward on 2026',
        'scope': 'long only; strict-R3 base-plus-consensus25 score; portfolio global auction',
    }, indent=2))
    print(json.dumps({'winner_threshold': threshold, 'output': str(args.out_dir)}))


if __name__ == '__main__':
    main()
