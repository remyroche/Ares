#!/usr/bin/env python3
"""Frozen Jul11-17 replay and supplementary causal L2 confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.l2_execution_confirmation import (
    L2ConfirmationConfig,
    apply_confirmed_l2_cost,
    confirm_l2_execution,
    summarise_l2_confirmation,
)
from extreme_price_movements.simple_policy_1m_constrained import (
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from extreme_price_movements.simple_policy_1m_joint_objective import (
    evaluate_joint_wallet_objective,
)
from extreme_price_movements.simple_policy_candidate_context import (
    RAW_BAYESIAN_CONTEXT_COLUMNS,
    join_candidate_execution_context,
)
from scripts.reoptimise_simple_policy_1m_spread_only_geometry import (
    _cost_rows,
    _fill_context,
    _monitor_asset_spread_fill,
)
from scripts.report_simple_policy_1m_winner_daily_july import (
    OLD_ATR,
    OLD_CACHE,
    OLD_CANDIDATES,
    PARAMS,
    PARENT,
    STORE,
    _prediction_candidates,
)
from scripts.report_simple_policy_1m_winner_daily_nonzero_volume import (
    DAILY_DIR,
    FORWARD_DIR,
    VOLUME_CACHE,
    _entry_minute_volume,
)
from scripts.report_simple_policy_1m_winner_forward_july import _forward_context
from scripts.run_simple_policy_1m_capital_ablation import (
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (
    ExperimentData,
    _indices_between,
)
from scripts.run_simple_policy_1m_contextual_ablation import _bayesian_sizes, _load_atr

RICH_LEDGER = Path("data_perp/reports/meta_v9_recovery_20260717/residual_state_mda95_hier_newaegmm_downstream_retrain_v1/admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet")
WINNER_DIR = Path("data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1/joint_layered_wallet80_holdeff_l2_20260718_v2")


def _data(rows, arrays, atr, spec, deployed):
    open0, high, low, close, valid = arrays
    return ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)


def _load_books(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("kraken_futures_perp_orderbooks_*.parquet"))
    if not files:
        raise RuntimeError(f"No L2 snapshots under {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-dir", type=Path, default=WINNER_DIR)
    parser.add_argument("--wallet-usdt", type=float, default=32.594905632133376)
    parser.add_argument("--max-snapshot-age-minutes", type=float, default=75.0)
    parser.add_argument("--max-walk-slippage-bps", type=float, default=50.0)
    parser.add_argument("--l2-root", type=Path, default=Path("data_perp/exchanges/krakenfutures/spread_snapshots"))
    args = parser.parse_args()
    deployed, _ = _load_deployed_side_params(PARENT)
    base_spec = ConstrainedReplaySpec()
    spec = replace(base_spec, fee_per_side=0.0)
    winner = json.loads((args.winner_dir / "winner_params.json").read_text())
    saved = json.loads(PARAMS.read_text())
    baseline_params = saved["fold_3"]["full_train_parent"]

    old = pd.read_parquet(OLD_CANDIDATES)
    old["timestamp"] = pd.to_datetime(old["timestamp"], utc=True)
    old = old.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="stable").reset_index(drop=True)
    old, _ = join_candidate_execution_context(old, pd.read_parquet(RICH_LEDGER))
    old_context = old.loc[:, list(RAW_BAYESIAN_CONTEXT_COLUMNS)].copy()
    old_atr = _load_atr(old, OLD_ATR)
    old_arrays = _load_or_build_path_cache(old, store_root=STORE, cache_dir=OLD_CACHE, spec=base_spec, rebuild=False)[:5]
    old_volume = np.memmap(VOLUME_CACHE, mode="r", dtype="float32", shape=(len(old), base_spec.path_len))
    old_liquid = np.isfinite(old_volume[:, 0]) & (old_volume[:, 0] > 0.0)
    old = old.loc[old_liquid].reset_index(drop=True)
    old_context = old_context.loc[old_liquid].reset_index(drop=True)
    old_atr = np.asarray(old_atr)[old_liquid]
    old_arrays = tuple(np.asarray(value)[old_liquid] for value in old_arrays)
    old_data = _data(old, old_arrays, old_atr, base_spec, deployed)
    old_data.rank = old["ev_rank_pct"].to_numpy(float)
    fit_idx = _indices_between(old_data, "2026-05-01", "2026-07-01")
    baseline_fit_outputs = old_data.simulate(fit_idx, baseline_params, FAMILY_TRAILING_ONLY)

    forward = pd.read_parquet(FORWARD_DIR / "forward_candidates_jul11_16.parquet")
    forward["timestamp"] = pd.to_datetime(forward["timestamp"], utc=True)
    forward = forward.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="stable").reset_index(drop=True)
    forward_context, _ = _forward_context(forward)
    forward_atr = _load_atr(forward, FORWARD_DIR / "causal_entry_atr_audit.parquet")
    forward_arrays = _load_or_build_path_cache(forward, store_root=STORE, cache_dir=FORWARD_DIR / "path_cache", spec=base_spec, rebuild=False)[:5]
    forward_admitted = (
        np.isfinite(pd.to_numeric(forward["threshold_basis_corrected_expected_ev"], errors="coerce"))
        & np.isfinite(pd.to_numeric(forward["threshold_basis_corrected_expected_ev_rank"], errors="coerce"))
    )
    forward_liquid = (_entry_minute_volume(forward, STORE) > 0.0) & forward_admitted.to_numpy()

    july17 = pd.read_parquet(DAILY_DIR / "july17_partial_candidates.parquet")
    july17["timestamp"] = pd.to_datetime(july17["timestamp"], utc=True)
    july17 = july17.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="stable").reset_index(drop=True)
    spread_reference = pd.read_parquet("data_perp/reports/july_01_16_current_policy_metrics_20260717/current_policy_candidates_through_july16.parquet")
    rebuilt17, july17_context = _prediction_candidates(DAILY_DIR / "jul17_prediction_ledger.parquet", pd.Timestamp("2026-07-17 08:00", tz="UTC"), spread_reference)
    if not july17[["timestamp", "symbol", "side", "rank_pct"]].equals(rebuilt17[["timestamp", "symbol", "side", "rank_pct"]]):
        raise RuntimeError("July17 reconstructed context mismatch")
    july17_atr = _load_atr(july17, DAILY_DIR / "july17_causal_entry_atr_audit.parquet")
    july17_arrays = _load_or_build_path_cache(july17, store_root=STORE, cache_dir=DAILY_DIR / "july17_path_cache", spec=base_spec, rebuild=False)[:5]
    july17_liquid = _entry_minute_volume(july17, STORE) > 0.0

    rows = pd.concat([forward.loc[forward_liquid], july17.loc[july17_liquid]], ignore_index=True)
    rows, spread_audit = _monitor_asset_spread_fill(rows, monitor_root=args.l2_root, quantile=0.75)
    rows = _cost_rows(rows, 1.5)
    context = pd.concat([
        forward_context.loc[forward_liquid, list(RAW_BAYESIAN_CONTEXT_COLUMNS)],
        july17_context.loc[july17_liquid, list(RAW_BAYESIAN_CONTEXT_COLUMNS)],
    ], ignore_index=True)
    arrays = tuple(np.concatenate([np.asarray(a)[forward_liquid], np.asarray(b)[july17_liquid]]) for a, b in zip(forward_arrays, july17_arrays))
    atr = np.concatenate([np.asarray(forward_atr)[forward_liquid], np.asarray(july17_atr)[july17_liquid]])
    data = _data(rows, arrays, atr, spec, deployed)
    corrected_ev = rows["threshold_basis_corrected_expected_ev"].to_numpy(float)
    corrected_rank = rows["threshold_basis_corrected_expected_ev_rank"].to_numpy(float)
    data.rank = corrected_rank.copy()

    all_rows = pd.concat([old, rows], ignore_index=True)
    all_context = pd.concat([old_context, context], ignore_index=True)
    all_context, fallback = _fill_context(all_context, fit_idx)
    sizing_data = SimpleNamespace(
        rows=all_rows,
        side=pd.to_numeric(all_rows["side"], errors="raise").to_numpy(float),
        rank=np.concatenate([old_data.rank, corrected_rank]),
    )
    apply_idx = np.arange(len(all_rows), dtype=np.int64)
    arms = {
        "joint_layered_wallet80": (winner["params_by_side"], winner["sizing"]),
        "joint_trailing_raw_bayesian_wallet80_baseline": (
            baseline_params,
            {
                "strength": float(saved["fold_3"]["sizing"]["strength"]),
                "ood_weight": float(saved["fold_3"]["sizing"]["ood_weight"]),
            },
        ),
    }
    books = _load_books(args.l2_root)
    metrics, ledgers = [], []
    for arm, (params, sizing) in arms.items():
        full_sizes, _ = _bayesian_sizes(
            sizing_data, fit_idx, apply_idx, baseline_fit_outputs, all_context,
            strength=float(sizing["strength"]), ood_weight=float(sizing["ood_weight"]),
        )
        sizes = full_sizes[len(old):]
        if not np.isfinite(sizes).all():
            bad = int((~np.isfinite(sizes)).sum())
            raise RuntimeError(
                f"{arm}: {bad}/{len(sizes)} non-finite frozen Bayesian multipliers; "
                f"context_nonfinite={int((~np.isfinite(all_context.to_numpy(float))).sum())}"
            )
        outputs = data.simulate(np.arange(len(rows)), params, FAMILY_TRAILING_ONLY)
        _, portfolio, detail = evaluate_joint_wallet_objective(
            rows=rows, timestamps_ns=data.timestamps, symbol_codes=data.symbol_codes,
            side=data.side, raw_entry_prices=data.open0, entry_half_spread_bps=data.entry_spread,
            close_paths=data.close, exit_bars=outputs["exit_bars"], net_returns=outputs["net_return"],
            corrected_ev=corrected_ev, corrected_ev_rank=corrected_rank,
            bayesian_multiplier=sizes, holding_power=0.8, holding_efficiency_weight=0.10,
            max_wallet_invested=0.80, max_new_per_bar=2,
        )
        chosen = np.flatnonzero(detail["selected"])
        trades = rows.iloc[chosen][["timestamp", "symbol", "side_name"]].copy()
        trades = trades.rename(columns={"timestamp": "entry_ts", "side_name": "side"})
        trades["exit_ts"] = trades["entry_ts"] + pd.to_timedelta(outputs["exit_bars"][chosen] + 1, unit="m")
        trades["baseline_net_return"] = outputs["net_return"][chosen]
        trades["admitted_quote_notional"] = detail["admitted_notional"][chosen] * float(args.wallet_usdt)
        entry_px = data.open0[chosen] * (1.0 + np.where(data.side[chosen] > 0, 1.0, -1.0) * data.entry_spread[chosen] / 10000.0)
        trades["admitted_quantity"] = trades["admitted_quote_notional"].to_numpy() / entry_px
        diag = confirm_l2_execution(
            trades, books,
            config=L2ConfirmationConfig(
                max_snapshot_age=pd.Timedelta(minutes=args.max_snapshot_age_minutes),
                max_walk_slippage_bps=args.max_walk_slippage_bps,
            ),
        )
        adjusted = apply_confirmed_l2_cost(trades["baseline_net_return"], diag)
        ledger = pd.concat([trades.reset_index(drop=True), diag.reset_index(drop=True), adjusted.reset_index(drop=True)], axis=1)
        ledger["arm"] = arm
        covered = ledger["l2_cost_applied"].to_numpy(bool)
        quote = ledger["admitted_quote_notional"].to_numpy(float)
        base_return = ledger["baseline_net_return"].to_numpy(float)
        adjusted_return = ledger["l2_adjusted_net_return"].to_numpy(float)
        summary = summarise_l2_confirmation(diag)
        summary.update({
            "arm": arm, **portfolio, "wallet_anchor_usdt": float(args.wallet_usdt),
            "l2_matched_baseline_pnl_usdt": float(np.sum(quote[covered] * base_return[covered])),
            "l2_matched_adjusted_pnl_usdt": float(np.sum(quote[covered] * adjusted_return[covered])),
            "l2_matched_delta_pnl_usdt": float(np.sum(quote[covered] * (adjusted_return[covered] - base_return[covered]))),
            "l2_matched_baseline_net_ev": float(np.mean(base_return[covered])) if covered.any() else np.nan,
            "l2_matched_adjusted_net_ev": float(np.mean(adjusted_return[covered])) if covered.any() else np.nan,
        })
        metrics.append(summary)
        ledgers.append(ledger)

    metric_frame = pd.DataFrame(metrics)
    ledger_frame = pd.concat(ledgers, ignore_index=True)
    key = ["entry_ts", "symbol", "side"]
    covered_by_arm = []
    for arm in arms:
        part = ledger_frame.loc[
            ledger_frame["arm"].eq(arm) & ledger_frame["l2_cost_applied"],
            key,
        ].drop_duplicates()
        covered_by_arm.append(pd.MultiIndex.from_frame(part))
    matched_keys = covered_by_arm[0].intersection(covered_by_arm[1])
    matched_records = []
    for arm in arms:
        part = ledger_frame.loc[ledger_frame["arm"].eq(arm)].copy()
        mask = pd.MultiIndex.from_frame(part[key]).isin(matched_keys)
        matched = part.loc[mask]
        quote = matched["admitted_quote_notional"].to_numpy(float)
        before = matched["baseline_net_return"].to_numpy(float)
        after = matched["l2_adjusted_net_return"].to_numpy(float)
        matched_records.append({
            "arm": arm, "matched_trade_count": int(len(matched)),
            "baseline_net_ev": float(np.mean(before)) if len(matched) else np.nan,
            "l2_adjusted_net_ev": float(np.mean(after)) if len(matched) else np.nan,
            "baseline_pnl_usdt": float(np.sum(quote * before)),
            "l2_adjusted_pnl_usdt": float(np.sum(quote * after)),
            "delta_pnl_usdt": float(np.sum(quote * (after - before))),
        })
    matched_frame = pd.DataFrame(matched_records)
    metric_frame.to_csv(args.winner_dir / "forward_jul11_17_l2_confirmation_metrics.csv", index=False)
    matched_frame.to_csv(args.winner_dir / "forward_jul11_17_l2_matched_intersection.csv", index=False)
    ledger_frame.to_parquet(args.winner_dir / "forward_jul11_17_l2_confirmation_ledger.parquet", index=False)
    (args.winner_dir / "l2_confirmation_manifest.json").write_text(json.dumps({
        "role": "supplementary post-freeze confirmation only", "no_extrapolation": True,
        "wallet_anchor_usdt": args.wallet_usdt, "wallet_anchor_source": "extreme_price_movements/logs/daily_report_state.json:last_available_balance_usdt",
        "max_snapshot_age_minutes": args.max_snapshot_age_minutes,
        "max_walk_slippage_bps": args.max_walk_slippage_bps,
        "cost_application": "incremental depth slippage only; 1.5x spread remains in baseline",
        "spread_audit": spread_audit, "context_fallback": fallback,
    }, indent=2, default=str))
    print(metric_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
