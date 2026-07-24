#!/usr/bin/env python3
"""Apply the fold-3 frozen joint DCA/exit choice through July 16."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_1m_constrained import FAMILY_TRAILING_ONLY, ConstrainedReplaySpec
from extreme_price_movements.simple_policy_optimiser import _with_policy_spread_cost_columns
from scripts.report_simple_policy_1m_winner_forward_july import BASE, CHAMPION, FORWARD_SOURCE, _forward_context
from scripts.run_simple_policy_1m_capital_ablation import _load_deployed_side_params, _load_or_build_path_cache
from scripts.run_simple_policy_1m_constrained_search import ExperimentData, _indices_between
from scripts.run_simple_policy_1m_contextual_ablation import _bayesian_sizes, _load_atr, _load_context
from scripts.run_simple_policy_1m_dca_ablation import _apply_dca, _combine_outputs, _metric, _weekly_ledger


OUTPUT = CHAMPION / "joint_dca_exit_activation_v1"


def main() -> int:
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    choice = manifest["choices"]["fold_3"]
    nested = json.loads((CHAMPION / "evidence/nested_params.json").read_text())
    fold3_base = nested["fold_3"]["full_train_parent"]
    fold3_sizing = nested["fold_3"]["sizing"]
    joint_params = choice["outer_params_by_side"]
    x, y = int(choice["x"]), float(choice["y_fraction"])

    candidates = BASE / "execution_candidates_may_july_v1/simple_policy_candidates_with_archetypes.parquet"
    rich = BASE / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
    posterior = BASE / "complete_parent_state_july_v1/complete_oos_residual_event_states.parquet"
    parent_summary = BASE / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
    rows = pd.read_parquet(candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, _ = _load_context(rows, rich, posterior)
    atr = _load_atr(rows, CHAMPION / "replay/causal_entry_atr_audit.parquet")
    deployed, _ = _load_deployed_side_params(parent_summary)
    spec = ConstrainedReplaySpec()
    store_root = Path("data_perp/exchanges/krakenfutures/execution_1m")
    open0, high, low, close, valid, _ = _load_or_build_path_cache(
        rows, store_root=store_root, cache_dir=CHAMPION / "replay/path_cache", spec=spec, rebuild=False
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)
    train_idx = _indices_between(data, "2026-05-01", "2026-06-14")
    july_old_idx = _indices_between(data, "2026-07-01", "2026-07-11")
    frozen_train = data.simulate(train_idx, fold3_base, FAMILY_TRAILING_ONLY)
    frozen_old_base = data.simulate(july_old_idx, fold3_base, FAMILY_TRAILING_ONLY)
    old_sizes, _ = _bayesian_sizes(
        data, train_idx, july_old_idx, frozen_train, context,
        strength=float(fold3_sizing["strength"]), ood_weight=float(fold3_sizing["ood_weight"]),
    )

    forward = pd.read_parquet(FORWARD_SOURCE)
    forward["timestamp"] = pd.to_datetime(forward["timestamp"], utc=True)
    forward = forward.loc[
        forward["timestamp"].ge(pd.Timestamp("2026-07-11", tz="UTC"))
        & forward["timestamp"].lt(pd.Timestamp("2026-07-17", tz="UTC"))
    ].copy()
    forward = _with_policy_spread_cost_columns(forward, market_mode="perps")
    forward = forward.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    forward_context, _ = _forward_context(forward)
    for column in forward_context.columns:
        values = pd.to_numeric(forward_context[column], errors="coerce")
        missing = ~np.isfinite(values.to_numpy(dtype=np.float64))
        if missing.any():
            values.loc[missing] = float(np.nanmedian(pd.to_numeric(context.iloc[train_idx][column], errors="coerce")))
        forward_context[column] = values
    forward_atr = _load_atr(forward, CHAMPION / "forward_replay_jul11_17_v1/causal_entry_atr_audit.parquet")
    f_open, f_high, f_low, f_close, f_valid, _ = _load_or_build_path_cache(
        forward, store_root=store_root, cache_dir=CHAMPION / "forward_replay_jul11_17_v1/path_cache",
        spec=spec, rebuild=False,
    )
    forward_data = ExperimentData(
        forward, f_open, f_high, f_low, f_close, f_valid, forward_atr, spec, deployed
    )
    forward_idx = np.arange(len(forward), dtype=np.int64)
    combined_rows = pd.concat([rows, forward], ignore_index=True, copy=False)
    combined_context = pd.concat([context, forward_context], ignore_index=True, copy=False)
    sizing_data = SimpleNamespace(
        rows=combined_rows,
        side=pd.to_numeric(combined_rows["side"], errors="coerce").to_numpy(dtype=np.float64),
        rank=pd.to_numeric(combined_rows["rank_pct"], errors="coerce").to_numpy(dtype=np.float64),
    )
    forward_combined_idx = np.arange(len(rows), len(combined_rows), dtype=np.int64)
    combined_sizes, _ = _bayesian_sizes(
        sizing_data, train_idx, forward_combined_idx, frozen_train, combined_context,
        strength=float(fold3_sizing["strength"]), ood_weight=float(fold3_sizing["ood_weight"]),
    )
    july_sizes = np.concatenate([old_sizes[july_old_idx], combined_sizes[forward_combined_idx]])
    july_rows = pd.concat([rows.iloc[july_old_idx], forward], ignore_index=True, copy=False)
    july_data = SimpleNamespace(rows=july_rows)
    july_idx = np.arange(len(july_rows), dtype=np.int64)

    records, weekly_parts, ledger_parts = [], [], []
    for policy, params, dca_first in (
        ("winner_baseline", fold3_base, False),
        ("joint_dca_exit_activation_exit_first", joint_params, False),
        ("joint_dca_exit_activation_dca_first_bound", joint_params, True),
    ):
        arm_x, arm_y = (1, 0.0) if policy == "winner_baseline" else (x, y)
        old_exits = data.simulate(july_old_idx, params, FAMILY_TRAILING_ONLY)
        f_exits = forward_data.simulate(forward_idx, params, FAMILY_TRAILING_ONLY)
        old_out, old_diag = _apply_dca(
            data, july_old_idx, old_exits, x=arm_x, y=arm_y, literal=False, dca_first=dca_first
        )
        f_out, f_diag = _apply_dca(
            forward_data, forward_idx, f_exits, x=arm_x, y=arm_y, literal=False, dca_first=dca_first
        )
        output = _combine_outputs([old_out, f_out])
        diag = {key: np.concatenate([old_diag[key], f_diag[key]]) for key in old_diag}
        metrics = _metric(july_data, july_idx, output, july_sizes, diag, x=arm_x, y=arm_y, literal=False)
        records.append({"policy": policy, **metrics})
        weekly, ledger = _weekly_ledger(july_rows, output, july_sizes, diag, policy=policy)
        weekly_parts.append(weekly); ledger_parts.append(ledger)
    frame = pd.DataFrame(records)
    baseline = float(frame.loc[frame.policy.eq("winner_baseline"), "net_pnl_bankroll"].iloc[0])
    frame["delta_net_pnl_vs_winner"] = frame["net_pnl_bankroll"] - baseline
    frame.to_csv(OUTPUT / "july_frozen_metrics.csv", index=False)
    pd.concat(weekly_parts, ignore_index=True).to_csv(OUTPUT / "july_weekly_metrics.csv", index=False)
    pd.concat(ledger_parts, ignore_index=True).to_parquet(OUTPUT / "july_trade_ledger.parquet", index=False)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
