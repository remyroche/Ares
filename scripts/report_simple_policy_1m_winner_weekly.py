#!/usr/bin/env python3
"""Weekly matched replay for the joint-trailing + raw-Bayesian winner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_ablation import (  # noqa: E402
    REASON_ADVERSE,
    REASON_CAPITAL,
    REASON_FULL_SL,
    REASON_TIMEOUT,
    REASON_TRAILING,
    evaluate_results,
)
from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    ExperimentData,
    _causal_entry_atr,
    _indices_between,
)
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    _bayesian_sizes,
    _load_atr,
    _load_context,
)


BASE = Path(
    "data_perp/reports/meta_v9_recovery_20260717/"
    "residual_state_mda95_hier_newaegmm_downstream_retrain_v1"
)
CHAMPION = Path(
    "data_perp/reports/"
    "simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _empty_outputs(n: int) -> dict[str, np.ndarray]:
    return {
        "exit_bars": np.full(n, -1, dtype=np.int32),
        "exit_price": np.full(n, np.nan),
        "gross_return": np.full(n, np.nan),
        "net_return": np.full(n, np.nan),
        "reason": np.zeros(n, dtype=np.int8),
        "mfe": np.full(n, np.nan),
        "mae": np.full(n, np.nan),
    }


def _weekly_rows(
    data: ExperimentData,
    idx: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    size_multiplier: np.ndarray,
    *,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = data.rows.iloc[idx].reset_index(drop=True)
    _, selected = evaluate_results(
        rows,
        outputs["exit_bars"],
        outputs["gross_return"],
        outputs["net_return"],
        outputs["reason"],
        outputs["mfe"],
        outputs["mae"],
        bar_minutes=1,
        apply_capacity=True,
    )
    valid = np.isfinite(outputs["net_return"]) & (outputs["exit_bars"] >= 0)
    grouping_ts_col = "signal_bar_ts" if "signal_bar_ts" in rows else "timestamp"
    ts = pd.to_datetime(rows[grouping_ts_col], utc=True)
    week_period = ts.dt.tz_localize(None).dt.to_period("W-SUN")
    week = week_period.astype(str)
    base_size = 0.075 + 0.075 * np.power(
        np.clip(pd.to_numeric(rows["rank_pct"], errors="coerce").fillna(0.9), 0.0, 1.0),
        1.1,
    )
    multiplier = np.asarray(size_multiplier, dtype=np.float64)
    pnl = np.asarray(outputs["net_return"], dtype=np.float64) * base_size * multiplier
    gross_pnl = np.asarray(outputs["gross_return"], dtype=np.float64) * base_size * multiplier
    exposure = base_size * multiplier
    side = np.where(pd.to_numeric(rows["side"], errors="coerce").fillna(1.0) > 0, "long", "short")

    ledger_columns = ["timestamp", "symbol", "policy_archetype", "rank_pct"]
    if "signal_bar_ts" in rows:
        ledger_columns.append("signal_bar_ts")
    if "signal_bar_close_ts" in rows:
        ledger_columns.append("signal_bar_close_ts")
    ledger = rows[ledger_columns].copy()
    ledger["policy"] = policy
    ledger["week"] = week
    ledger["side_name"] = side
    ledger["valid_path"] = valid
    ledger["selected"] = selected
    ledger["exit_bars"] = outputs["exit_bars"]
    ledger["exit_reason_code"] = outputs["reason"]
    ledger["gross_return"] = outputs["gross_return"]
    ledger["net_return"] = outputs["net_return"]
    ledger["base_size"] = base_size
    ledger["size_multiplier"] = multiplier
    ledger["position_size"] = exposure
    ledger["gross_pnl_bankroll"] = gross_pnl
    ledger["net_pnl_bankroll"] = pnl

    records: list[dict[str, Any]] = []
    for label in sorted(week.unique()):
        local = week.eq(label).to_numpy()
        chosen = local & selected
        chosen_idx = np.flatnonzero(chosen)
        period = pd.Period(label, freq="W-SUN")
        if not len(chosen_idx):
            continue
        local_pnl = pnl[chosen_idx]
        equity = np.cumsum(local_pnl)
        drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity])[-len(equity) :]
        local_exposure = exposure[chosen_idx]
        local_base = base_size.to_numpy(dtype=np.float64)[chosen_idx]
        local_mult = multiplier[chosen_idx]
        local_reason = outputs["reason"][chosen_idx]
        local_net = outputs["net_return"][chosen_idx]
        local_gross = outputs["gross_return"][chosen_idx]
        local_side = side[chosen_idx]
        observed_start = max(pd.Timestamp(period.start_time, tz="UTC"), ts[local].min())
        observed_end = min(pd.Timestamp(period.end_time, tz="UTC"), ts[local].max())
        observed_days = max((observed_end - observed_start).total_seconds() / 86400.0 + 1.0 / 24.0, 1.0 / 24.0)
        records.append(
            {
                "policy": policy,
                "week": label,
                "week_start_utc": pd.Timestamp(period.start_time, tz="UTC"),
                "week_end_utc": pd.Timestamp(period.end_time, tz="UTC"),
                "observed_through_utc": ts[local].max(),
                "partial_week": bool(ts[local].max() < pd.Timestamp(period.end_time, tz="UTC") - pd.Timedelta(hours=1)),
                "candidate_count": int(local.sum()),
                "valid_path_count": int((local & valid).sum()),
                "trades": int(len(chosen_idx)),
                "trades_per_day": float(len(chosen_idx) / observed_days),
                "gross_pnl_bankroll": float(gross_pnl[chosen_idx].sum()),
                "fee_pnl_bankroll": float((gross_pnl[chosen_idx] - local_pnl).sum()),
                "net_pnl_bankroll": float(local_pnl.sum()),
                "mean_net_return": float(np.mean(local_net)),
                "median_net_return": float(np.median(local_net)),
                "hit_rate": float(np.mean(local_net > 0.0)),
                "max_drawdown": float(drawdown.min()),
                "gross_notional_exposure": float(local_exposure.sum()),
                "base_notional_exposure": float(local_base.sum()),
                "exposure_ratio": float(local_exposure.sum() / max(local_base.sum(), 1e-12)),
                "mean_size_multiplier": float(np.average(local_mult, weights=local_base)),
                "size_multiplier_p10": float(np.quantile(local_mult, 0.10)),
                "size_multiplier_p90": float(np.quantile(local_mult, 0.90)),
                "mean_holding_hours": float(np.mean(outputs["exit_bars"][chosen_idx] + 1) / 60.0),
                "p90_holding_hours": float(np.quantile(outputs["exit_bars"][chosen_idx] + 1, 0.90) / 60.0),
                "full_sl_rate": float(np.mean(local_reason == REASON_FULL_SL)),
                "adverse_exit_rate": float(np.mean(local_reason == REASON_ADVERSE)),
                "capital_protect_rate": float(np.mean(local_reason == REASON_CAPITAL)),
                "trailing_rate": float(np.mean(local_reason == REASON_TRAILING)),
                "timeout_rate": float(np.mean(local_reason == REASON_TIMEOUT)),
                "long_trades": int(np.sum(local_side == "long")),
                "short_trades": int(np.sum(local_side == "short")),
                "long_net_pnl": float(local_pnl[local_side == "long"].sum()),
                "short_net_pnl": float(local_pnl[local_side == "short"].sum()),
                "mean_gross_return": float(np.mean(local_gross)),
            }
        )

    selected_ledger = ledger.loc[ledger["selected"]].copy()
    breakdown = (
        selected_ledger.groupby(["policy", "week", "side_name", "policy_archetype"], dropna=False)
        .agg(
            trades=("net_pnl_bankroll", "size"),
            net_pnl_bankroll=("net_pnl_bankroll", "sum"),
            gross_pnl_bankroll=("gross_pnl_bankroll", "sum"),
            mean_net_return=("net_return", "mean"),
            hit_rate=("net_return", lambda values: float(np.mean(np.asarray(values) > 0.0))),
            gross_notional_exposure=("position_size", "sum"),
        )
        .reset_index()
    )
    return pd.DataFrame(records), breakdown, selected_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CHAMPION / "weekly_replay_through_20260710_v1")
    parser.add_argument("--report-start", default="2026-06-01")
    parser.add_argument("--report-end", default="2026-07-11")
    parser.add_argument("--path-cache-dir", type=Path)
    parser.add_argument(
        "--causal-signal-close-entry",
        action="store_true",
        default=True,
        help="Deprecated compatibility flag; causal signal-close entry is mandatory.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = BASE / "execution_candidates_may_july_v1/simple_policy_candidates_with_archetypes.parquet"
    rich = BASE / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
    posterior = BASE / "complete_parent_state_july_v1/complete_oos_residual_event_states.parquet"
    parent_summary = BASE / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
    params_path = CHAMPION / "evidence/nested_params.json"
    atr_path = CHAMPION / "replay/causal_entry_atr_audit.parquet"
    cache_path = CHAMPION / "replay/path_cache"

    rows = pd.read_parquet(candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, context_audit = _load_context(rows, rich, posterior)
    deployed, _ = _load_deployed_side_params(parent_summary)
    if args.causal_signal_close_entry:
        rows["signal_bar_ts"] = rows["timestamp"]
        rows["signal_bar_close_ts"] = rows["signal_bar_ts"] + pd.Timedelta(hours=1)
        rows["timestamp"] = rows["signal_bar_close_ts"]
        atr, atr_audit, atr_manifest = _causal_entry_atr(
            rows,
            store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"),
            deployed_by_side=deployed,
            parent_summary=parent_summary,
            warmup_hours=48,
        )
        atr_audit.to_parquet(args.output_dir / "causal_entry_atr_audit.parquet", index=False)
    else:
        atr = _load_atr(rows, atr_path)
        atr_manifest = {"source": str(atr_path)}
    spec = ConstrainedReplaySpec()
    cache_path = args.path_cache_dir or (
        args.output_dir / "path_cache" if args.causal_signal_close_entry else cache_path
    )
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows,
        store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"),
        cache_dir=cache_path,
        spec=spec,
        rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)
    params = json.loads(params_path.read_text())

    signal_ts = pd.to_datetime(rows.get("signal_bar_ts", rows["timestamp"]), utc=True)
    report_mask = signal_ts.ge(pd.Timestamp(args.report_start, tz="UTC"))
    report_mask &= signal_ts.lt(pd.Timestamp(args.report_end, tz="UTC"))
    report_idx = np.flatnonzero(report_mask.to_numpy() & data.valid)
    report_rows = rows.iloc[report_idx]
    updated_outputs = _empty_outputs(len(report_idx))
    updated_size = np.ones(len(report_idx), dtype=np.float64)
    fold_contracts: list[dict[str, Any]] = []
    segments = (
        ("fold_1", "2026-05-15", "2026-06-01", "2026-05-01", "2026-05-14"),
        ("fold_2", "2026-06-01", "2026-06-15", "2026-05-01", "2026-05-31"),
        ("fold_3", "2026-06-15", "2026-07-01", "2026-05-01", "2026-06-14"),
    )
    for fold, apply_start, apply_end, train_start, train_end in segments:
        report_signal_ts = pd.to_datetime(
            report_rows.get("signal_bar_ts", report_rows["timestamp"]), utc=True
        )
        positions = np.flatnonzero(
            report_signal_ts.ge(pd.Timestamp(apply_start, tz="UTC")).to_numpy()
            & report_signal_ts.lt(pd.Timestamp(apply_end, tz="UTC")).to_numpy()
        )
        apply_idx = report_idx[positions]
        all_signal_ts = pd.to_datetime(
            data.rows.get("signal_bar_ts", data.rows["timestamp"]), utc=True
        )
        train_idx = np.flatnonzero(
            all_signal_ts.ge(pd.Timestamp(train_start, tz="UTC")).to_numpy()
            & all_signal_ts.lt(pd.Timestamp(train_end, tz="UTC")).to_numpy()
            & data.valid
        ).astype(np.int64)
        fold_params = params[fold]["full_train_parent"]
        train_outputs = data.simulate(train_idx, fold_params, FAMILY_TRAILING_ONLY)
        apply_outputs = data.simulate(apply_idx, fold_params, FAMILY_TRAILING_ONLY)
        sizing = params[fold]["sizing"]
        size_all, size_state = _bayesian_sizes(
            data,
            train_idx,
            apply_idx,
            train_outputs,
            context,
            strength=float(sizing["strength"]),
            ood_weight=float(sizing["ood_weight"]),
        )
        for key in updated_outputs:
            updated_outputs[key][positions] = apply_outputs[key]
        updated_size[positions] = size_all[apply_idx]
        fold_contracts.append(
            {
                "fold": fold,
                "train_start": train_start,
                "train_end_exclusive": train_end,
                "apply_start": apply_start,
                "apply_end_exclusive": apply_end,
                "apply_rows": int(len(apply_idx)),
                "strength": float(sizing["strength"]),
                "ood_weight": float(sizing["ood_weight"]),
                "sizing_state": size_state,
            }
        )

    deployed_outputs = data.simulate_deployed(report_idx)
    updated_weekly, updated_breakdown, updated_ledger = _weekly_rows(
        data,
        report_idx,
        updated_outputs,
        updated_size,
        policy="joint_trailing_plus_bayesian_raw",
    )
    deployed_weekly, deployed_breakdown, deployed_ledger = _weekly_rows(
        data,
        report_idx,
        deployed_outputs,
        np.ones(len(report_idx)),
        policy="current_deployed_reference",
    )
    weekly = pd.concat([updated_weekly, deployed_weekly], ignore_index=True)
    reference = deployed_weekly.set_index("week")
    weekly["delta_net_pnl_vs_deployed"] = [
        float(row.net_pnl_bankroll - reference.loc[row.week, "net_pnl_bankroll"])
        if row.policy != "current_deployed_reference"
        else 0.0
        for row in weekly.itertuples()
    ]
    weekly["delta_hit_rate_vs_deployed"] = [
        float(row.hit_rate - reference.loc[row.week, "hit_rate"])
        if row.policy != "current_deployed_reference"
        else 0.0
        for row in weekly.itertuples()
    ]
    weekly = weekly.sort_values(["week_start_utc", "policy"]).reset_index(drop=True)
    breakdown = pd.concat([updated_breakdown, deployed_breakdown], ignore_index=True)
    ledger = pd.concat([updated_ledger, deployed_ledger], ignore_index=True)

    weekly.to_csv(args.output_dir / "weekly_metrics.csv", index=False)
    breakdown.to_csv(args.output_dir / "weekly_side_archetype_metrics.csv", index=False)
    ledger.to_parquet(args.output_dir / "selected_trade_ledger.parquet", index=False)
    manifest = {
        "status": "complete",
        "evidence": "walk-forward policy-validation OOS for June; untouched frozen holdout for July",
        "policy": "joint_trailing_total_mfe_raw_bayesian_v1",
        "comparison": "same candidates, 1m paths, causal ATR, costs, and 8-open/2-new capacity",
        "entry_start_utc": str(report_rows["timestamp"].min()),
        "entry_cutoff_utc": str(report_rows["timestamp"].max()),
        "latest_exit_observable_utc": str(report_rows["timestamp"].max() + pd.Timedelta(minutes=1440)),
        "candidate_rows": int(len(report_rows)),
        "valid_path_rows": int(data.valid[report_idx].sum()),
        "partial_last_week": True,
        "fold_contracts": fold_contracts,
        "context_audit": context_audit,
        "path_manifest": path_manifest,
        "atr_manifest": atr_manifest,
        "entry_timestamp_contract": (
            "completed hourly signal close" if args.causal_signal_close_entry
            else "legacy stored timestamp"
        ),
        "outputs": ["weekly_metrics.csv", "weekly_side_archetype_metrics.csv", "selected_trade_ledger.parquet"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(weekly.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
