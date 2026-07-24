#!/usr/bin/env python3
"""Daily July replay table for the joint-trailing + raw-Bayesian winner.

July 1--16 are read from the frozen matched replay.  Optional July 17
historical prediction rows are replayed on exact 1m paths, with the preceding
selected ledger prepended so portfolio-capacity state carries over exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_ablation import evaluate_results
from extreme_price_movements.simple_policy_1m_constrained import (
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from extreme_price_movements.simple_policy_optimiser import _with_policy_spread_cost_columns
from scripts.report_simple_policy_1m_winner_forward_july import BASE, CHAMPION
from scripts.run_simple_policy_1m_capital_ablation import (
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (
    ExperimentData,
    _causal_entry_atr,
    _indices_between,
)
from scripts.run_simple_policy_1m_contextual_ablation import (
    _bayesian_sizes,
    _load_atr,
    _load_context,
)


PRIOR = CHAMPION / "forward_replay_jul11_17_v1/selected_trade_ledger.parquet"
OLD_CANDIDATES = BASE / "execution_candidates_may_july_v1/simple_policy_candidates_with_archetypes.parquet"
RICH = BASE / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
POSTERIOR = BASE / "complete_parent_state_july_v1/complete_oos_residual_event_states.parquet"
PARENT = BASE / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
OLD_ATR = CHAMPION / "replay/causal_entry_atr_audit.parquet"
OLD_CACHE = CHAMPION / "replay/path_cache"
PARAMS = CHAMPION / "evidence/nested_params.json"
STORE = Path("data_perp/exchanges/krakenfutures/execution_1m")


def _side_number(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return np.where(numeric.fillna(1.0).to_numpy() > 0.0, 1, -1)
    return np.where(values.astype(str).str.lower().str.startswith("short"), -1, 1)


def _prediction_candidates(path: Path, cutoff: pd.Timestamp, spread_reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(path)
    raw["signal_bar_ts"] = pd.to_datetime(raw["signal_bar_ts"], utc=True)
    raw = raw.loc[
        raw["threshold_basis_selected"].fillna(False)
        & raw["signal_bar_ts"].ge(pd.Timestamp("2026-07-17", tz="UTC"))
        & raw["signal_bar_ts"].lt(cutoff)
    ].copy()
    raw = raw.sort_values(
        ["signal_bar_ts", "threshold_basis_rank_score"],
        ascending=[True, False],
        kind="mergesort",
    ).drop_duplicates(["signal_bar_ts", "symbol", "side"], keep="last")
    rows = pd.DataFrame(index=raw.index)
    rows["timestamp"] = raw["signal_bar_ts"]
    rows["symbol"] = raw["symbol"].astype(str)
    rows["side"] = _side_number(raw["side"])
    rows["side_name"] = np.where(rows["side"] > 0, "long", "short")
    rows["strategy_id"] = raw["strategy_id"].astype(str)
    rows["rank_pct"] = pd.to_numeric(raw["threshold_basis_rank_score"], errors="coerce")
    rows["calibrated_score"] = pd.to_numeric(raw["calibrated_score"], errors="coerce")
    rows["policy_archetype"] = raw["policy_archetype"].astype(str)
    rows["local_side_archetype"] = rows["policy_archetype"]
    rows["archetype_policy_key"] = rows["policy_archetype"].str.replace(
        r"^(long|short)__", "", regex=True
    )
    rows["threshold_basis_corrected_expected_ev"] = pd.to_numeric(
        raw["threshold_basis_corrected_expected_ev"], errors="coerce"
    )
    rows["threshold_basis_corrected_expected_ev_rank"] = pd.to_numeric(
        raw["threshold_basis_corrected_expected_ev_rank"], errors="coerce"
    )

    # The historical prediction pass intentionally does not fetch a live
    # ticker.  Carry forward each symbol's latest pre-Jul17 policy spread;
    # fall back to its side/archetype median, then the global median.
    ref = spread_reference.copy()
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], utc=True)
    ref = ref.loc[ref["timestamp"] < pd.Timestamp("2026-07-17", tz="UTC")]
    latest = (
        ref.sort_values("timestamp")
        .dropna(subset=["expected_spread_bps"])
        .drop_duplicates("symbol", keep="last")
        .set_index("symbol")["expected_spread_bps"]
    )
    group_median = ref.groupby(["side_name", "policy_archetype"])["expected_spread_bps"].median()
    global_median = float(pd.to_numeric(ref["expected_spread_bps"], errors="coerce").median())
    spreads = []
    for row in rows.itertuples():
        value = latest.get(row.symbol, np.nan)
        if not np.isfinite(value):
            value = group_median.get((row.side_name, row.policy_archetype), np.nan)
        spreads.append(float(value) if np.isfinite(value) else global_median)
    rows["expected_spread_bps"] = spreads
    rows = _with_policy_spread_cost_columns(rows.reset_index(drop=True), market_mode="perps")

    predecessor_rank = pd.to_numeric(raw["v9_tail95_predecessor_rank"], errors="coerce").clip(0, 1)
    context = pd.DataFrame(
        {
            "expected_net_ev_after_1pct_mlp_direct": pd.to_numeric(
                raw["expected_net_ev_after_1pct"], errors="coerce"
            ).to_numpy(),
            "meta_hit_probability_uncertainty_p1mp": (
                predecessor_rank * (1.0 - predecessor_rank)
            ).to_numpy(),
            "gmm_ood_score": pd.to_numeric(raw["gmm_ood_score"], errors="coerce").to_numpy(),
            # OOD weight is zero in the frozen winner.  Preserve a finite
            # entropy column because the shared sizing helper validates it.
            "cluster_entropy_norm": np.zeros(len(raw), dtype=np.float64),
        }
    )
    return rows, context


def _prepend_capacity(prior: pd.DataFrame, rows: pd.DataFrame, outputs: dict[str, np.ndarray]) -> np.ndarray:
    prefix = prior.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort")
    combined = pd.concat(
        [prefix[["timestamp", "symbol", "rank_pct"]], rows[["timestamp", "symbol", "rank_pct"]]],
        ignore_index=True,
    )
    n0 = len(prefix)
    exit_bars = np.concatenate([prefix["exit_bars"].to_numpy(np.int32), outputs["exit_bars"]])
    gross = np.concatenate([prefix["gross_return"].to_numpy(float), outputs["gross_return"]])
    net = np.concatenate([prefix["net_return"].to_numpy(float), outputs["net_return"]])
    reason = np.concatenate([prefix["exit_reason_code"].to_numpy(np.int8), outputs["reason"]])
    filler = np.full(len(combined), np.nan)
    _, selected = evaluate_results(
        combined, exit_bars, gross, net, reason, filler, filler, bar_minutes=1, apply_capacity=True
    )
    if not selected[:n0].all():
        raise RuntimeError("Prepending the frozen selected ledger changed historical capacity choices")
    return selected[n0:]


def _daily_from_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    work = ledger.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work = work.loc[
        work["timestamp"].ge(pd.Timestamp("2026-07-01", tz="UTC"))
        & work["timestamp"].lt(pd.Timestamp("2026-07-17", tz="UTC"))
    ]
    work["day"] = work["timestamp"].dt.floor("D")
    return (
        work.groupby(["day", "policy"], as_index=False)
        .agg(
            trades=("net_return", "size"),
            net_ev_per_trade=("net_return", "mean"),
            net_pnl_bankroll=("net_pnl_bankroll", "sum"),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-ledger", type=Path, required=True)
    parser.add_argument("--entry-cutoff-exclusive", required=True)
    parser.add_argument("--output-dir", type=Path, default=CHAMPION / "daily_replay_july01_17_v1")
    parser.add_argument("--rebuild-path-cache", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(args.entry_cutoff_exclusive)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")

    prior = pd.read_parquet(PRIOR)
    spread_reference = pd.read_parquet(
        "data_perp/reports/july_01_16_current_policy_metrics_20260717/current_policy_candidates_through_july16.parquet"
    )
    rows, forward_context = _prediction_candidates(args.prediction_ledger, cutoff, spread_reference)
    deployed, _ = _load_deployed_side_params(PARENT)
    spec = ConstrainedReplaySpec()
    atr, atr_audit, atr_manifest = _causal_entry_atr(
        rows, store_root=STORE, deployed_by_side=deployed, parent_summary=PARENT, warmup_hours=48
    )
    atr_audit.to_parquet(args.output_dir / "july17_causal_entry_atr_audit.parquet", index=False)
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows,
        store_root=STORE,
        cache_dir=args.output_dir / "july17_path_cache",
        spec=spec,
        rebuild=args.rebuild_path_cache,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)
    if not valid.all() or not np.isfinite(atr).all():
        raise RuntimeError(
            f"Jul17 replay is incomplete: path={valid.mean():.2%} ATR={np.isfinite(atr).mean():.2%}"
        )

    old_rows = pd.read_parquet(OLD_CANDIDATES)
    old_rows["timestamp"] = pd.to_datetime(old_rows["timestamp"], utc=True)
    old_rows = old_rows.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    old_context, _, _ = _load_context(old_rows, RICH, POSTERIOR)
    old_atr = _load_atr(old_rows, OLD_ATR)
    oo, oh, ol, oc, ov, _ = _load_or_build_path_cache(
        old_rows, store_root=STORE, cache_dir=OLD_CACHE, spec=spec, rebuild=False
    )
    old_data = ExperimentData(old_rows, oo, oh, ol, oc, ov, old_atr, spec, deployed)
    train_idx = _indices_between(old_data, "2026-05-01", "2026-06-14")
    params = json.loads(PARAMS.read_text())
    geometry = params["fold_3"]["full_train_parent"]
    sizing = params["fold_3"]["sizing"]
    train_outputs = old_data.simulate(train_idx, geometry, FAMILY_TRAILING_ONLY)
    combined_rows = pd.concat([old_rows, rows], ignore_index=True, copy=False)
    combined_context = pd.concat([old_context, forward_context], ignore_index=True, copy=False)
    sizing_data = SimpleNamespace(
        rows=combined_rows,
        side=pd.to_numeric(combined_rows["side"], errors="coerce").to_numpy(float),
        rank=pd.to_numeric(combined_rows["rank_pct"], errors="coerce").to_numpy(float),
    )
    apply_idx = np.arange(len(old_rows), len(combined_rows), dtype=np.int64)
    all_sizes, sizing_state = _bayesian_sizes(
        sizing_data,
        train_idx,
        apply_idx,
        train_outputs,
        combined_context,
        strength=float(sizing["strength"]),
        ood_weight=float(sizing["ood_weight"]),
    )
    winner_sizes = all_sizes[apply_idx]
    winner_outputs = data.simulate(np.arange(len(rows)), geometry, FAMILY_TRAILING_ONLY)
    deployed_outputs = data.simulate_deployed(np.arange(len(rows)))

    policy_specs = {
        "joint_trailing_plus_bayesian_raw": (winner_outputs, winner_sizes),
        "current_deployed_reference": (deployed_outputs, np.ones(len(rows))),
    }
    july17_records = []
    new_ledgers = []
    for policy, (outputs, multipliers) in policy_specs.items():
        prefix = prior.loc[prior["policy"].eq(policy)].copy()
        selected = _prepend_capacity(prefix, rows, outputs)
        chosen = np.flatnonzero(selected)
        base = 0.075 + 0.075 * np.power(np.clip(rows["rank_pct"].to_numpy(float), 0, 1), 1.1)
        pnl = outputs["net_return"] * base * multipliers
        july17_records.append(
            {
                "day": pd.Timestamp("2026-07-17", tz="UTC"),
                "policy": policy,
                "trades": int(len(chosen)),
                "net_ev_per_trade": float(np.mean(outputs["net_return"][chosen])) if len(chosen) else np.nan,
                "net_pnl_bankroll": float(np.sum(pnl[chosen])),
            }
        )
        ledger = rows.iloc[chosen][["timestamp", "symbol", "side_name", "policy_archetype", "rank_pct"]].copy()
        ledger["policy"] = policy
        ledger["exit_bars"] = outputs["exit_bars"][chosen]
        ledger["exit_reason_code"] = outputs["reason"][chosen]
        ledger["gross_return"] = outputs["gross_return"][chosen]
        ledger["net_return"] = outputs["net_return"][chosen]
        ledger["base_size"] = base[chosen]
        ledger["size_multiplier"] = multipliers[chosen]
        ledger["net_pnl_bankroll"] = pnl[chosen]
        new_ledgers.append(ledger)

    long = pd.concat([_daily_from_ledger(prior), pd.DataFrame(july17_records)], ignore_index=True)
    winner = long.loc[long["policy"].eq("joint_trailing_plus_bayesian_raw")].set_index("day")
    deployed_daily = long.loc[long["policy"].eq("current_deployed_reference")].set_index("day")
    table = winner[["trades", "net_ev_per_trade", "net_pnl_bankroll"]].rename(
        columns={"net_pnl_bankroll": "winner_pnl"}
    )
    table["deployed_pnl"] = deployed_daily["net_pnl_bankroll"]
    table["delta_pnl"] = table["winner_pnl"] - table["deployed_pnl"]
    table["status"] = "complete"
    table.loc[pd.Timestamp("2026-07-17", tz="UTC"), "status"] = (
        f"partial: entries before {cutoff.isoformat()}"
    )
    table = table.reset_index()
    table.to_csv(args.output_dir / "daily_comparison.csv", index=False)
    long.to_csv(args.output_dir / "daily_metrics_long.csv", index=False)
    pd.concat(new_ledgers, ignore_index=True).to_parquet(
        args.output_dir / "july17_partial_selected_trade_ledger.parquet", index=False
    )
    rows.to_parquet(args.output_dir / "july17_partial_candidates.parquet", index=False)
    manifest = {
        "status": "complete_with_partial_july17",
        "entry_cutoff_exclusive_utc": cutoff.isoformat(),
        "july17_candidate_rows": int(len(rows)),
        "july17_exact_path_rows": int(valid.sum()),
        "july17_atr_rows": int(np.isfinite(atr).sum()),
        "comparison": "matched candidate stream; exact 1m paths; causal entry-frozen ATR; 1% fee once; spread baseline; 8-open/2-new capacity",
        "july17_spread_contract": "latest pre-Jul17 symbol policy spread, with causal side/archetype then global fallback",
        "sizing_state": sizing_state,
        "atr_manifest": atr_manifest,
        "path_manifest": path_manifest,
    }
    (args.output_dir / "daily_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
