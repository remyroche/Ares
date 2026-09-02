#!/usr/bin/env python3
"""Compare direct future-EV replacement with a bounded recovery overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _longest_zero(values: pd.Series) -> int:
    best = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _shortlist(metrics: pd.DataFrame) -> pd.DataFrame:
    development = metrics.loc[metrics["period"].astype(str).eq("2025")].copy()
    # Preserve both target semantics and every requested model family/depth.
    # The shortlist is frozen exclusively by chronological 2025 OOF forecast
    # quality; no 2026 economic value participates in selection.
    development = development.sort_values(
        ["forecast_target_ic", "median_monthly_ic", "worst_monthly_ic", "mae_bps"],
        ascending=[False, False, False, True], kind="stable",
    )
    keys = ["target_family", "model_spec"]
    return development.groupby(keys, sort=False, as_index=False).head(1).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funnel-root", type=Path, required=True)
    parser.add_argument("--future-ev-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    metrics = pd.read_csv(args.future_ev_root / "future_ev_daily_architecture_a_metrics.csv")
    shortlist = _shortlist(metrics)
    shortlist.to_csv(args.out_dir / "development_shortlist.csv", index=False)
    forecasts = pd.read_parquet(args.future_ev_root / "future_ev_combined_forecasts.parquet")
    forecasts["snapshot_utc"] = pd.to_datetime(forecasts["snapshot_utc"], utc=True)
    keys = ["target_family", "normalization", "model_spec", "calibrator", "forecast_combination"]
    selected = forecasts.merge(shortlist[keys].drop_duplicates(), on=keys, how="inner", validate="many_to_many")
    state = pd.read_parquet(args.funnel_root / "recovery_state_features.parquet", columns=["snapshot_utc", "ev_28d"])
    state["snapshot_utc"] = pd.to_datetime(state["snapshot_utc"], utc=True)
    selected = selected.merge(state, on="snapshot_utc", how="left", validate="many_to_one")
    selected["config_id"] = selected[keys].astype(str).agg("__".join, axis=1)
    selected.to_parquet(args.out_dir / "selected_daily_forecasts.parquet", index=False)

    connection = duckdb.connect()
    connection.execute("set TimeZone='UTC'")
    connection.execute("set memory_limit='6GB'")
    connection.execute(f"set temp_directory='{(args.out_dir / 'duckdb_tmp').as_posix()}'")
    connection.register("selected_forecasts", selected.loc[:, ["snapshot_utc", "config_id", "forecast_expected_ev_bps", "ev_28d"]])
    maps = args.funnel_root / "multiwindow_ev_maps.parquet"
    summary_rows: list[dict[str, object]] = []
    daily_tables: list[pd.DataFrame] = []

    # Architecture A is a daily replacement map: it directly admits the
    # frozen score's top two per timestamp when forecast EV clears +50 bps.
    daily_outcome = selected.loc[:, ["snapshot_utc", "config_id", "forecast_expected_ev_bps", "top2_net_sum", "top2_resolved_trades"]].drop_duplicates()
    daily_outcome["architecture"] = "A_direct_future_ev_map"
    daily_outcome["trades"] = daily_outcome["top2_resolved_trades"].where(daily_outcome["forecast_expected_ev_bps"].ge(50.0), 0).fillna(0).astype(int)
    daily_outcome["total_net_bps"] = daily_outcome["top2_net_sum"].where(daily_outcome["forecast_expected_ev_bps"].ge(50.0), 0.0).fillna(0.0)
    daily_outcome["net_bps_per_trade"] = daily_outcome["total_net_bps"] / daily_outcome["trades"].replace(0, np.nan)
    daily_tables.append(daily_outcome.loc[:, ["snapshot_utc", "config_id", "architecture", "trades", "total_net_bps", "net_bps_per_trade"]])

    for config_id in selected["config_id"].unique():
        escaped = config_id.replace("'", "''")
        for architecture, authority in (
            ("B_future_ev_recovery_continuous", "1.0/(1.0+exp(-(f.forecast_expected_ev_bps-f.ev_28d)/50.0))"),
            ("B_future_ev_recovery_bounded50", "0.5/(1.0+exp(-(f.forecast_expected_ev_bps-f.ev_28d)/50.0))"),
        ):
            result = connection.execute(f"""
              with joined as (
                select m.__decision_ts__,m.policy_path_valid,m.policy_net_bps,
                  (1-({authority}))*list_median([m.m28,m.m35,m.m42])
                    +({authority})*0.5*(m.m14+m.m21) mapped_ev
                from read_parquet('{maps.as_posix()}') m
                join selected_forecasts f
                  on cast(m.__decision_ts__ as date)=cast(f.snapshot_utc as date)
                where f.config_id='{escaped}'
              )
              select cast(__decision_ts__ as date) as day,
                count(*) filter(where mapped_ev>=50 and policy_path_valid and isfinite(policy_net_bps)) trades,
                sum(policy_net_bps) filter(where mapped_ev>=50 and policy_path_valid and isfinite(policy_net_bps)) total_net_bps,
                avg(policy_net_bps) filter(where mapped_ev>=50 and policy_path_valid and isfinite(policy_net_bps)) net_bps_per_trade
              from joined group by day order by day
            """).fetchdf()
            result["snapshot_utc"] = pd.to_datetime(result.pop("day"), utc=True)
            result["config_id"] = config_id
            result["architecture"] = architecture
            result["trades"] = result["trades"].fillna(0).astype(int)
            result["total_net_bps"] = result["total_net_bps"].fillna(0.0)
            daily_tables.append(result.loc[:, ["snapshot_utc", "config_id", "architecture", "trades", "total_net_bps", "net_bps_per_trade"]])
    daily = pd.concat(daily_tables, ignore_index=True)
    daily.to_parquet(args.out_dir / "architecture_daily_economics.parquet", index=False)
    for (config_id, architecture), block in daily.groupby(["config_id", "architecture"], sort=False):
        for year, local in block.groupby(block["snapshot_utc"].dt.year, sort=True):
            trades = int(local["trades"].sum())
            total = float(local["total_net_bps"].sum())
            monthly = local.assign(month=local["snapshot_utc"].dt.strftime("%Y-%m")).groupby("month").agg(trades=("trades", "sum"), total=("total_net_bps", "sum"))
            monthly["ev"] = monthly["total"] / monthly["trades"].replace(0, np.nan)
            summary_rows.append({
                "config_id": config_id, "architecture": architecture, "period": str(year),
                "trades": trades, "net_bps_per_trade": total / trades if trades else np.nan,
                "total_net_bps": total, "active_days": int(local["trades"].gt(0).sum()),
                "zero_trade_days": int(local["trades"].eq(0).sum()), "max_zero_day_streak": _longest_zero(local["trades"].eq(0)),
                "positive_months": int(monthly["ev"].gt(0).sum()), "months": len(monthly),
                "worst_month_ev_bps": float(monthly["ev"].min()), "median_month_ev_bps": float(monthly["ev"].median()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "architecture_summary.csv", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_future_ev_architecture_comparison_v1", "status": "complete",
        "shortlist_selection": "best chronological 2025 OOF forecast per target family x requested model/depth; 2026 excluded",
        "architecture_a": "daily forecast directly replaces historical EV map and admits frozen top-2 per timestamp at forecast >= +50bps",
        "architecture_b": "forecast-vs-causal-28d opportunity gap controls continuous or max-50% authority between slow 28/35/42 and fast 14/21 candidate maps",
        "resolved_set_rule": "all outcomes with exact-H12 availability <= snapshot; unresolved ignored",
        "shortlist_configs": shortlist[keys].to_dict(orient="records"),
        "forecasts_sha256": _sha(args.future_ev_root / "future_ev_combined_forecasts.parquet"),
        "maps_sha256": _sha(maps),
    }, indent=2) + "\n")
    print(json.dumps({"event": "future_ev_architectures_complete", "configs": int(shortlist[keys].drop_duplicates().shape[0])}))


if __name__ == "__main__":
    main()
