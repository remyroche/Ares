#!/usr/bin/env python3
"""Report the canonical base-to-portfolio chain on comparable OOS periods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_ID = "base_global_top10_fixed_activity"
META_ID = "canonical_meta_global_top10_fixed_activity"
V9_ID = (
    "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_"
    "20260712_v9::forced_local_tail_0.950"
)
MLP_ID = "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1"
ADMISSION_ID = "ev_target_side_archetype_global_top10_before_mlp_28d_flat_v1"
EXIT_ID = "side_x_archetype_optimized_exit_and_sizing_v1"
PORTFOLIO_ID = "global_auction_v1"
KEYS = ["__ts__", "__symbol__", "side_name", "archetype_policy_key"]


def _top_n_mask(values: pd.Series, count: int) -> np.ndarray:
    score = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    mask = np.zeros(len(score), dtype=bool)
    finite = np.flatnonzero(np.isfinite(score))
    count = min(max(int(count), 0), len(finite))
    if not count:
        return mask
    chosen = finite[np.argpartition(score[finite], -count)[-count:]]
    mask[chosen] = True
    return mask


def _fixed_monthly_activity_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    month = frame["__ts__"].dt.strftime("%Y-%m")
    parent = np.zeros(len(frame), dtype=bool)
    mlp = np.zeros(len(frame), dtype=bool)
    for month_id in sorted(month.dropna().unique()):
        idx = np.flatnonzero(month.eq(month_id).to_numpy())
        budget = int(
            pd.to_numeric(frame.iloc[idx]["policy_parent_rank"], errors="coerce")
            .ge(0.90)
            .sum()
        )
        parent[idx] = _top_n_mask(frame.iloc[idx]["policy_parent_rank"], budget)
        mlp[idx] = _top_n_mask(frame.iloc[idx]["expected_ev_rank_score"], budget)
    return parent, mlp


def _global_top_fraction_mask(values: pd.Series, fraction: float = 0.10) -> np.ndarray:
    finite = int(pd.to_numeric(values, errors="coerce").notna().sum())
    count = max(1, int(np.ceil(float(fraction) * finite))) if finite else 0
    return _top_n_mask(values, count)


def _stage_rows(
    frame: pd.DataFrame,
    *,
    stage: str,
    timestamp_col: str,
    return_col: str,
    outcome_contract: str,
    bankroll_col: str | None = None,
    rank_col: str | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "stage": stage,
            "timestamp": pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce"),
            "net_return": pd.to_numeric(frame[return_col], errors="coerce"),
            "outcome_contract": outcome_contract,
        }
    )
    if bankroll_col and bankroll_col in frame:
        out["bankroll_pnl"] = pd.to_numeric(frame[bankroll_col], errors="coerce")
    else:
        out["bankroll_pnl"] = np.nan
    out["ranking_score"] = (
        pd.to_numeric(frame[rank_col], errors="coerce")
        if rank_col and rank_col in frame
        else np.nan
    )
    for column in ("side_name", "archetype_policy_key", "__symbol__", "symbol"):
        if column in frame:
            out[column] = frame[column].to_numpy()
    if "__symbol__" not in out and "symbol" in out:
        out["__symbol__"] = out["symbol"]
    return out.dropna(subset=["timestamp", "net_return"]).reset_index(drop=True)


def _period_metrics(rows: pd.DataFrame, period: str) -> pd.DataFrame:
    work = rows.copy()
    if period == "week":
        work["period"] = work["timestamp"].dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    elif period == "month":
        work["period"] = work["timestamp"].dt.strftime("%Y-%m")
    elif period == "day":
        work["period"] = work["timestamp"].dt.floor("D")
    else:
        raise ValueError(period)
    records: list[dict[str, Any]] = []
    for (stage, value), group in work.groupby(["stage", "period"], sort=True):
        ret = group["net_return"].to_numpy(dtype=np.float64)
        records.append(
            {
                "stage": stage,
                "period": value,
                "trades": int(len(group)),
                "net_ev_per_trade": float(np.mean(ret)),
                "sum_notional_net_return": float(np.sum(ret)),
                "positive_trade_rate": float(np.mean(ret > 0.0)),
                "bankroll_pnl": float(group["bankroll_pnl"].sum(min_count=1)),
            }
        )
    return pd.DataFrame(records)


def _global_metrics(rows: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    daily = _period_metrics(rows, "day")
    records: list[dict[str, Any]] = []
    for stage, group in rows.groupby("stage", sort=False):
        week = weekly.loc[weekly["stage"].eq(stage), "net_ev_per_trade"]
        month = monthly.loc[monthly["stage"].eq(stage), "net_ev_per_trade"]
        day = daily.loc[daily["stage"].eq(stage)]
        ret = group["net_return"].to_numpy(dtype=np.float64)
        ranked = group.loc[group["ranking_score"].notna()].copy()
        top5 = ranked.loc[
            _top_n_mask(ranked["ranking_score"], int(np.ceil(len(ranked) * 0.50)))
        ]
        top5_daily = _period_metrics(top5, "day") if not top5.empty else pd.DataFrame()
        top10_daily_ev = day.set_index("period")["sum_notional_net_return"].sort_index()
        top5_daily_ev = (
            top5_daily.set_index("period")["sum_notional_net_return"].sort_index()
            if not top5_daily.empty
            else pd.Series(dtype=float)
        )
        aligned = pd.concat(
            [top10_daily_ev.rename("top10"), top5_daily_ev.rename("top5")],
            axis=1,
            join="inner",
        ).dropna()
        top10_negative = aligned["top10"].lt(0.0).astype(float)
        top5_negative = aligned["top5"].lt(0.0).astype(float)
        records.append(
            {
                "stage": stage,
                "outcome_contract": group["outcome_contract"].iloc[0],
                "trades": int(len(group)),
                "weeks": int(len(week)),
                "trades_per_week": float(len(group) / max(len(week), 1)),
                "net_ev_per_trade": float(np.mean(ret)),
                "sum_notional_net_return": float(np.sum(ret)),
                "positive_trade_rate": float(np.mean(ret > 0.0)),
                "worst_week_ev_per_trade": float(week.min()),
                "q01_week_ev_per_trade": float(week.quantile(0.01)),
                "q10_week_ev_per_trade": float(week.quantile(0.10)),
                "q25_week_ev_per_trade": float(week.quantile(0.25)),
                "q33_week_ev_per_trade": float(week.quantile(0.33)),
                "q50_week_ev_per_trade": float(week.quantile(0.50)),
                "q75_week_ev_per_trade": float(week.quantile(0.75)),
                "q90_week_ev_per_trade": float(week.quantile(0.90)),
                "worst_month_ev_per_trade": float(month.min()),
                "negative_ev_day_rate": float(
                    (day["sum_notional_net_return"] < 0.0).mean()
                ),
                "top5_net_ev_per_trade": float(top5["net_return"].mean())
                if not top5.empty
                else np.nan,
                "top5_negative_ev_day_rate": float(top5_negative.mean())
                if len(top5_negative)
                else np.nan,
                "top10_daily_ev_autocorr_lag1": float(top10_daily_ev.autocorr(lag=1)),
                "top5_daily_ev_autocorr_lag1": float(top5_daily_ev.autocorr(lag=1)),
                "top10_negative_day_indicator_autocorr_lag1": float(
                    top10_negative.autocorr(lag=1)
                ),
                "top5_negative_day_indicator_autocorr_lag1": float(
                    top5_negative.autocorr(lag=1)
                ),
                "negative_day_indicator_corr_top10_top5": float(
                    top10_negative.corr(top5_negative)
                )
                if len(aligned) > 1
                else np.nan,
                "bankroll_pnl": float(group["bankroll_pnl"].sum(min_count=1)),
            }
        )
    output = pd.DataFrame(records)
    metric_cols = [
        column
        for column in output.columns
        if column
        not in {
            "stage",
            "outcome_contract",
            "trades",
            "weeks",
            "trades_per_week",
        }
        and pd.api.types.is_numeric_dtype(output[column])
    ]
    for column in metric_cols:
        output[f"delta_from_previous__{column}"] = output[column].diff()
        output[f"delta_from_base__{column}"] = output[column] - output[column].iloc[0]
    return output


def _weekly_with_negative_days(rows: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    daily = _period_metrics(rows, "day")
    if daily.empty:
        return weekly
    daily["week_start"] = (
        pd.to_datetime(daily["period"], utc=True)
        .dt.to_period("W-SUN")
        .dt.start_time
        .dt.tz_localize("UTC")
    )
    negative = (
        daily.assign(negative_day=daily["sum_notional_net_return"].lt(0.0).astype(int))
        .groupby(["stage", "week_start"], as_index=False)
        .agg(days=("period", "size"), negative_days=("negative_day", "sum"))
        .rename(columns={"week_start": "period"})
    )
    return weekly.merge(negative, on=["stage", "period"], how="left", validate="one_to_one")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-oos", type=Path, required=True)
    parser.add_argument("--chain-rows", type=Path, required=True)
    parser.add_argument("--exit-replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(
        args.meta_oos,
        columns=KEYS
        + [
            "ev_after_1pct",
            "score_base_ev_mapped",
            "score_base_ev_residual_expert_hier_mapped",
        ],
    )
    chain = pd.read_parquet(args.chain_rows)
    for frame in (meta, chain):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if int(meta.duplicated(KEYS).sum()) or int(chain.duplicated(KEYS).sum()):
        raise ValueError("Meta or chain rows are not unique on canonical keys")
    frame = chain.merge(
        meta,
        on=KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_meta"),
    ).sort_values(KEYS, kind="stable").reset_index(drop=True)
    if len(frame) != len(chain) or len(frame) != len(meta):
        raise ValueError(
            f"Canonical row mismatch: meta={len(meta)} chain={len(chain)} overlap={len(frame)}"
        )

    parent_mask, mlp_mask = _fixed_monthly_activity_masks(frame)
    base_mask = _global_top_fraction_mask(frame["score_base_ev_mapped"])
    meta_mask = _global_top_fraction_mask(
        frame["score_base_ev_residual_expert_hier_mapped"]
    )
    stage_frames = [
        _stage_rows(
            frame.loc[base_mask],
            stage=BASE_ID,
            timestamp_col="__ts__",
            return_col="ev_after_1pct",
            outcome_contract="label_net_ev_after_1pct",
            rank_col="score_base_ev_mapped",
        ),
        _stage_rows(
            frame.loc[meta_mask],
            stage=META_ID,
            timestamp_col="__ts__",
            return_col="ev_after_1pct",
            outcome_contract="label_net_ev_after_1pct",
            rank_col="score_base_ev_residual_expert_hier_mapped",
        ),
        _stage_rows(
            frame.loc[parent_mask],
            stage=V9_ID,
            timestamp_col="__ts__",
            return_col="ev_after_1pct",
            outcome_contract="label_net_ev_after_1pct",
            rank_col="policy_parent_rank",
        ),
        _stage_rows(
            frame.loc[mlp_mask],
            stage=MLP_ID,
            timestamp_col="__ts__",
            return_col="ev_after_1pct",
            outcome_contract="label_net_ev_after_1pct",
            rank_col="expected_ev_rank_score",
        ),
        _stage_rows(
            frame.loc[frame["threshold_basis_selected"].fillna(False)],
            stage=ADMISSION_ID,
            timestamp_col="__ts__",
            return_col="ev_after_1pct",
            outcome_contract="label_net_ev_after_1pct",
            rank_col="threshold_basis_rank_score",
        ),
    ]

    exit_rows = pd.read_parquet(args.exit_replay_dir / "exit_policy_rows.parquet")
    stage_frames.append(
        _stage_rows(
            exit_rows,
            stage=EXIT_ID,
            timestamp_col="timestamp",
            return_col="net_return",
            outcome_contract="kraken_15m_optimized_exit_net_after_1pct",
            rank_col="expected_ev_rank_score",
        )
    )
    decisions = pd.read_parquet(
        args.exit_replay_dir / "portfolio_decisions_after_exit_policy.parquet"
    )
    accepted = decisions.loc[decisions["accepted"].fillna(False)].copy()
    accepted["bankroll_pnl"] = (
        pd.to_numeric(accepted["position_size"], errors="coerce")
        * pd.to_numeric(accepted["position_net_return"], errors="coerce")
    )
    accepted["side_name"] = accepted["side"].map({1.0: "long", -1.0: "short"})
    stage_frames.append(
        _stage_rows(
            accepted,
            stage=PORTFOLIO_ID,
            timestamp_col="timestamp",
            return_col="position_net_return",
            outcome_contract="kraken_15m_portfolio_net_after_1pct",
            bankroll_col="bankroll_pnl",
            rank_col="effective_rank_score",
        )
    )
    rows = pd.concat(stage_frames, ignore_index=True, copy=False)
    weekly = _period_metrics(rows, "week")
    weekly = _weekly_with_negative_days(rows, weekly)
    monthly = _period_metrics(rows, "month")
    daily = _period_metrics(rows, "day")
    global_metrics = _global_metrics(rows, weekly, monthly)
    july_daily = daily.loc[
        pd.to_datetime(daily["period"], utc=True, errors="coerce").dt.strftime("%Y-%m").eq("2026-07")
    ].copy()

    rows.to_parquet(args.output_dir / "seven_stage_selected_rows.parquet", index=False, compression="zstd")
    global_metrics.to_csv(args.output_dir / "seven_stage_global_metrics.csv", index=False)
    weekly.to_csv(args.output_dir / "seven_stage_weekly_metrics.csv", index=False)
    monthly.to_csv(args.output_dir / "seven_stage_monthly_metrics.csv", index=False)
    july_daily.to_csv(args.output_dir / "seven_stage_july_daily_metrics.csv", index=False)
    weekly.loc[
        pd.to_datetime(weekly["period"], utc=True, errors="coerce")
        .ge(pd.Timestamp("2026-04-27", tz="UTC"))
    ].to_csv(args.output_dir / "seven_stage_weekly_metrics_may_to_latest.csv", index=False)
    weekly_recent = weekly.loc[
        pd.to_datetime(weekly["period"], utc=True, errors="coerce")
        .ge(pd.Timestamp("2026-04-27", tz="UTC"))
    ].copy()
    weekly_wide = weekly_recent.pivot(
        index="period",
        columns="stage",
        values=["trades", "net_ev_per_trade", "negative_days"],
    )
    weekly_wide.columns = [f"{metric}__{stage}" for metric, stage in weekly_wide.columns]
    weekly_wide.reset_index().to_csv(
        args.output_dir / "seven_stage_weekly_comparison_wide_may_to_latest.csv",
        index=False,
    )
    manifest = {
        "schema": "canonical_seven_stage_metrics_v1",
        "stages": global_metrics["stage"].tolist(),
        "model_stage_contract": "identical global top10 activity; shared label EV outcome",
        "v9_mlp_contract": "same monthly activity",
        "admission_contract": ADMISSION_ID,
        "execution_contract": "Kraken 15m optimized side x archetype geometry; 1% fee once",
        "portfolio_contract": PORTFOLIO_ID,
        "row_overlap": int(len(frame)),
        "evaluation_start": rows["timestamp"].min().isoformat(),
        "evaluation_end": rows["timestamp"].max().isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(global_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
