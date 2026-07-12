#!/usr/bin/env python3
"""Audit historical OOS top-tail stability before the April-June test period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
OUTCOMES = (
    "ev_after_1pct",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "timeout",
)
STATE_FEATURES = (
    "xs_dispersion__ob_depth_to_qv_z_x_rvol_z",
    "rv_rel_universe",
    "ret48h_bench_resid",
    "carry_adj_ret_self_z_10h",
    "asset_minus_mkt_oi_chg_1h_rz",
    "asset_minus_mkt_oi_chg_4h_rz",
    "asset_minus_mkt_oi_drawdown_24h",
    "asset_minus_mkt_oi_recovery_fraction_24h",
    "asset_minus_mkt_price_recovery_fraction_24h",
    "asset_minus_mkt_long_flush_intensity_4h",
    "asset_minus_mkt_short_cover_intensity_1h",
    "asset_mkt_liquidation_phase_divergence",
    "asset_mkt_exhaustion_phase_divergence",
    "mkt_median_oi_chg_4h_rz",
    "mkt_pct_oi_chg_4h_rz_lt_minus2",
    "mkt_oi_flush_breadth_accel_1h",
    "mkt_oi_flush_breadth_recovery_4h",
    "mkt_pct_price_down_oi_down_4h",
    "mkt_pct_price_up_oi_down_1h",
    "market_breadth_recovery_from_24h_min",
    "market_breadth_drawdown_from_6h_max",
    "market_pct_recovering_from_24h_low",
    "market_pc1_variance_share_12h",
    "market_pc1_variance_share_chg_4h",
    "market_downside_pairwise_corr_24h",
    "mkt_systemic_deleveraging_score",
    "mkt_flush_exhaustion_score",
    "mkt_leverage_rebuild_score",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load(path: Path) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    requested = [
        *KEYS,
        "score_meta_base_soft_label",
        "oos_fold",
        *OUTCOMES,
        *STATE_FEATURES,
    ]
    frame = pd.read_parquet(
        path, columns=[name for name in requested if name in available]
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame = frame[
        frame["score_meta_base_soft_label"].notna() & frame["__ts__"].notna()
    ].copy()
    for name in OUTCOMES:
        if name in frame.columns:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(np.float32)
    frame["score_meta_base_soft_label"] = pd.to_numeric(
        frame["score_meta_base_soft_label"], errors="coerce"
    ).astype(np.float32)
    frame["global_batch_rank"] = (
        frame["score_meta_base_soft_label"]
        .groupby(frame["__ts__"], sort=False)
        .rank(method="average", pct=True)
        .astype(np.float32)
    )
    frame["day"] = frame["__ts__"].dt.floor("D")
    frame["week_start"] = frame["day"] - pd.to_timedelta(
        frame["day"].dt.weekday.to_numpy(), unit="D"
    )
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    return frame


def _aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    grouped: Iterable[tuple[Any, pd.DataFrame]] = (
        [((), frame)]
        if not groups
        else frame.groupby(groups, observed=True, dropna=False, sort=True)
    )
    rows: list[dict[str, Any]] = []
    for key, part in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        ev = pd.to_numeric(part["ev_after_1pct"], errors="coerce")
        record: dict[str, Any] = {
            "selected_rows": int(len(part)),
            "symbols": int(part["__symbol__"].nunique()),
            "timestamps": int(part["__ts__"].nunique()),
            "mean_ev_after_1pct": float(ev.mean()),
            "sum_ev_after_1pct": float(ev.sum(min_count=1)),
            "positive_ev_rate": float(ev.gt(0.0).mean()),
        }
        for name in OUTCOMES[1:]:
            if name in part.columns:
                record[f"{name}_rate"] = float(
                    pd.to_numeric(part[name], errors="coerce").mean()
                )
        for name, value in zip(groups, keys, strict=False):
            record[name] = value
        rows.append(record)
    return pd.DataFrame(rows)


def _bad_periods(metrics: pd.DataFrame, period_col: str) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    values = pd.to_numeric(metrics["mean_ev_after_1pct"], errors="coerce")
    q10 = float(values.quantile(0.10))
    out = metrics.copy()
    out["bottom_decile_cutoff"] = q10
    out["negative_ev"] = values.le(0.0)
    out["bottom_decile"] = values.le(q10)
    return out.loc[out["negative_ev"] | out["bottom_decile"]].sort_values(
        ["mean_ev_after_1pct", period_col], kind="stable"
    )


def _feature_contrasts(selected: pd.DataFrame, *, local_breakout: bool) -> pd.DataFrame:
    work = selected
    scope = "short_breakout_impulse" if local_breakout else "global_top10"
    if local_breakout:
        work = work.loc[
            work["side_name"].astype(str).str.lower().eq("short")
            & work["archetype_policy_key"]
            .astype(str)
            .str.contains("breakout", case=False, na=False)
        ]
    if work.empty:
        return pd.DataFrame()
    daily_ev = work.groupby("day", observed=True)["ev_after_1pct"].mean()
    bad_days = set(daily_ev[daily_ev.le(0.0)].index)
    daily_features = work.groupby("day", observed=True)[
        [name for name in STATE_FEATURES if name in work.columns]
    ].median(numeric_only=True)
    rows: list[dict[str, Any]] = []
    bad_mask = daily_features.index.isin(bad_days)
    for name in daily_features.columns:
        values = pd.to_numeric(daily_features[name], errors="coerce")
        bad = values[bad_mask].dropna()
        good = values[~bad_mask].dropna()
        if len(bad) < 3 or len(good) < 3:
            continue
        iqr = float(values.quantile(0.75) - values.quantile(0.25))
        delta = float(bad.median() - good.median())
        rows.append(
            {
                "scope": scope,
                "feature": name,
                "bad_days": int(len(bad)),
                "good_days": int(len(good)),
                "bad_median": float(bad.median()),
                "good_median": float(good.median()),
                "median_delta": delta,
                "robust_effect": delta / max(abs(iqr), 1e-6),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "robust_effect", key=lambda x: x.abs(), ascending=False
    )


def _autocorrelation(daily_local: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (side, archetype), group in daily_local.groupby(
        ["side_name", "archetype_policy_key"], observed=True, dropna=False
    ):
        values = group.sort_values("day")["mean_ev_after_1pct"]
        loss = values.lt(0.0).astype(np.float32)
        rows.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "days": int(len(values)),
                "mean_ev_after_1pct": float(values.mean()),
                "negative_day_rate": float(loss.mean()),
                "ev_autocorr_lag1": float(values.autocorr(1))
                if len(values) >= 3
                else np.nan,
                "loss_autocorr_lag1": float(loss.autocorr(1))
                if len(loss) >= 3
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2025-03-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-04-01T00:00:00Z")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = _load(args.input)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end_exclusive)
    frame = frame.loc[frame["__ts__"].ge(start) & frame["__ts__"].lt(end)].copy()
    selected = frame.loc[frame["global_batch_rank"].ge(0.90)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "daily_top10": _aggregate(selected, ["day"]),
        "weekly_top10": _aggregate(selected, ["week_start"]),
        "monthly_top10": _aggregate(selected, ["month"]),
        "daily_side_archetype_top10": _aggregate(
            selected, ["day", "side_name", "archetype_policy_key"]
        ),
        "weekly_side_archetype_top10": _aggregate(
            selected, ["week_start", "side_name", "archetype_policy_key"]
        ),
    }
    outputs["bad_days"] = _bad_periods(outputs["daily_top10"], "day")
    outputs["bad_weeks"] = _bad_periods(outputs["weekly_top10"], "week_start")
    breakout = selected.loc[
        selected["side_name"].astype(str).str.lower().eq("short")
        & selected["archetype_policy_key"]
        .astype(str)
        .str.contains("breakout", case=False, na=False)
    ]
    outputs["short_breakout_daily"] = _aggregate(breakout, ["day"])
    outputs["short_breakout_weekly"] = _aggregate(breakout, ["week_start"])
    outputs["feature_bad_good_contrasts"] = pd.concat(
        [
            _feature_contrasts(selected, local_breakout=False),
            _feature_contrasts(selected, local_breakout=True),
        ],
        ignore_index=True,
    )
    outputs["side_archetype_autocorrelation"] = _autocorrelation(
        outputs["daily_side_archetype_top10"]
    )
    for name, values in outputs.items():
        values.to_csv(args.output_dir / f"{name}.csv", index=False)

    manifest = {
        "schema": "meta_residual_pre_oos_stability_v1",
        "input": str(args.input),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "candidate_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "days": int(selected["day"].nunique()),
        "weeks": int(selected["week_start"].nunique()),
        "symbols": int(selected["__symbol__"].nunique()),
        "global_top10_contract": "within-timestamp percentile of frozen OOS meta score",
        "cost_contract": "ev_after_1pct includes the existing 1% round-trip cost",
        "bad_days": int(len(outputs["bad_days"])),
        "bad_weeks": int(len(outputs["bad_weeks"])),
        "short_breakout_selected_rows": int(len(breakout)),
        "leakage_contract": (
            "Only frozen OOS meta scores are ranked; realized outcomes are used solely for reporting."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_json_safe(manifest), indent=2), flush=True)
    print("\nWorst days:")
    print(outputs["bad_days"].head(20).to_string(index=False))
    print("\nWorst weeks:")
    print(outputs["bad_weeks"].head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
