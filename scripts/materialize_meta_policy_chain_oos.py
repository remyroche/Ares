#!/usr/bin/env python3
"""Materialize a causal meta -> EV map -> admission policy candidate chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    _select_ev_curve,
    fit_hierarchical_ev_curves,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _rank_timestamp_side(frame: pd.DataFrame, score_col: str) -> pd.Series:
    return (
        frame.groupby(["__ts__", "side_name"], sort=False)[score_col]
        .rank(method="average", pct=True)
        .astype(np.float32)
    )


def _curve_prediction(curves: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    out = np.full(len(frame), np.nan, dtype=np.float64)
    rank = frame["rank_mlp_direct"].to_numpy(dtype=np.float64)
    strategy = frame["strategy_id"].astype(str).to_numpy()
    side = frame["side_name"].astype(str).to_numpy()
    archetype = frame["policy_archetype"].astype(str).to_numpy()
    keys = np.asarray(
        [f"{sid}|{s}|{a}" for sid, s, a in zip(strategy, side, archetype)],
        dtype=object,
    )
    for key in np.unique(keys):
        mask = keys == key
        sid, side_name, arch = str(key).split("|", 2)
        curve = _select_ev_curve(
            curves,
            strategy_id=sid,
            side=side_name,
            policy_archetype=arch,
        )
        x = np.asarray(curve.get("x", [0.0, 1.0]), dtype=np.float64)
        y = np.asarray(curve.get("y", [0.0, 0.0]), dtype=np.float64)
        out[mask] = np.interp(rank[mask], x, y)
    return out.astype(np.float32)


def _metrics(frame: pd.DataFrame, stage: str) -> dict[str, Any]:
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce")
    days = frame["__ts__"].dt.floor("D").nunique()
    return {
        "stage": stage,
        "rows": int(len(frame)),
        "days": int(days),
        "trades_per_day": float(len(frame) / max(int(days), 1)),
        "mean_net_ev": float(ev.mean()),
        "sum_net_ev": float(ev.sum()),
        "positive_ev_rate": float(ev.gt(0.0).mean()),
        "clean_exec_rate": float(pd.to_numeric(frame["clean_exec"], errors="coerce").mean()),
        "bad_mae_rate": float(pd.to_numeric(frame["first_touch_bad_mae_1r"], errors="coerce").mean()),
        "timeout_rate": float(pd.to_numeric(frame["timeout"], errors="coerce").mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eligible-symbols", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fee-round-trip-pct", type=float, default=0.0015)
    parser.add_argument("--mapping-start", default="2026-04-01")
    parser.add_argument("--mapping-end", default="2026-07-11")
    parser.add_argument("--min-map-history-rows", type=int, default=1000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    eligible = pd.read_csv(args.eligible_symbols)
    symbol_col = "symbol" if "symbol" in eligible else eligible.columns[0]
    spread_col = "p90_spread_bps"
    spread = eligible.set_index(symbol_col)[spread_col]
    frame = frame.loc[frame["__symbol__"].astype(str).isin(spread.index.astype(str))].copy()
    frame["p90_spread_bps"] = frame["__symbol__"].astype(str).map(spread).astype(np.float32)
    frame["ev_after_1pct"] = (
        pd.to_numeric(frame["first_touch_gross"], errors="coerce")
        - float(args.fee_round_trip_pct)
        - frame["p90_spread_bps"] / 10_000.0
    ).astype(np.float32)
    frame["policy_archetype"] = frame["archetype_policy_key"].astype(str)
    frame["strategy_id"] = frame["side_name"].astype(str) + "_s59_meta_oos"
    frame["rank_mlp_direct"] = _rank_timestamp_side(frame, "score_meta_base_soft_label")
    frame["policy_parent_rank"] = frame["rank_mlp_direct"]
    frame["outcome_resolved_at"] = frame["__ts__"] + pd.Timedelta(hours=25)
    frame = frame.sort_values(["__ts__", "side_name", "__symbol__"], kind="stable").reset_index(drop=True)

    start = pd.Timestamp(args.mapping_start, tz="UTC")
    end = pd.Timestamp(args.mapping_end, tz="UTC")
    mapped_parts: list[pd.DataFrame] = []
    map_manifests: list[dict[str, Any]] = []
    for month_start in pd.date_range(start, end, freq="MS", inclusive="left"):
        month_end = min(month_start + pd.offsets.MonthBegin(1), end)
        train = frame.loc[frame["outcome_resolved_at"].lt(month_start)].copy()
        test = frame.loc[frame["__ts__"].ge(month_start) & frame["__ts__"].lt(month_end)].copy()
        if test.empty:
            continue
        if len(train) < int(args.min_map_history_rows):
            raise ValueError(f"insufficient EV-map history before {month_start}: {len(train)}")
        fit_rows = pd.DataFrame(
            {
                "timestamp": train["__ts__"],
                "symbol": train["__symbol__"].astype(str),
                "side": train["side_name"].astype(str),
                "strategy_id": train["strategy_id"],
                "policy_archetype": train["policy_archetype"],
                "normalized_rank_score": train["rank_mlp_direct"],
                "base_strategy_threshold": 0.0,
                "calibrated_score": train["rank_mlp_direct"],
                "entry_price": 1.0,
                "exit_price": 1.0,
                "exit_timestamp": train["__ts__"] + pd.Timedelta(hours=1),
                "holding_bars": 1.0,
                "net_return": train["ev_after_1pct"],
                "gross_return": train["first_touch_gross"],
                "simple_policy_exit_reason": "label_path",
                "fees_bps": float(args.fee_round_trip_pct) * 10_000.0,
            }
        )
        curves = fit_hierarchical_ev_curves(fit_rows, bins=30, min_group_rows=80, shrink_rows=240)
        test["expected_net_ev_after_1pct_mlp_direct"] = _curve_prediction(curves, test)
        test["ev_map_fit_end"] = month_start
        test["ev_map_fit_rows"] = int(len(train))
        mapped_parts.append(test)
        map_manifests.append(
            {
                "test_start": month_start,
                "test_end": month_end,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_max_resolved_at": train["outcome_resolved_at"].max(),
                "side_archetype_curves": len(curves.get("side_archetype", {})),
            }
        )
        (args.out_dir / f"ev_map_{month_start.strftime('%Y%m')}.json").write_text(
            json.dumps(_json_safe(curves), indent=2, sort_keys=True) + "\n"
        )

    mapped = pd.concat(mapped_parts, ignore_index=True, copy=False)
    history = frame.loc[frame["__ts__"].lt(start)].copy()
    history_curve_rows = pd.DataFrame(
        {
            "timestamp": history["__ts__"], "symbol": history["__symbol__"].astype(str),
            "side": history["side_name"].astype(str), "strategy_id": history["strategy_id"],
            "policy_archetype": history["policy_archetype"],
            "normalized_rank_score": history["rank_mlp_direct"], "base_strategy_threshold": 0.0,
            "calibrated_score": history["rank_mlp_direct"], "entry_price": 1.0, "exit_price": 1.0,
            "exit_timestamp": history["__ts__"] + pd.Timedelta(hours=1), "holding_bars": 1.0,
            "net_return": history["ev_after_1pct"], "gross_return": history["first_touch_gross"],
            "simple_policy_exit_reason": "label_path", "fees_bps": float(args.fee_round_trip_pct) * 10_000.0,
        }
    )
    history_curves = fit_hierarchical_ev_curves(history_curve_rows, bins=30, min_group_rows=80, shrink_rows=240)
    history["expected_net_ev_after_1pct_mlp_direct"] = _curve_prediction(history_curves, history)

    history_path = args.out_dir / "ev_mapped_history.parquet"
    mapped_path = args.out_dir / "ev_mapped_oos.parquet"
    history.to_parquet(history_path, index=False, compression="zstd")
    mapped.to_parquet(mapped_path, index=False, compression="zstd")
    metrics = pd.DataFrame([_metrics(history, "mapping_history"), _metrics(mapped, "mapped_oos")])
    metrics.to_csv(args.out_dir / "mapping_stage_metrics.csv", index=False)
    manifest = {
        "schema": "meta_policy_chain_oos_v1",
        "predictions": str(args.predictions),
        "eligible_symbols": str(args.eligible_symbols),
        "eligible_symbol_count": int(frame["__symbol__"].nunique()),
        "fee_round_trip_pct": float(args.fee_round_trip_pct),
        "spread_contract": "pooled per-symbol Kraken p90 full spread deducted once",
        "fee_contract": "round-trip fee deducted once",
        "rank_contract": "timestamp_x_side percentile of OOS meta score",
        "ev_map_contract": "monthly expanding, outcomes resolved before month start, side_x_archetype hierarchical monotone map",
        "history_path": str(history_path),
        "mapped_oos_path": str(mapped_path),
        "folds": map_manifests,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
