#!/usr/bin/env python3
"""Score a residual-only meta bundle and report causal side-level daily EV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.inference.side_residual_expert import (
    SideResidualExpertBundle,
)
from scripts.run_label_first_touch_capture_proxy import (
    _fetch_policy_paths,
    _first_touch_capture_outcome,
)
from scripts.run_label_widestop_capture_proxy import CaptureArm


SCORE_COLUMN = "score_base_ev_residual_expert_hier_mapped"
GEOMETRY_COLUMNS = {
    "tp_r": "__archetype_policy_tp_r__",
    "sl_r": "__archetype_policy_sl_r__",
    "trail_r": "__archetype_policy_trail_r__",
    "max_bars_to_mfe": "__archetype_policy_max_bars_to_mfe__",
    "max_barrier": "__archetype_policy_max_barrier__",
}


def _select_top_fraction(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    score = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    eligible = frame.loc[np.isfinite(score)].copy()
    count = max(1, int(math.ceil(len(eligible) * float(fraction))))
    return eligible.sort_values(SCORE_COLUMN, ascending=False, kind="stable").head(count)


def _geometry_key(frame: pd.DataFrame) -> pd.Series:
    values = []
    for output, source in GEOMETRY_COLUMNS.items():
        numeric = pd.to_numeric(frame[source], errors="coerce")
        if output == "max_bars_to_mfe":
            numeric = numeric.fillna(24.0)
        elif output == "max_barrier":
            numeric = numeric.fillna(0.05)
        elif output == "tp_r":
            numeric = numeric.fillna(0.40)
        elif output == "sl_r":
            numeric = numeric.fillna(1.00)
        else:
            numeric = numeric.fillna(0.25)
        values.append(numeric.astype(np.float32))
    return pd.Series(list(zip(*(value.tolist() for value in values))), index=frame.index)


def _capture_selected(
    selected: pd.DataFrame,
    *,
    data_root: Path,
    path_len: int,
    exchange: str,
    side: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = selected.reset_index(drop=True).copy()
    rows, paths, path_stats = _fetch_policy_paths(
        selected,
        labels_path=Path(f"residual_only_july_{side}_top10"),
        side=side,
        data_root=data_root,
        market_mode="perps",
        exchange=exchange,
        path_len=int(path_len),
        apply_delayed_entry=False,
        timeframe="1h",
    )
    del rows
    selected["_geometry_key"] = _geometry_key(selected)
    outcome_parts: list[pd.DataFrame] = []
    for key, index in selected.groupby("_geometry_key", sort=False).groups.items():
        pos = np.asarray(list(index), dtype=np.int64)
        tp_r, sl_r, trail_r, max_bars, max_barrier = key
        arm = CaptureArm(
            name="conditioned_trailing",
            tp_r=float(tp_r),
            sl_r=float(sl_r),
            trail_r=float(trail_r),
            max_bars_to_mfe=float(max_bars),
            max_barrier=float(max_barrier),
        )
        local_paths = tuple(np.asarray(values)[pos] for values in paths)
        local = _first_touch_capture_outcome(
            selected.iloc[pos],
            local_paths,
            arm,
            side_name=side,
            outcome_mode="trailing_profit",
            round_trip_cost=0.01,
            target_mode="path_ordered",
            executable_cost_floor=0.01,
        )
        local["_position"] = pos
        outcome_parts.append(local)
    outcomes = pd.concat(outcome_parts, ignore_index=True).sort_values("_position")
    outcomes = outcomes.drop(columns="_position").reset_index(drop=True)
    return pd.concat([selected.drop(columns="_geometry_key"), outcomes], axis=1), path_stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--existing-oos",
        type=Path,
        help="Exact expanding-window OOS scores to prefer on overlapping rows.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--start", default="2026-07-01T00:00:00Z")
    parser.add_argument("--end-exclusive", default="2026-07-21T00:00:00Z")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--path-len", type=int, default=96)
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--side", choices=("long", "short"), default="long")
    args = parser.parse_args()

    bundle = SideResidualExpertBundle.load(args.bundle)
    required = bundle.required_input_features(args.side)
    columns = list(
        dict.fromkeys(
            [
                "__ts__",
                "__symbol__",
                "side_name",
                "archetype_policy_key",
                "score",
                "__barrier_pct__",
                *GEOMETRY_COLUMNS.values(),
                *required,
            ]
        )
    )
    schema = pd.read_parquet(args.candidates, columns=[]).columns
    # PyArrow validates requested columns, so inspect the actual schema once.
    import pyarrow.parquet as pq

    available = set(pq.read_schema(args.candidates).names)
    missing = sorted(set(columns) - available - {"score_base"})
    if missing:
        raise ValueError(f"Candidate artifact is missing required columns: {missing[:20]}")
    frame = pd.read_parquet(args.candidates, columns=[c for c in columns if c in available])
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end_exclusive)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    frame = frame.loc[
        frame["__ts__"].ge(start)
        & frame["__ts__"].lt(end)
        & frame["side_name"].astype(str).str.lower().eq(args.side)
    ].copy()
    frame["score_base"] = pd.to_numeric(frame["score"], errors="coerce").astype(np.float32)
    scored = bundle.transform(frame)
    frame = pd.concat([frame, scored], axis=1)
    frame["score_provenance"] = "final_refit_forward"
    oos_replaced = 0
    oos_max_ts: pd.Timestamp | None = None
    if args.existing_oos is not None:
        oos_columns = ["__ts__", "__symbol__", "side_name", SCORE_COLUMN]
        oos = pd.read_parquet(args.existing_oos, columns=oos_columns)
        oos["__ts__"] = pd.to_datetime(oos["__ts__"], utc=True, errors="coerce")
        oos = oos.loc[
            oos["__ts__"].ge(start)
            & oos["__ts__"].lt(end)
            & oos["side_name"].astype(str).str.lower().eq(args.side)
        ].copy()
        oos[SCORE_COLUMN] = pd.to_numeric(oos[SCORE_COLUMN], errors="coerce")
        oos = oos.dropna(subset=["__ts__", "__symbol__", SCORE_COLUMN])
        oos = oos.drop_duplicates(["__ts__", "__symbol__", "side_name"], keep="last")
        oos_max_ts = oos["__ts__"].max() if not oos.empty else None
        oos = oos.rename(columns={SCORE_COLUMN: "_exact_oos_score"})
        frame = frame.merge(oos, on=["__ts__", "__symbol__", "side_name"], how="left")
        exact_oos = np.isfinite(pd.to_numeric(frame["_exact_oos_score"], errors="coerce"))
        frame.loc[exact_oos, SCORE_COLUMN] = frame.loc[exact_oos, "_exact_oos_score"]
        frame.loc[exact_oos, "score_provenance"] = "expanding_window_oos"
        oos_replaced = int(exact_oos.sum())
        frame = frame.drop(columns="_exact_oos_score")
    complete = np.isfinite(pd.to_numeric(frame[SCORE_COLUMN], errors="coerce"))
    selected = _select_top_fraction(frame.loc[complete], args.top_fraction)
    captured, path_stats = _capture_selected(
        selected,
        data_root=args.data_root,
        path_len=args.path_len,
        exchange=args.exchange,
        side=args.side,
    )
    valid = captured["capture_valid_path"].eq(1.0)
    evaluated = captured.loc[valid].copy()
    evaluated["utc_day"] = evaluated["__ts__"].dt.strftime("%Y-%m-%d")
    daily = (
        evaluated.groupby("utc_day", sort=True, observed=True)
        .agg(
            selected_trades=("capture_net", "size"),
            net_ev_per_trade=("capture_net", "mean"),
            total_net_ev=("capture_net", "sum"),
            positive_trade_rate=("capture_net", lambda x: float((x > 0).mean())),
            stop_rate=("capture_stop", "mean"),
            timeout_rate=("capture_timeout", "mean"),
        )
        .reset_index()
    )
    calendar = pd.DataFrame(
        {"utc_day": pd.date_range(start.normalize(), end.normalize(), inclusive="left").strftime("%Y-%m-%d")}
    )
    daily = calendar.merge(daily, on="utc_day", how="left")
    daily["selected_trades"] = daily["selected_trades"].fillna(0).astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    captured.to_parquet(args.output_dir / f"{args.side}_top10_scored_outcomes.parquet", index=False)
    daily.to_csv(args.output_dir / f"{args.side}_top10_daily_metrics.csv", index=False)
    manifest = {
        "schema": "residual_only_july_side_daily_v2",
        "candidates": str(args.candidates),
        "bundle": str(args.bundle),
        "existing_oos": str(args.existing_oos) if args.existing_oos else None,
        "score": SCORE_COLUMN,
        "side": args.side,
        "selection_scope": f"global_{args.side}_side_over_requested_period",
        "top_fraction": float(args.top_fraction),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "candidate_side_rows": int(len(frame)),
        "complete_case_rows": int(complete.sum()),
        "expanding_window_oos_rows": oos_replaced,
        "expanding_window_oos_max_ts": oos_max_ts.isoformat() if oos_max_ts is not None else None,
        "final_refit_forward_rows": int((frame["score_provenance"] == "final_refit_forward").sum()),
        "selected_rows": int(len(selected)),
        "valid_outcome_rows": int(valid.sum()),
        "trades_per_calendar_day": float(valid.sum() / max(len(calendar), 1)),
        "net_ev_per_trade": float(evaluated["capture_net"].mean()),
        "total_net_ev": float(evaluated["capture_net"].sum()),
        "round_trip_cost": 0.01,
        "path_len_15m_bars": int(args.path_len),
        "path_stats": path_stats,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str))
    print(daily.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
