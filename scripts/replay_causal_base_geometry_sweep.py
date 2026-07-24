#!/usr/bin/env python3
"""Replay a fixed base top-k stream under a few causal trailing geometries.

This is deliberately a geometry-only diagnostic: the selected rows and base
scores are fixed. Each arm starts at the signal-close executable path and uses
the same round-trip cost as the labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_first_touch_capture_proxy import (  # noqa: E402
    _fetch_policy_paths,
    _trailing_profit_capture_outcome,
)
from scripts.run_label_widestop_capture_proxy import CaptureArm  # noqa: E402


DEFAULT_LEDGER = Path(
    "data_perp/reports/dae150k_gmm_density_ablation_20260720_candleclose_v2/"
    "base/dae150k_reference__s1__whitened__diag__k6__r0.001/"
    "best_oos_scored_ledger.parquet"
)
DEFAULT_LABELS = Path(
    "data_perp/artifacts/20260720_s59_h5_fullthroughjul10_"
    "candleclose_trailing_cost100bps_labels/labels"
)
DEFAULT_OUT = Path("data_perp/reports/causal_base_geometry_sweep_20260720_v1")


GEOMETRIES: dict[str, dict[str, CaptureArm]] = {
    "parent_defaults": {
        "long": CaptureArm("long_parent", 0.40, 1.00, 24.0, 0.05, 0.25),
        "short": CaptureArm("short_parent", 0.70, 1.00, 16.0, 0.05, 0.25),
    },
    "tight_fast": {
        "long": CaptureArm("long_tight_fast", 0.35, 0.75, 16.0, 0.05, 0.15),
        "short": CaptureArm("short_tight_fast", 0.50, 0.75, 12.0, 0.05, 0.20),
    },
    "wide_slow": {
        "long": CaptureArm("long_wide_slow", 0.50, 1.25, 32.0, 0.05, 0.25),
        "short": CaptureArm("short_wide_slow", 0.75, 1.25, 24.0, 0.05, 0.30),
    },
    "higher_activation": {
        "long": CaptureArm("long_high_activation", 0.60, 1.00, 24.0, 0.05, 0.25),
        "short": CaptureArm("short_high_activation", 0.90, 1.00, 20.0, 0.05, 0.30),
    },
    "wider_trail": {
        "long": CaptureArm("long_wider_trail", 0.40, 1.00, 24.0, 0.05, 0.35),
        "short": CaptureArm("short_wider_trail", 0.70, 1.00, 16.0, 0.05, 0.35),
    },
}


def _side_name(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame:
        return frame["side_name"].astype(str).str.lower()
    numeric = pd.to_numeric(frame["side"], errors="coerce")
    return pd.Series(np.where(numeric.lt(0.0), "short", "long"), index=frame.index)


def _top_rows(
    ledger: Path,
    top_fraction: float,
    *,
    selection_basis: str,
) -> pd.DataFrame:
    columns = [
        "__ts__", "__symbol__", "__barrier_pct__", "score", "side", "side_name",
        "selected_top10", "__u_policy_net__", "__first_touch_stop__",
        "__first_touch_timeout__", "__path_full_bad_mae_1r__",
    ]
    available = pq.ParquetFile(ledger).schema.names
    frame = pd.read_parquet(ledger, columns=[c for c in columns if c in available])
    if (
        selection_basis == "global"
        and abs(float(top_fraction) - 0.10) < 1e-12
        and "selected_top10" in frame
    ):
        selected = frame["selected_top10"].astype(bool)
    else:
        score = pd.to_numeric(frame["score"], errors="coerce")
        if selection_basis == "per_side":
            sides = _side_name(frame)
            selected = score.groupby(sides, observed=True).rank(
                method="first", pct=True, ascending=False
            ).le(float(top_fraction))
        else:
            selected = score.rank(method="first", pct=True, ascending=False).le(
                float(top_fraction)
            )
    frame = frame.loc[selected].copy()
    frame["side_name"] = _side_name(frame)
    frame["timestamp"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    frame["symbol"] = frame["__symbol__"].astype(str)
    frame = frame.dropna(subset=["timestamp", "symbol", "__barrier_pct__"])
    return frame


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    ev = pd.to_numeric(frame["net_ev"], errors="coerce")
    weeks = frame.assign(
        week_start=pd.to_datetime(frame["__ts__"], utc=True).dt.floor("D")
        - pd.to_timedelta(pd.to_datetime(frame["__ts__"], utc=True).dt.weekday, unit="D")
    ).groupby("week_start", observed=True)["net_ev"].mean()
    return {
        "selected_rows": int(len(frame)),
        "mean_net_ev": float(ev.mean()),
        "mean_gross_return": float(pd.to_numeric(frame["gross_return"], errors="coerce").mean()),
        "positive_ev_rate": float(ev.gt(0.0).mean()),
        "stop_rate": float(pd.to_numeric(frame["stop"], errors="coerce").mean()),
        "timeout_rate": float(pd.to_numeric(frame["timeout"], errors="coerce").mean()),
        "worst_week_net_ev": float(weeks.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument(
        "--selection-basis",
        choices=("global", "per_side"),
        default="global",
        help="Rank globally or independently within each side before replay.",
    )
    parser.add_argument("--round-trip-cost", type=float, default=0.01)
    parser.add_argument("--path-len", type=int, default=96)
    args = parser.parse_args()

    rows = _top_rows(
        args.ledger,
        args.top_fraction,
        selection_basis=str(args.selection_basis),
    )
    paths_by_side: dict[str, tuple[pd.DataFrame, tuple[np.ndarray, ...]]] = {}
    for side in ("long", "short"):
        side_rows = rows.loc[rows["side_name"].eq(side)].copy()
        executable, paths, _ = _fetch_policy_paths(
            side_rows,
            labels_path=args.labels_dir,
            side=side,
            data_root=Path("data_perp"),
            market_mode="perps",
            exchange="krakenfutures",
            path_len=int(args.path_len),
            apply_delayed_entry=False,
            entry_delay_hours=1,
            timeframe="1h",
        )
        if "__barrier_pct__" not in executable and "barrier_pct" in executable:
            executable["__barrier_pct__"] = pd.to_numeric(
                executable["barrier_pct"], errors="coerce"
            )
        if "__ts__" not in executable and "timestamp" in executable:
            executable["__ts__"] = pd.to_datetime(executable["timestamp"], utc=True)
        if "__symbol__" not in executable and "symbol" in executable:
            executable["__symbol__"] = executable["symbol"].astype(str)
        # The replay loader returns only execution columns. The side is fixed by
        # this branch, so restore it explicitly for reporting and geometry use.
        executable["side_name"] = side
        paths_by_side[side] = (executable, paths)

    output_rows: list[pd.DataFrame] = []
    summary: list[dict[str, Any]] = []
    for geometry_name, by_side in GEOMETRIES.items():
        arm_rows: list[pd.DataFrame] = []
        for side, arm in by_side.items():
            source, paths = paths_by_side[side]
            outcome = _trailing_profit_capture_outcome(
                source,
                paths,
                arm,
                side_name=side,
                round_trip_cost=float(args.round_trip_cost),
            )
            local = source.loc[:, ["__ts__", "__symbol__", "side_name"]].copy()
            local["geometry"] = geometry_name
            local["net_ev"] = outcome["capture_net"].to_numpy(dtype=np.float64)
            local["gross_return"] = outcome["capture_gross"].to_numpy(dtype=np.float64)
            local["stop"] = outcome["capture_stop"].to_numpy(dtype=np.float64)
            local["timeout"] = outcome["capture_timeout"].to_numpy(dtype=np.float64)
            arm_rows.append(local)
        combined = pd.concat(arm_rows, ignore_index=True)
        output_rows.append(combined)
        for scope, scoped in [("overall", combined), *[(side, combined.loc[combined.side_name.eq(side)]) for side in ("long", "short")]]:
            row = {"geometry": geometry_name, "scope": scope, **_metrics(scoped)}
            summary.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(output_rows, ignore_index=True).to_parquet(args.out_dir / "geometry_ledger.parquet", index=False)
    pd.DataFrame(summary).to_csv(args.out_dir / "geometry_summary.csv", index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "ledger": str(args.ledger), "labels_dir": str(args.labels_dir),
        "top_fraction": float(args.top_fraction), "round_trip_cost": float(args.round_trip_cost),
        "selection_basis": str(args.selection_basis),
        "path_contract": "signal_timestamp_plus_1h_then_first_15m_open",
        "geometries": {name: {side: arm.__dict__ for side, arm in value.items()} for name, value in GEOMETRIES.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
