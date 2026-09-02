#!/usr/bin/env python3
"""Compare old and rebuilt base scores on identical corrected-label OOS rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["__ts__", "__symbol__", "side_name"]
TARGET = "__first_touch_capture_net__"


def _compact_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace wide string keys with a collision-checked 128-bit row key."""

    key_frame = frame.loc[:, KEYS]
    frame["__join_key_0"] = pd.util.hash_pandas_object(
        key_frame,
        index=False,
        hash_key="0123456789abcdef",
    ).to_numpy(dtype=np.uint64, copy=False)
    frame["__join_key_1"] = pd.util.hash_pandas_object(
        key_frame,
        index=False,
        hash_key="fedcba9876543210",
    ).to_numpy(dtype=np.uint64, copy=False)
    if frame[["__join_key_0", "__join_key_1"]].duplicated().any():
        duplicates = frame.loc[
            frame[["__join_key_0", "__join_key_1"]].duplicated(keep=False),
            KEYS,
        ]
        if not duplicates.duplicated(KEYS).all():
            raise RuntimeError("128-bit comparison-key collision detected")
        raise RuntimeError("comparison input contains duplicate candidate keys")
    return frame.drop(columns=["__symbol__", "side_name"])


def _metric_rows(frame: pd.DataFrame, *, group: str, key: str) -> list[dict]:
    rows: list[dict] = []
    for model in ("previous", "rebuilt"):
        for frac in (0.10, 0.20, 0.30):
            selection_col = f"selected_top{int(round(frac * 100))}_{model}"
            selected = frame.loc[frame[selection_col].fillna(False).astype(bool)]
            ev = pd.to_numeric(selected[TARGET], errors="coerce")
            week = selected.assign(
                __week=pd.to_datetime(selected["__ts__"], utc=True).dt.strftime("%G-W%V")
            ).groupby("__week")[TARGET].mean()
            month = selected.assign(
                __month=pd.to_datetime(selected["__ts__"], utc=True).dt.strftime("%Y-%m")
            ).groupby("__month")[TARGET].mean()
            rows.append(
                {
                    "group": group,
                    "key": key,
                    "model": model,
                    "top_frac": frac,
                    "available_rows": int(len(frame)),
                    "selected_rows": int(len(selected)),
                    "mean_net_ev": float(ev.mean()),
                    "sum_net_ev": float(ev.sum()),
                    "positive_ev_rate": float((ev > 0.0).mean()),
                    "stop_rate": float(
                        pd.to_numeric(
                            selected.get("__first_touch_stop__"), errors="coerce"
                        ).mean()
                    ),
                    "timeout_rate": float(
                        pd.to_numeric(
                            selected.get("__first_touch_timeout__"), errors="coerce"
                        ).mean()
                    ),
                    "worst_week_mean_ev": float(week.min()),
                    "worst_month_mean_ev": float(month.min()),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilt-ledger", type=Path, required=True)
    parser.add_argument("--previous-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    target_columns = [
        *KEYS,
        "score",
        "selected_top10",
        "selected_top20",
        "selected_top30",
        TARGET,
        "__first_touch_stop__",
        "__first_touch_timeout__",
    ]
    rebuilt = pd.read_parquet(args.rebuilt_ledger, columns=target_columns).rename(
        columns={
            "score": "score_rebuilt",
            "selected_top10": "selected_top10_rebuilt",
            "selected_top20": "selected_top20_rebuilt",
            "selected_top30": "selected_top30_rebuilt",
        }
    )
    previous = pd.read_parquet(
        args.previous_ledger,
        columns=[*KEYS, "score", "selected_top10", "selected_top20", "selected_top30"],
    ).rename(
        columns={
            "score": "score_previous",
            "selected_top10": "selected_top10_previous",
            "selected_top20": "selected_top20_previous",
            "selected_top30": "selected_top30_previous",
        }
    )
    for frame in (rebuilt, previous):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    rebuilt = _compact_join_keys(rebuilt)
    previous = _compact_join_keys(previous).drop(columns=["__ts__"])
    joined = rebuilt.merge(
        previous,
        on=["__join_key_0", "__join_key_1"],
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    joined = joined.dropna(subset=[TARGET, "score_rebuilt", "score_previous"])
    rows = _metric_rows(joined, group="overall", key="all")
    month = pd.to_datetime(joined["__ts__"], utc=True).dt.strftime("%Y-%m")
    for value, idx in month.groupby(month).groups.items():
        rows.extend(_metric_rows(joined.loc[idx], group="month", key=str(value)))
    week = pd.to_datetime(joined["__ts__"], utc=True).dt.strftime("%G-W%V")
    for value, idx in week.groupby(week).groups.items():
        rows.extend(_metric_rows(joined.loc[idx], group="week", key=str(value)))

    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "corrected_base_comparison.csv", index=False)
    summary = metrics.loc[metrics["group"].eq("overall")].copy()
    pivot = summary.pivot(index="top_frac", columns="model", values="mean_net_ev")
    pivot["rebuilt_minus_previous"] = pivot["rebuilt"] - pivot["previous"]
    pivot.reset_index().to_csv(args.output_dir / "corrected_base_gate.csv", index=False)
    top10 = pivot.loc[0.10]
    gate_pass = bool(top10["rebuilt"] > top10["previous"])
    manifest = {
        "rebuilt_ledger": str(args.rebuilt_ledger),
        "previous_ledger": str(args.previous_ledger),
        "joined_rows": int(len(joined)),
        "join_contract": "collision-checked 128-bit hash of UTC timestamp/symbol/side",
        "target_source": "rebuilt corrected causal labels",
        "ranking_basis": (
            "each model's persisted fold-level selected_top10/20/30 flags on "
            "identical joined OOS rows"
        ),
        "primary_gate": "rebuilt top10 mean net EV > previous top10 mean net EV",
        "gate_pass": gate_pass,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
