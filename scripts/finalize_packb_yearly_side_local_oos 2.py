#!/usr/bin/env python3
"""Join monthly raw Pack-B OOS scores and apply the causal 21-day admission."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_packb_yearly_side_local_oos import (
    DECISION, MAP_DAYS, MAP_THRESHOLD, MIN_MAP_ROWS, NET_TARGET, SIDES, _robust_map,
)


def _day_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["side_name", "meta_expected_net_return", "candidate_id"], ascending=[True, False, True], kind="mergesort").copy()
    rank = ordered.groupby("side_name", sort=False).cumcount() + 1
    count = ordered.groupby("side_name", sort=False).candidate_id.transform("size")
    ordered["meta_score_percentile_side_day"] = 1. - (rank - 1.) / np.maximum(count - 1., 1.)
    return ordered


def _update_metrics(store: dict[tuple[str, pd.Timestamp, str], dict[str, object]], frame: pd.DataFrame) -> None:
    for frequency, label in (("W-SUN", "weekly"), ("M", "monthly")):
        period = frame[DECISION].dt.to_period(frequency).dt.start_time.dt.tz_localize("UTC")
        for start, local in frame.groupby(period, sort=False):
            for side in ("global", *SIDES):
                part = local if side == "global" else local[local.side_name.eq(side)]
                key = (label, start, side)
                stat = store.setdefault(key, {"candidate_rows": 0, "admitted_trades": 0, "mapped_sum": 0., "net_sum": 0., "score": [], "target": []})
                selected = part[part.admitted_21d_ev_ge_0p5pct]
                stat["candidate_rows"] += len(part)
                stat["admitted_trades"] += len(selected)
                stat["mapped_sum"] += float(selected.mapped_21d_ev_net_return.sum())
                stat["net_sum"] += float(selected[NET_TARGET].sum())
                stat["score"].append(part.meta_expected_net_return.to_numpy(np.float32))
                stat["target"].append(part[NET_TARGET].to_numpy(np.float32))


def _write_metrics(store: dict[tuple[str, pd.Timestamp, str], dict[str, object]], output: Path) -> None:
    rows: dict[str, list[dict[str, object]]] = {"weekly": [], "monthly": []}
    for (label, start, side), stat in sorted(store.items()):
        n = int(stat["candidate_rows"])
        selected = int(stat["admitted_trades"])
        score = np.concatenate(stat["score"])
        target = np.concatenate(stat["target"])
        rows[label].append({"period_start": start, "side": side, "candidate_rows": n, "admitted_trades": selected, "admission_rate": selected / n if n else np.nan, "mapped_ev_bps": stat["mapped_sum"] / selected * 1e4 if selected else np.nan, "realised_net_bps": stat["net_sum"] / selected * 1e4 if selected else np.nan, "realised_net_sum_bps": stat["net_sum"] * 1e4, "raw_score_ic": pd.Series(score).corr(pd.Series(target), method="spearman") if len(score) > 2 else np.nan})
    for label, result in rows.items():
        pd.DataFrame(result).to_parquet(output / f"{label}_metrics.parquet", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monthly-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default="2025-08-01")
    p.add_argument("--end", default="2026-07-11")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    parts = sorted(a.monthly_root.glob("*/raw_oos_predictions.parquet"))
    if len(parts) != 12:
        raise ValueError(f"require exactly twelve monthly raw ledgers, found {len(parts)}")
    start, end = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC")
    a.out.mkdir(parents=True)
    prediction_root = a.out / "oos_predictions"
    prediction_root.mkdir()
    history = pd.DataFrame(columns=["side_name", "__label_available_at__", "meta_score_percentile_side_day", NET_TARGET])
    metrics: dict[tuple[str, pd.Timestamp, str], dict[str, object]] = {}
    seen: set[str] = set()
    total, accepted = 0, 0
    for path in parts:
        raw = pd.read_parquet(path)
        raw[DECISION] = pd.to_datetime(raw[DECISION], utc=True)
        raw["__label_available_at__"] = pd.to_datetime(raw["__label_available_at__"], utc=True)
        month = raw[DECISION].dt.to_period("M").astype(str).unique().tolist()
        if len(month) != 1 or month[0] in seen:
            raise ValueError(f"invalid/duplicate monthly raw ledger: {path}")
        seen.add(month[0])
        mapped_parts: list[pd.DataFrame] = []
        for day, current in raw.groupby(raw[DECISION].dt.normalize(), sort=True):
            current = _day_percentile(current)
            current["mapped_21d_ev_net_return"] = np.nan
            current["admitted_21d_ev_ge_0p5pct"] = False
            for side in SIDES:
                local = current[current.side_name.eq(side)]
                reference = history[(history.side_name.eq(side)) & (history["__label_available_at__"].lt(day)) & (history["__label_available_at__"].ge(day - pd.Timedelta(days=MAP_DAYS)))]
                if len(reference) < MIN_MAP_ROWS:
                    continue
                current.loc[local.index, "mapped_21d_ev_net_return"] = _robust_map(reference, local)
                current.loc[local.index, "admitted_21d_ev_ge_0p5pct"] = current.loc[local.index, "mapped_21d_ev_net_return"].ge(MAP_THRESHOLD)
            mapped_parts.append(current)
            history = pd.concat([history, current[["side_name", "__label_available_at__", "meta_score_percentile_side_day", NET_TARGET]]], ignore_index=True)
            history = history[history["__label_available_at__"].ge(day - pd.Timedelta(days=MAP_DAYS))].copy()
        mapped = pd.concat(mapped_parts, ignore_index=True)
        if mapped.candidate_id.duplicated().any() or mapped[DECISION].min() < start or mapped[DECISION].max() >= end:
            raise ValueError(f"raw ledger has invalid OOS boundaries: {path}")
        mapped.to_parquet(prediction_root / f"{month[0]}.parquet", index=False)
        _update_metrics(metrics, mapped)
        total += len(mapped); accepted += int(mapped.admitted_21d_ev_ge_0p5pct.sum())
    expected = [str(value) for value in pd.period_range("2025-08", "2026-07", freq="M")]
    if sorted(seen) != expected:
        raise ValueError(f"missing/non-contiguous reporting months: {sorted(seen)}")
    _write_metrics(metrics, a.out)
    manifest = {
        "schema": "packb_yearly_side_local_oos_final_v1", "status": "materialized",
        "window": {"start": str(start), "end_exclusive": str(end)},
        "monthly_raw_ledgers": [str(path) for path in parts],
        "admission": {"side_local": True, "robust_ev_mapping_days": MAP_DAYS, "estimator": "5% winsorised conditional bin mean + isotonic", "net_ev_threshold": MAP_THRESHOLD, "target": NET_TARGET},
        "rows": {"oos_predictions": total, "accepted": accepted}, "prediction_format": "one parquet partition per reporting month under oos_predictions/",
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
